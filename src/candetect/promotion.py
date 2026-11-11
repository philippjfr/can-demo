"""Promotion gate and bookkeeping.

The decision logic is pure so it can be unit tested. ``promote_run`` touches runs only through
the Metaflow Client tag API, and ``publish_promotion_event`` is the only Argo Events call.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime

from metaflow.integrations import ArgoEvent
from metaflow.plugins.namespaced_events import _make_namespaced_event_name

from .config import PRODUCTION_TAG, PROJECT_NAME, PROMOTION_EVENT

APPROVER_PREFIX = "approved_by:"
APPROVED_AT_PREFIX = "approved_at:"
EVAL_F1_PREFIX = "eval_f1:"


class PromotionRefused(Exception):
    """Raised when a run does not meet the promotion criteria."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass(frozen=True)
class Gate:
    reasons: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return not self.reasons


def check_gate(
    *,
    successful: bool,
    metrics: dict | None,
    parity: dict | None,
    dataset_hash: str | None,
    min_f1: float,
) -> Gate:
    """Return the reasons a run may not be promoted. Empty reasons mean it may."""
    reasons: list[str] = []
    if not successful:
        reasons.append("run did not finish successfully")
    if not metrics or "f1" not in metrics:
        reasons.append("run has no evaluation metrics")
    elif metrics["f1"] < min_f1:
        reasons.append(f"test F1 {metrics['f1']:.4f} is below the threshold {min_f1:.4f}")
    if not parity or not parity.get("passed"):
        reasons.append("ONNX parity check did not pass")
    if not dataset_hash:
        reasons.append("run did not record a dataset hash")
    return Gate(reasons)


def approval_tags(approver: str, f1: float, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(UTC)
    return [
        PRODUCTION_TAG,
        f"{APPROVER_PREFIX}{approver}",
        f"{APPROVED_AT_PREFIX}{now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"{EVAL_F1_PREFIX}{f1:.4f}",
    ]


@dataclass
class PromotionResult:
    pathspec: str
    tags_added: list[str]
    retired: list[str]


def promote_run(run, production_runs, approver: str, min_f1: float) -> PromotionResult:
    """Apply the gate, tag ``run`` as production and retire the previous production runs.

    ``run`` and ``production_runs`` are Metaflow ``Run`` objects (or look-alikes in tests). The
    new tag is added before the old one is removed so a production model exists throughout.
    """
    data = run.data
    gate = check_gate(
        successful=run.successful,
        metrics=getattr(data, "metrics", None),
        parity=getattr(data, "parity", None),
        dataset_hash=getattr(data, "dataset_hash", None),
        min_f1=min_f1,
    )
    if not gate.allowed:
        raise PromotionRefused(gate.reasons)

    tags = approval_tags(approver, data.metrics["f1"])
    run.add_tags(tags)
    retired = []
    for previous in production_runs:
        if previous.pathspec != run.pathspec:
            previous.remove_tag(PRODUCTION_TAG)
            retired.append(previous.pathspec)
    return PromotionResult(run.pathspec, tags, retired)


def image_tag(model_version: str) -> str:
    """Docker tags allow letters, digits, underscore, dot and dash, up to 128 characters."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", model_version)[:128]


def branch_label(tags) -> str:
    """Project branch of a run as the project UI names it: laptop runs are ``scratch``, the
    production Metaflow branch ``prod`` is ``main``."""
    branch = next((t.split(":", 1)[1] for t in tags if t.startswith("project_branch:")), "")
    if branch.startswith("user."):
        return "scratch"
    return {"prod": "main"}.get(branch, branch.removeprefix("test."))


def summarize_run(run, min_f1: float) -> dict:
    """One row for ``scripts/compare.py``: key parameters, metrics, tags and gate status."""
    data = run.data
    metrics = getattr(data, "metrics", None) or {}
    gate = check_gate(
        successful=run.successful,
        metrics=metrics,
        parity=getattr(data, "parity", None),
        dataset_hash=getattr(data, "dataset_hash", None),
        min_f1=min_f1,
    )
    tags = set(run.tags)
    approvers = [t[len(APPROVER_PREFIX) :] for t in tags if t.startswith(APPROVER_PREFIX)]
    return {
        "run": run.id,
        "pathspec": run.pathspec,
        "branch": branch_label(run.tags),
        "created": run.created_at.strftime("%m-%d %H:%M"),
        "arch": getattr(data, "arch_used", "?"),
        "epochs": getattr(data, "epochs", None),
        "images": getattr(data, "n_images", None),
        "f1": metrics.get("f1"),
        "accuracy": metrics.get("accuracy"),
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "parity": bool((getattr(data, "parity", None) or {}).get("passed")),
        "dataset": (getattr(data, "dataset_hash", "") or "")[:10],
        "production": PRODUCTION_TAG in tags,
        "approved_by": ",".join(sorted(approvers)),
        "gate": "ok" if gate.allowed else "refused",
        "gate_reasons": gate.reasons,
    }


def promotion_event_name(branch: str = "prod") -> str:
    """Event name the build flow listens on, namespaced by @project branch."""
    return _make_namespaced_event_name(PROMOTION_EVENT, PROJECT_NAME, branch)


def publish_promotion_event(
    pathspec: str, model_version: str, dataset_hash: str, approver: str, branch: str = "prod"
) -> str:
    """Publish the event that starts ``BuildEdgeImageFlow``. Returns the event name."""
    name = promotion_event_name(branch)
    ArgoEvent(name).publish(
        payload={
            "run_id": pathspec.split("/")[-1],
            "pathspec": pathspec,
            "model_version": model_version,
            "dataset_hash": dataset_hash,
            "approved_by": approver,
        },
        ignore_errors=False,
    )
    return name


def dispatch_build_workflow(
    repo: str, pathspec: str, model_version: str, dataset_hash: str, base_image: str
) -> list[str]:
    """Start ``build-image.yml`` with the presenter's own ``gh`` login. Returns the command run."""
    cmd = [
        "gh", "workflow", "run", "build-image.yml", "-R", repo,
        "-f", f"pathspec={pathspec}",
        "-f", f"model_version={model_version}",
        "-f", f"dataset_hash={dataset_hash}",
        "-f", f"base_image={base_image}",
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return cmd
