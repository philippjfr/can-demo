"""Markdown comparison of a pull request's training run with the production model."""

from __future__ import annotations

MARKER = "<!-- candetect-pr-report -->"
# F1 differences below this are noise: the same code and seed agree to about 1e-3 across hardware.
TOLERANCE = 1e-3
METRICS = (("f1", "F1"), ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"))


def short_id(run_id: str) -> str:
    """Numeric run ids unchanged. Argo ids are ``argo-<template>-<hash>`` on production and
    ``argo-<template>-<hash>-<hash>`` on test branches: keep the trailing hashes."""
    if run_id.isdigit() or "-" not in run_id:
        return run_id
    parts = run_id.split("-")[1:]
    return "-".join(parts[-2:]) if len(parts) >= 3 else parts[-1]


def run_state(run) -> str:
    """``succeeded``, ``failed`` or ``running``. A failed Argo run never gets an end task, so look
    for a finished task that did not succeed."""
    if run.successful:
        return "succeeded"
    for step in run:
        for task in step:
            if task.finished and not task.successful:
                return "failed"
    return "running"


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _delta(pr, prod) -> str:
    if pr is None or prod is None:
        return ""
    return f"{pr - prod:+.4f}"


def verdict(pr: dict, prod: dict | None, min_f1: float) -> list[str]:
    lines = []
    if pr["gate"] == "ok":
        lines.append(f"- Promotion gate: **passes** (test F1 {pr['f1']:.4f} against {min_f1:.2f}).")
    else:
        lines.append(f"- Promotion gate: **fails**, {'; '.join(pr['gate_reasons'])}.")
    if prod is None:
        lines.append("- No run is tagged `production` yet, so there is nothing to compare with.")
        return lines
    if pr["dataset"] != prod["dataset"]:
        lines.append("- **Different data versions**: the metrics are not directly comparable.")
    delta = pr["f1"] - prod["f1"]
    if delta > TOLERANCE:
        lines.append(f"- Against production: **better**, F1 {delta:+.4f}.")
    elif delta < -TOLERANCE:
        lines.append(f"- Against production: **worse**, F1 {delta:+.4f}.")
    else:
        lines.append(f"- Against production: **no meaningful change**, F1 {delta:+.4f}.")
    return lines


def build_report(
    pr: dict,
    prod: dict | None,
    *,
    min_f1: float,
    branch: str,
    pr_url: str | None = None,
    prod_url: str | None = None,
) -> str:
    """The pull request comment. ``pr`` and ``prod`` are ``summarize_run`` rows."""

    def head(row, url):
        label = f"`{short_id(row['run'])}`"
        return f"[{label}]({url})" if url else label

    prod_head = head(prod, prod_url) if prod else "none"
    rows = [
        f"| | This PR {head(pr, pr_url)} | Production {prod_head} | Change |",
        "| --- | --- | --- | --- |",
    ]
    for key, label in METRICS:
        rows.append(
            f"| {label} | {_fmt(pr[key])} | {_fmt(prod[key]) if prod else ''} "
            f"| {_delta(pr[key], prod[key]) if prod else ''} |"
        )
    for key, label in (("arch", "Architecture"), ("epochs", "Epochs"), ("images", "Images")):
        rows.append(f"| {label} | {pr[key]} | {prod[key] if prod else ''} | |")
    prod_hash = f"`{prod['dataset']}`" if prod else ""
    rows.append(f"| Dataset hash | `{pr['dataset']}` | {prod_hash} | |")
    rows.append(
        f"| ONNX parity | {'passed' if pr['parity'] else 'FAILED'} "
        f"| {('passed' if prod['parity'] else 'FAILED') if prod else ''} | |"
    )
    return "\n".join(
        [
            MARKER,
            "## Model comparison: pull request against production",
            "",
            f"Branch deployment `test.{branch}` trained `{pr['pathspec']}` with this PR's code.",
            "",
            *rows,
            "",
            *verdict(pr, prod, min_f1),
            "",
            "The full model cards of both runs are attached to the workflow run as the "
            "`model-cards` artifact" + (f" and the run is at {pr_url}." if pr_url else "."),
        ]
    )
