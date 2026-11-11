"""Train on a branch deployment and compare the result with the production model.

This is what ci.yml does for a pull request, after obproject-deploy has deployed the branch copy.
Run it from the project root.

    python scripts/pr_report.py                       # branch from GH_HEAD_REF or git
    python scripts/pr_report.py --train-args "--epochs 3"
    python scripts/pr_report.py --run TrainCanDetectorFlow/73007    # compare an existing run

It triggers the deployment, waits for the run, then writes report.md (the PR comment) and the
model cards of both runs as HTML into --out.
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from metaflow import Flow, Run, namespace
from metaflow.cards import get_cards
from metaflow.exception import MetaflowNotFound
from metaflow.metaflow_config import UI_URL

from candetect.config import PRODUCTION_TAG, TRAIN_FLOW, project_branch, settings
from candetect.promotion import summarize_run
from candetect.report import MARKER, build_report, run_state


def trigger(branch: str, env: str, train_args: str) -> str:
    """Start the branch deployment and return the run pathspec."""
    with tempfile.TemporaryDirectory() as tmp:
        run_id_file = Path(tmp) / "run_id"
        cmd = [
            sys.executable, "flows/train/flow.py", f"--environment={env}", "--branch", branch,
            "argo-workflows", "trigger", "--run-id-file", str(run_id_file),
            *shlex.split(train_args),
        ]  # fmt: skip
        subprocess.run(cmd, check=True)
        run_id = run_id_file.read_text().strip()
    return run_id if "/" in run_id else f"{TRAIN_FLOW}/{run_id}"


def wait(pathspec: str, timeout: int, poll: int = 15) -> str:
    """Block until the run succeeds or fails. Returns the final state."""
    namespace(None)
    started = time.time()
    while time.time() - started < timeout:
        try:
            state = run_state(Run(pathspec))
        except MetaflowNotFound:
            state = "running"  # Argo registers the run a few seconds after the trigger
        print(f"[{int(time.time() - started):4d}s] {pathspec}: {state}", flush=True)
        if state != "running":
            return state
        time.sleep(poll)
    return "timeout"


def card_html(run: Run, attempts: int = 12, delay: int = 5) -> str:
    """The run's summary card. It can appear a few seconds after the run is marked finished."""
    for _ in range(attempts):
        cards = get_cards(run["end"].task, id="summary")
        if len(cards):
            return cards[0].get()
        time.sleep(delay)
    raise RuntimeError(f"no summary card for {run.pathspec} after {attempts * delay}s")


def run_url(run) -> str | None:
    return f"{UI_URL.rstrip('/')}/{TRAIN_FLOW}/{run.id}" if UI_URL else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--branch",
        help="project branch of the PR deployment (default: GH_HEAD_REF or the current git branch)",
    )
    parser.add_argument("--run", help="compare this existing run instead of triggering one")
    parser.add_argument("--train-args", default="", help="extra flow parameters for the trigger")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds to wait for the run")
    parser.add_argument("--out", default="build/pr-report")
    args = parser.parse_args()
    git_branch = os.environ.get("GH_HEAD_REF") or subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True
    ).stdout.strip()
    args.branch = args.branch or (project_branch(git_branch) if git_branch else None)
    if not (args.branch or args.run):
        parser.error("not on a git branch: give --branch, or --run to compare an existing run")

    cfg = settings()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    branch = args.branch or "n/a"

    pathspec = args.run or trigger(args.branch, cfg.metaflow_env, args.train_args)
    if "/" not in pathspec:
        pathspec = f"{TRAIN_FLOW}/{pathspec}"
    print(f"training run: {pathspec}", flush=True)
    state = "succeeded" if args.run else wait(pathspec, args.timeout)
    if state != "succeeded":
        body = (
            f"{MARKER}\n## Model comparison: pull request against production\n\n"
            f"The training run `{pathspec}` on branch deployment `test.{branch}` {state}. "
            "Open the run in the Outerbounds UI for its logs."
        )
        (out / "report.md").write_text(body)
        print(body)
        return 1

    namespace(None)
    pr_run = Run(pathspec)
    prod_run = next(iter(Flow(TRAIN_FLOW).runs(PRODUCTION_TAG)), None)
    report = build_report(
        summarize_run(pr_run, cfg.min_f1),
        summarize_run(prod_run, cfg.min_f1) if prod_run else None,
        min_f1=cfg.min_f1,
        branch=branch,
        pr_url=run_url(pr_run),
        prod_url=run_url(prod_run) if prod_run else None,
    )
    (out / "report.md").write_text(report)
    (out / "pr-run-card.html").write_text(card_html(pr_run))
    if prod_run:
        (out / "production-card.html").write_text(card_html(prod_run))
    print("\n" + report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    print(f"\nwrote {out}/ (report.md and the model cards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
