"""Promote a run to production.

    python scripts/promote.py 73005          # gate, tag, publish event, start the image build
    python scripts/promote.py --build-only   # redo the build for the current production run
    python scripts/promote.py --clear        # remove production and approval tags (demo reset)

The gate refuses runs below the F1 threshold or without a passing ONNX parity check. On success
the run gets the `production` tag plus `approved_by`, `approved_at` and `eval_f1` tags, the previous
production run loses its tag, and an event starts BuildEdgeImageFlow.
"""

import argparse
import shutil
import subprocess
import sys

from metaflow import Flow, Run, namespace
from metaflow.util import get_username

from candetect.config import PRODUCTION_TAG, TRAIN_FLOW, settings
from candetect.promotion import (
    APPROVED_AT_PREFIX,
    APPROVER_PREFIX,
    EVAL_F1_PREFIX,
    PromotionRefused,
    dispatch_build_workflow,
    promote_run,
    publish_promotion_event,
)


def production_runs() -> list[Run]:
    return list(Flow(TRAIN_FLOW).runs(PRODUCTION_TAG))


def start_build(run: Run, approver: str, branch: str, build_workflow: bool) -> None:
    data = run.data
    name = publish_promotion_event(
        run.pathspec, data.model_version, data.dataset_hash, approver, branch
    )
    print(f"published event {name}")
    cfg = settings()
    if not build_workflow:
        return
    if not cfg.github_repo or cfg.github_repo.startswith("OWNER/"):
        print("CANDETECT_GITHUB_REPO is not set, so the GitHub build was not started.")
        print("Set it in config/demo.env and run `pixi run build-image`.")
        return
    if not shutil.which("gh"):
        print("The gh CLI is not installed, so the GitHub build was not started.")
        return
    try:
        cmd = dispatch_build_workflow(
            cfg.github_repo, run.pathspec, data.model_version, data.dataset_hash, cfg.base_image
        )
    except subprocess.CalledProcessError as err:
        print(f"gh workflow run failed ({err}).")
        print("Check `gh auth status` and that build-image.yml exists on the default branch.")
        raise SystemExit(4) from err
    print("started GitHub workflow:", " ".join(cmd[:6]), "...")


def cmd_promote(args) -> int:
    namespace(None)
    cfg = settings()
    approver = args.approver or cfg.approver or get_username()
    min_f1 = args.min_f1 if args.min_f1 is not None else cfg.min_f1
    run = Run(args.run if "/" in args.run else f"{TRAIN_FLOW}/{args.run}")
    data = run.data
    print(f"Run {run.pathspec}: F1 {data.metrics['f1']:.4f}, threshold {min_f1:.2f}")
    print(f"Approver: {approver}")
    if args.dry_run:
        print("dry run: no tags changed")
        return 0
    try:
        result = promote_run(run, production_runs(), approver, min_f1)
    except PromotionRefused as err:
        print("REFUSED: not promoted, no tags changed.")
        for reason in err.reasons:
            print(f"  - {reason}")
        return 1
    print("PROMOTED")
    print("  tags added:", ", ".join(result.tags_added))
    print("  retired production runs:", ", ".join(result.retired) or "none")
    if not args.no_event:
        start_build(run, approver, args.branch, not args.no_build)
    print()
    print("Next: watch BuildEdgeImageFlow and the build-image workflow, then deploy the dashboard.")
    return 0


def cmd_build_only(args) -> int:
    namespace(None)
    current = production_runs()
    if not current:
        print("No run has the production tag. Use `pixi run promote <run id>` first.")
        return 1
    run = current[0]
    approver = next(
        (t[len(APPROVER_PREFIX) :] for t in run.tags if t.startswith(APPROVER_PREFIX)), "unknown"
    )
    print(f"Rebuilding for production run {run.pathspec} (approved by {approver})")
    start_build(run, approver, args.branch, not args.no_build)
    return 0


def cmd_clear(_args) -> int:
    namespace(None)
    prefixes = (APPROVER_PREFIX, APPROVED_AT_PREFIX, EVAL_F1_PREFIX)
    cleared = 0
    for run in Flow(TRAIN_FLOW):
        stale = [t for t in run.user_tags if t == PRODUCTION_TAG or t.startswith(prefixes)]
        if stale:
            run.remove_tags(stale)
            cleared += 1
    print(f"cleared production and approval tags on {cleared} run(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", help="run id or pathspec to promote")
    parser.add_argument("--min-f1", type=float, default=None, help="override CANDETECT_MIN_F1")
    parser.add_argument("--approver", help="override the recorded approver")
    parser.add_argument("--branch", default="prod", help="@project branch of the build flow")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-event", action="store_true", help="tag only")
    parser.add_argument("--no-build", action="store_true", help="publish the event, skip GitHub")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    if args.clear:
        return cmd_clear(args)
    if args.build_only:
        return cmd_build_only(args)
    if not args.run:
        parser.error("a run id is required (or use --build-only / --clear)")
    return cmd_promote(args)


if __name__ == "__main__":
    sys.exit(main())
