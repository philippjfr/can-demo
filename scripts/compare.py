"""Compare training runs side by side.

    python scripts/compare.py                 # newest successful runs
    python scripts/compare.py 73005 73007     # specific runs
"""

import argparse
import sys

import pandas as pd
from metaflow import Flow, Run, namespace

from candetect.config import TRAIN_FLOW, settings
from candetect.promotion import summarize_run
from candetect.report import short_id


def pick_runs(run_ids: list[str], limit: int) -> list[Run]:
    namespace(None)
    if run_ids:
        return [Run(r if "/" in r else f"{TRAIN_FLOW}/{r}") for r in run_ids]
    runs = []
    for run in Flow(TRAIN_FLOW):
        if run.successful and hasattr(run.data, "metrics"):
            runs.append(run)
        if len(runs) == limit:
            break
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", help="run ids or pathspecs; default: newest runs")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-f1", type=float, default=settings().min_f1)
    args = parser.parse_args()

    runs = pick_runs(args.runs, args.limit)
    if not runs:
        print("No successful runs found. Run `make run-remote` first.")
        return 1
    rows = [summarize_run(r, args.min_f1) for r in runs]
    frame = pd.DataFrame(rows).drop(columns=["gate_reasons", "pathspec", "precision", "parity"])
    frame["run"] = frame["run"].astype(str).map(short_id)
    frame["production"] = frame["production"].map({True: "yes", False: ""})
    for col in ("f1", "accuracy", "recall"):
        frame[col] = frame[col].map(lambda v: "" if v is None else f"{v:.4f}")
    print(f"Runs of {TRAIN_FLOW}, newest first. Promotion threshold: F1 >= {args.min_f1:.2f}\n")
    print(frame.to_string(index=False))

    eligible = [r for r in rows if r["gate"] == "ok"]
    print()
    if len({r["dataset"] for r in rows}) > 1:
        print("Note: these runs used different data versions (dataset column), so metrics are not")
        print("directly comparable.")
    if eligible:
        best = max(eligible, key=lambda r: r["f1"])
        print(f"Best eligible run: {best['run']} (F1 {best['f1']:.4f})")
    for r in rows:
        if r["gate"] == "refused":
            print(f"Run {r['run']} would be refused: {'; '.join(r['gate_reasons'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
