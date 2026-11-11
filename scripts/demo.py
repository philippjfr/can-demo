"""Logic behind the pixi tasks that need more than one command.

    python scripts/demo.py setup | pr-preview | pr-clean | deploy-dashboard | reset
"""

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from candetect.config import project_branch

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def git_branch() -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


def project_name() -> str:
    return tomllib.loads((ROOT / "obproject.toml").read_text())["project"]


def drop_build_metadata() -> None:
    """src/*.egg-info would be packaged by Metaflow as a nameless distribution and break
    `import torch` in the pod. pixi does not create it; a pip install -e . elsewhere does."""
    for path in (ROOT / "src").glob("*.egg-info"):
        shutil.rmtree(path)


def setup(_args) -> None:
    env = ROOT / "config" / "demo.env"
    if not env.exists():
        shutil.copy(ROOT / "config" / "demo.env.example", env)
        print(f"created {env.relative_to(ROOT)}")
    run("outerbounds", "check")
    print("\nNext: edit config/demo.env if needed, then pixi run data.")


def pr_preview(args) -> None:
    branch = git_branch()
    if branch in ("", "main", "master"):
        sys.exit("Switch to a feature branch first: this deploys a branch copy.")
    drop_build_metadata()
    run("obproject-deploy", "--skip-apps", "--skip-assets")
    run(sys.executable, "-W", "ignore", "scripts/pr_report.py", "--train-args", args.train_args)
    print("\nNext: build/pr-report/ holds report.md (the PR comment) and both model cards as HTML.")
    print("      Clean up the branch deployment with pixi run pr-preview-clean.")


def pr_clean(_args) -> None:
    branch = git_branch()
    if branch in ("", "main", "master"):
        sys.exit("Refusing to tear down the production branch.")
    run(
        "outerbounds", "flowproject", "teardown-branch",
        "--id", f"{project_name()}/{project_branch(branch)}", "--yes",
    )  # fmt: skip


def deploy_dashboard(_args) -> None:
    branch = git_branch()
    if branch != "main":
        sys.exit(
            "The dashboard deploys from main only (deployments/monitor/obproject_deploy.toml). "
            f"You are on {branch or 'no branch'}."
        )
    drop_build_metadata()
    run("obproject-deploy", "--skip-flows", "--skip-assets")
    print(
        "\nNext: find the app URL with: outerbounds app info --name candetect-monitor "
        f"--project {project_name()} --branch main"
    )


def reset(_args) -> None:
    for name in ("artifacts", "data", "build"):
        shutil.rmtree(ROOT / name, ignore_errors=True)
    run(sys.executable, "-W", "ignore", "scripts/promote.py", "--clear")
    print("\nNext: pixi run data to start again.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup").set_defaults(func=setup)
    preview = sub.add_parser("pr-preview")
    preview.add_argument("--train-args", default="")
    preview.set_defaults(func=pr_preview)
    sub.add_parser("pr-clean").set_defaults(func=pr_clean)
    sub.add_parser("deploy-dashboard").set_defaults(func=deploy_dashboard)
    sub.add_parser("reset").set_defaults(func=reset)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
