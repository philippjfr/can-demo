"""The demo steps are pixi tasks: guard against one going missing or pointing at a moved file."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
CONFIG = tomllib.loads((ROOT / "pixi.toml").read_text())
TASKS = CONFIG["tasks"]

DEMO_STEPS = {
    "setup", "data", "train-local", "run-remote", "load-local", "compare", "promote",
    "build-image", "deploy-dashboard", "reset",
}  # fmt: skip


def test_every_demo_step_has_a_task_with_a_description():
    assert DEMO_STEPS <= set(TASKS)
    for name in DEMO_STEPS:
        assert TASKS[name]["description"], name


PATH_RE = re.compile(r"\b((?:scripts|flows|notebooks|deployments)/[\w./-]+\.\w+)")


def test_tasks_reference_files_that_exist():
    found = 0
    for name, task in TASKS.items():
        for path in PATH_RE.findall(task["cmd"]):
            found += 1
            assert (ROOT / path).exists(), f"{name} refers to {path}"
    assert found >= 8, "the path pattern stopped matching task commands"


def test_promote_requires_a_run_and_compare_does_not():
    assert TASKS["promote"]["args"] == [{"arg": "run"}]
    assert TASKS["compare"]["args"][0]["default"] == ""


def test_candetect_is_imported_from_src_and_demo_env_is_loaded():
    assert CONFIG["activation"]["env"]["PYTHONPATH"].endswith("/src")
    assert (ROOT / CONFIG["activation"]["scripts"][0]).exists()
