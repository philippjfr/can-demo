"""The repository must stay a valid Outerbounds project: obproject-deploy finds flows by regex."""

import re
import tomllib
from pathlib import Path

import yaml

from candetect import flowenv
from candetect.config import PROJECT_NAME, PROMOTION_EVENT, project_branch

ROOT = Path(__file__).parents[1]
# The same pattern obproject-deploy uses to discover flows.
FLOW_RE = re.compile(r"^class\s+(\w+)\s*\([^)]*?\bProjectFlow\b[^)]*\)\s*:", re.MULTILINE)


def test_obproject_toml_matches_config():
    conf = tomllib.loads((ROOT / "obproject.toml").read_text())
    assert conf["project"] == PROJECT_NAME
    assert conf["platform"] and conf["title"]
    assert conf["dependencies"]["include_pyproject_toml"] is False


def test_every_flow_is_discoverable_as_a_project_flow():
    flows = sorted((ROOT / "flows").glob("*/flow.py"))
    assert {f.parent.name for f in flows} == {"train", "build_image"}
    for flow in flows:
        assert FLOW_RE.search(flow.read_text()), f"{flow} has no ProjectFlow subclass"
        assert (flow.parent / "README.md").exists()


def test_flows_import_shared_code_from_the_package_not_a_sibling_module():
    for flow in (ROOT / "flows").glob("*/flow.py"):
        text = flow.read_text()
        assert "from _env" not in text and "import _env" not in text
        assert "@project(" not in text, "ProjectFlow already applies @project"


def test_promotion_event_name_is_shared():
    assert flowenv.PROMOTION_EVENT == PROMOTION_EVENT


def test_dashboard_deployment_is_valid_and_main_only():
    app = ROOT / "deployments" / "monitor"
    config = yaml.safe_load((app / "config.yml").read_text())
    assert config["name"] and config["port"]
    command = config["commands"][0]
    # Commands run from the project root, so paths must be relative to it.
    script = next(part for part in command.split() if part.endswith(".py"))
    assert (ROOT / script).exists()
    assert "torch" not in config["dependencies"]["pypi"]
    deploy = tomllib.loads((app / "obproject_deploy.toml").read_text())
    assert deploy["deploy"]["branches"] == ["main"]
    assert (app / "README.md").exists()


def test_branch_names_match_obproject_deploy():
    assert project_branch("main") == "main"
    assert project_branch("demo/fewer-epochs") == "demo_fewer_epochs"
    assert project_branch("Feature/Add-Scoring") == "feature_add_scoring"
