import subprocess
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from candetect import promotion
from candetect.config import PRODUCTION_TAG, PROJECT_NAME, PROMOTION_EVENT
from candetect.promotion import (
    PromotionRefused,
    approval_tags,
    check_gate,
    promote_run,
    promotion_event_name,
    summarize_run,
)

GOOD = {
    "successful": True,
    "metrics": {"f1": 0.95},
    "parity": {"passed": True},
    "dataset_hash": "abc",
    "min_f1": 0.90,
}


class FakeRun:
    def __init__(self, run_id, f1=0.95, parity=True, successful=True, tags=()):
        self.id = run_id
        self.pathspec = f"TrainCanDetectorFlow/{run_id}"
        self.successful = successful
        self.created_at = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
        self.tags = set(tags)
        self.log = []
        self.data = SimpleNamespace(
            metrics={"f1": f1, "accuracy": f1, "recall": f1},
            parity={"passed": parity},
            dataset_hash="deadbeefcafe1234",
            arch_used="mobilenet_v3_small",
            epochs=6,
            n_images=2000,
        )

    def add_tags(self, tags):
        self.log.append(("add", tuple(tags)))
        self.tags |= set(tags)

    def remove_tag(self, tag):
        self.log.append(("remove", tag))
        self.tags.discard(tag)


def test_gate_allows_good_run():
    assert check_gate(**GOOD).allowed


def test_gate_threshold_is_inclusive():
    assert check_gate(**{**GOOD, "metrics": {"f1": 0.90}}).allowed
    refused = check_gate(**{**GOOD, "metrics": {"f1": 0.8999}})
    assert not refused.allowed and "below the threshold" in refused.reasons[0]


@pytest.mark.parametrize(
    "override, fragment",
    [
        ({"successful": False}, "did not finish"),
        ({"metrics": None}, "no evaluation metrics"),
        ({"parity": {"passed": False}}, "parity"),
        ({"parity": None}, "parity"),
        ({"dataset_hash": ""}, "dataset hash"),
    ],
)
def test_gate_refusal_reasons(override, fragment):
    gate = check_gate(**{**GOOD, **override})
    assert not gate.allowed
    assert any(fragment in r for r in gate.reasons)


def test_gate_reports_every_failure():
    gate = check_gate(
        successful=False, metrics={"f1": 0.1}, parity=None, dataset_hash=None, min_f1=0.9
    )
    assert len(gate.reasons) == 4


def test_approval_tags_record_approver_and_time():
    now = datetime(2026, 11, 3, 13, 5, 9, tzinfo=UTC)
    tags = approval_tags("jane@example.com", 0.97123, now)
    assert tags == [
        PRODUCTION_TAG,
        "approved_by:jane@example.com",
        "approved_at:2026-11-03T13:05:09Z",
        "eval_f1:0.9712",
    ]


def test_refused_promotion_changes_no_tags():
    weak, previous = FakeRun("2", f1=0.5), FakeRun("1", tags=[PRODUCTION_TAG])
    with pytest.raises(PromotionRefused) as err:
        promote_run(weak, [previous], "jane", min_f1=0.9)
    assert "below the threshold" in str(err.value)
    assert weak.log == [] and previous.log == []
    assert PRODUCTION_TAG in previous.tags


def test_promotion_tags_run_and_retires_previous():
    new, previous = FakeRun("2"), FakeRun("1", tags=[PRODUCTION_TAG])
    result = promote_run(new, [previous], "jane", min_f1=0.9)
    assert PRODUCTION_TAG in new.tags and "approved_by:jane" in new.tags
    assert PRODUCTION_TAG not in previous.tags
    assert result.retired == [previous.pathspec]


def test_repromoting_same_run_keeps_its_tag():
    run = FakeRun("1", tags=[PRODUCTION_TAG])
    result = promote_run(run, [run], "jane", min_f1=0.9)
    assert PRODUCTION_TAG in run.tags and result.retired == []
    assert ("remove", PRODUCTION_TAG) not in run.log


def test_unsuccessful_run_is_refused():
    with pytest.raises(PromotionRefused):
        promote_run(FakeRun("3", successful=False), [], "jane", min_f1=0.9)


def test_event_name_matches_project_branch_namespacing():
    assert promotion_event_name() == f"mfns.{PROJECT_NAME}.prod.{PROMOTION_EVENT}"
    assert promotion_event_name("test.pr7").endswith(f".test.pr7.{PROMOTION_EVENT}")


def test_dispatch_builds_gh_command(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append((cmd, check)))
    cmd = promotion.dispatch_build_workflow("acme/cans", "Flow/9", "v9", "hash9", "debian:12")
    assert calls == [(cmd, True)]
    assert cmd[:6] == ["gh", "workflow", "run", "build-image.yml", "-R", "acme/cans"]
    for expected in ("pathspec=Flow/9", "dataset_hash=hash9", "base_image=debian:12"):
        assert expected in cmd


def test_summarize_run_flags_production_and_refusals():
    row = summarize_run(FakeRun("5", f1=0.97, tags=[PRODUCTION_TAG, "approved_by:jane"]), 0.9)
    assert row["production"] and row["approved_by"] == "jane" and row["gate"] == "ok"
    refused = summarize_run(FakeRun("6", f1=0.5), 0.9)
    assert refused["gate"] == "refused" and refused["gate_reasons"]


def test_image_tag_is_docker_safe():
    from candetect.promotion import image_tag

    assert image_tag("20261103-argo-flow/73007@x") == "20261103-argo-flow-73007-x"
    assert len(image_tag("a" * 300)) == 128


def test_branch_label_matches_the_project_ui():
    from candetect.promotion import branch_label

    assert branch_label({"project_branch:user.jane@example.com"}) == "scratch"
    assert branch_label({"project_branch:prod"}) == "main"
    assert branch_label({"project_branch:test.demo_fewer_epochs"}) == "demo_fewer_epochs"
    assert branch_label({"user:jane"}) == ""
