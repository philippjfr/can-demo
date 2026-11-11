from types import SimpleNamespace

from candetect.report import MARKER, build_report, run_state, verdict


def row(run="argo-x-1234567890", f1=0.99, dataset="aaaa", gate="ok", reasons=(), parity=True, **kw):
    base = {
        "run": run,
        "pathspec": f"TrainCanDetectorFlow/{run}",
        "f1": f1,
        "accuracy": f1,
        "precision": f1,
        "recall": f1,
        "arch": "mobilenet_v3_small",
        "epochs": 6,
        "images": 2000,
        "dataset": dataset,
        "parity": parity,
        "gate": gate,
        "gate_reasons": list(reasons),
    }
    return {**base, **kw}


def task(finished, successful):
    return SimpleNamespace(finished=finished, successful=successful)


def make_run(successful, task_states):
    class R:
        pass

    r = R()
    r.successful = successful
    r.__class__.__iter__ = lambda self: iter([iter([task(*t) for t in task_states])])
    return r


def test_run_state_succeeded_failed_running():
    assert run_state(make_run(True, [(True, True)])) == "succeeded"
    assert run_state(make_run(False, [(True, True), (True, False)])) == "failed"
    assert run_state(make_run(False, [(True, True), (False, False)])) == "running"


def test_report_shows_deltas_and_marker():
    text = build_report(row(f1=0.95), row(run="prod", f1=0.99), min_f1=0.9, branch="pr7")
    assert text.startswith(MARKER)
    assert "-0.0400" in text and "**worse**" in text
    assert "test.pr7" in text


def test_better_worse_and_noise_band():
    prod = row(f1=0.99)
    assert "**better**" in " ".join(verdict(row(f1=0.995), prod, 0.9))
    assert "**worse**" in " ".join(verdict(row(f1=0.97), prod, 0.9))
    assert "no meaningful change" in " ".join(verdict(row(f1=0.9905), prod, 0.9))


def test_different_data_is_flagged_not_compared_blindly():
    lines = " ".join(verdict(row(dataset="bbbb"), row(dataset="aaaa"), 0.9))
    assert "Different data versions" in lines


def test_gate_failure_reasons_are_shown():
    bad = row(f1=0.5, gate="refused", reasons=["test F1 0.5000 is below the threshold 0.9000"])
    assert "**fails**" in " ".join(verdict(bad, None, 0.9))


def test_no_production_run():
    text = build_report(row(), None, min_f1=0.9, branch="pr1")
    assert "nothing to compare with" in text
    assert "Production none" in text


def test_links_are_included_when_a_ui_url_is_known():
    text = build_report(row(), row(run="prod"), min_f1=0.9, branch="pr1", pr_url="https://ui/x/1")
    assert "(https://ui/x/1)" in text


def test_short_id_keeps_numbers_and_shortens_argo_ids():
    from candetect.report import short_id

    assert short_id("73007") == "73007"
    assert short_id("argo-candeondemo.test.pr7.traincatorflow-dapqv-9mt94") == "dapqv-9mt94"


def test_short_id_for_production_ids_has_a_single_hash():
    from candetect.report import short_id

    assert short_id("argo-candetectiondemo.prod.traincandetectorflow-kj856") == "kj856"
