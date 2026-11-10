import numpy as np
import pytest

from candetect.data import DataConfig, generate_dataset, shifted_config
from candetect.drift import (
    ALERT_PSI,
    WARN_PSI,
    drift_report,
    image_features,
    overall_status,
    psi,
    status_for,
)


@pytest.fixture(scope="module")
def reference():
    return image_features(generate_dataset(DataConfig(n_images=200, seed=11)).images)


def test_features_have_expected_shape_and_range(reference):
    assert list(reference.columns) == ["brightness", "contrast", "sharpness"]
    assert len(reference) == 200
    assert reference["brightness"].between(0, 1).all()


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000)
    small = psi(base, rng.normal(0.2, 1, size=2000))
    large = psi(base, rng.normal(2.0, 1, size=2000))
    assert 0 <= small < large
    assert large > ALERT_PSI


def test_constant_reference_does_not_crash():
    assert psi(np.ones(50), np.ones(50)) == 0.0


def test_status_thresholds():
    assert status_for(WARN_PSI - 1e-6) == "ok"
    assert status_for(WARN_PSI) == "warn"
    assert status_for(ALERT_PSI) == "drift"


def test_same_distribution_is_not_flagged(reference):
    fresh = image_features(generate_dataset(DataConfig(n_images=200, seed=12)).images)
    assert overall_status(drift_report(reference, fresh)) in {"ok", "warn"}
    assert drift_report(reference, fresh)["psi"].max() < ALERT_PSI


def test_deliberately_shifted_batch_is_flagged(reference):
    shifted = generate_dataset(shifted_config(DataConfig(n_images=200, seed=11), n_images=200))
    report = drift_report(reference, image_features(shifted.images))
    assert overall_status(report) == "drift"
    assert (report["status"] == "drift").sum() >= 2
