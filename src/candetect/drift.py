"""Input drift indicator: population stability index on simple image statistics.

The reference is a sample of the training distribution and the batch is whatever the camera sends
now. Brightness, contrast and sharpness catch the usual line problems (lighting, dirty lens,
focus) without needing labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("brightness", "contrast", "sharpness")
# Conventional PSI cut-offs: below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 major shift.
WARN_PSI = 0.10
ALERT_PSI = 0.25
_EPS = 1e-4


def image_features(images: np.ndarray) -> pd.DataFrame:
    """Per-image brightness (mean), contrast (std) and sharpness (mean absolute gradient)."""
    gray = images.astype(np.float64).mean(axis=3) / 255.0
    dx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    dy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    return pd.DataFrame(
        {
            "brightness": gray.mean(axis=(1, 2)),
            "contrast": gray.std(axis=(1, 2)),
            "sharpness": (dx + dy) / 2,
        }
    )


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index with bin edges at the quantiles of ``expected``."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(expected, edges)[0] / len(expected)
    q = np.histogram(actual, edges)[0] / len(actual)
    p, q = np.clip(p, _EPS, None), np.clip(q, _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def status_for(value: float) -> str:
    if value >= ALERT_PSI:
        return "drift"
    return "warn" if value >= WARN_PSI else "ok"


def drift_report(reference: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """One row per feature with its PSI and status."""
    rows = []
    for feature in FEATURES:
        value = psi(reference[feature].to_numpy(), batch[feature].to_numpy())
        rows.append({"feature": feature, "psi": value, "status": status_for(value)})
    return pd.DataFrame(rows)


def overall_status(report: pd.DataFrame) -> str:
    """The worst status across features."""
    order = {"ok": 0, "warn": 1, "drift": 2}
    return max(report["status"], key=order.__getitem__)
