import numpy as np
import pytest

from candetect import train as train_mod
from candetect.model import ARCH_MOBILENET, ARCH_SMALL_CNN
from candetect.train import TrainConfig, confusion_matrix, metrics_from_confusion, train


def test_confusion_and_metrics():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0, 1, 1, 1, 0, 1])
    cm = confusion_matrix(y, p)
    assert cm.tolist() == [[1, 1], [1, 3]]
    m = metrics_from_confusion(cm)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_metrics_handle_no_positive_predictions():
    m = metrics_from_confusion(np.array([[5, 0], [5, 0]]))
    assert m["f1"] == 0.0 and m["precision"] == 0.0


def test_small_cnn_trains_end_to_end(small_dataset, small_splits):
    cfg = TrainConfig(arch=ARCH_SMALL_CNN, pretrained=False, epochs=2, batch_size=32, num_threads=2)
    result = train(small_dataset, small_splits, cfg, log=lambda _: None)
    assert result.arch == ARCH_SMALL_CNN and not result.fallback_used
    assert len(result.history) == 2
    assert sum(sum(r) for r in result.confusion_matrix) == len(small_splits.test)


def test_falls_back_to_small_cnn_when_download_fails(monkeypatch, small_dataset, small_splits):
    real = train_mod.build_model

    def failing(arch, pretrained=True):
        if pretrained:
            raise OSError("no network")
        return real(arch, pretrained)

    monkeypatch.setattr(train_mod, "build_model", failing)
    logs = []
    cfg = TrainConfig(epochs=1, batch_size=32, num_threads=2)
    result = train(small_dataset, small_splits, cfg, log=logs.append)
    assert result.fallback_used and result.arch == ARCH_SMALL_CNN and not result.pretrained
    assert any("WARNING" in line for line in logs)


def test_no_fallback_raises(monkeypatch, small_dataset, small_splits):
    def failing(*args, **kwargs):
        raise OSError("x")

    monkeypatch.setattr(train_mod, "build_model", failing)
    cfg = TrainConfig(arch=ARCH_MOBILENET, fallback_to_cnn=False, epochs=1)
    with pytest.raises(OSError):
        train(small_dataset, small_splits, cfg, log=lambda _: None)
