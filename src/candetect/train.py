"""Training and evaluation. The notebook and the Metaflow flow both call ``train``."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .data import CLASS_NAMES, Dataset, Splits
from .model import ARCH_MOBILENET, ARCH_SMALL_CNN, CanClassifier, build_model


@dataclass(frozen=True)
class TrainConfig:
    arch: str = ARCH_MOBILENET
    pretrained: bool = True
    # Train the small CNN from scratch when ImageNet weights cannot be downloaded.
    fallback_to_cnn: bool = True
    epochs: int = 6
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7
    use_gpu: bool = False
    # None keeps the torch default. Lower it on a busy or shared machine.
    num_threads: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    model: CanClassifier
    arch: str
    pretrained: bool
    fallback_used: bool
    history: list[dict]
    val_metrics: dict
    test_metrics: dict
    confusion_matrix: list[list[int]]
    train_seconds: float


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Rows are the true class, columns the predicted class, ordered as ``CLASS_NAMES``."""
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Binary metrics with ``fallen_can`` as the positive class."""
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": int(cm.sum()),
    }


def _to_tensor(images_u8: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images_u8)).permute(0, 3, 1, 2)
    return x.to(device).float().div_(255.0)


def predict_proba(
    model: CanClassifier, images: np.ndarray, batch_size: int = 256, device="cpu"
) -> np.ndarray:
    """Softmax probabilities, shape (N, len(CLASS_NAMES))."""
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            logits = model(_to_tensor(images[i : i + batch_size], device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


def evaluate(
    model: CanClassifier, images: np.ndarray, labels: np.ndarray, device="cpu"
) -> tuple[dict, np.ndarray]:
    pred = predict_proba(model, images, device=device).argmax(1)
    cm = confusion_matrix(labels, pred)
    return metrics_from_confusion(cm), cm


def _build_with_fallback(cfg: TrainConfig, log: Callable[[str], None]):
    try:
        model = build_model(cfg.arch, cfg.pretrained)
        return model, cfg.pretrained and cfg.arch == ARCH_MOBILENET, False
    except (OSError, RuntimeError) as err:
        if not (cfg.fallback_to_cnn and cfg.pretrained and cfg.arch == ARCH_MOBILENET):
            raise
        log(f"WARNING: pretrained weights unavailable ({err!r}); training small_cnn from scratch")
        return build_model(ARCH_SMALL_CNN, pretrained=False), False, True


def train(
    dataset: Dataset,
    splits: Splits,
    cfg: TrainConfig | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Fine-tune on train, keep the best epoch by validation F1, report on the test split."""
    cfg = cfg or TrainConfig()
    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    model, pretrained, fallback = _build_with_fallback(cfg, log)
    model.to(device)
    log(f"arch={model.arch} pretrained={pretrained} device={device}")

    images, labels = dataset.images, dataset.labels
    y_all = torch.from_numpy(labels)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg.seed)

    best_f1, best_state, history = -1.0, None, []
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = splits.train[torch.randperm(len(splits.train), generator=gen).numpy()]
        total = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            x = _to_tensor(images[idx], device)
            # Top-down views have no preferred orientation, so flips are label preserving.
            flip_h = torch.rand(len(idx), 1, 1, 1, generator=gen).to(device) < 0.5
            flip_v = torch.rand(len(idx), 1, 1, 1, generator=gen).to(device) < 0.5
            x = torch.where(flip_h, x.flip(3), x)
            x = torch.where(flip_v, x.flip(2), x)
            loss = loss_fn(model(x), y_all[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val, _ = evaluate(model, images[splits.val], labels[splits.val], device)
        history.append(
            {
                "epoch": epoch,
                "loss": total / len(perm),
                "val_f1": val["f1"],
                "val_accuracy": val["accuracy"],
            }
        )
        log(f"epoch {epoch}/{cfg.epochs} loss={history[-1]['loss']:.4f} val_f1={val['f1']:.4f}")
        if val["f1"] >= best_f1:
            best_f1, best_state = val["f1"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.cpu().eval()
    val_metrics, _ = evaluate(model, images[splits.val], labels[splits.val])
    test_metrics, cm = evaluate(model, images[splits.test], labels[splits.test])
    log(f"test f1={test_metrics['f1']:.4f} accuracy={test_metrics['accuracy']:.4f}")
    return TrainResult(
        model=model,
        arch=model.arch,
        pretrained=pretrained,
        fallback_used=fallback,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        confusion_matrix=cm.tolist(),
        train_seconds=time.time() - started,
    )
