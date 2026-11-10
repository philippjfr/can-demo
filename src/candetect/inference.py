"""Inference helpers shared by notebook 02, the dashboard and the flows.

``load_run`` pulls artifacts of a platform run through the Metaflow Client API. Weights arrive as
raw safetensors bytes and are never unpickled into a model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from metaflow import Flow, Run, namespace
from PIL import Image, ImageDraw

from .config import INPUT_NAME, OUTPUT_NAME, TRAIN_FLOW
from .data import to_nchw_float

# Short captions so they fit the 128 px tile width.
_SHORT = {0: "ok", 1: "fallen"}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class OnnxModel:
    """onnxruntime wrapper around the exported single-file model."""

    def __init__(self, onnx_bytes: bytes):
        self.session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Class probabilities for uint8 images of shape (N, H, W, 3)."""
        logits = self.session.run([OUTPUT_NAME], {INPUT_NAME: to_nchw_float(images)})[0]
        return softmax(logits)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images).argmax(1)


@dataclass
class LoadedRun:
    pathspec: str
    weights: bytes
    onnx: bytes
    dataset_hash: str
    data_config: dict
    metrics: dict
    model_version: str
    arch: str
    tags: list[str]

    def onnx_model(self) -> OnnxModel:
        return OnnxModel(self.onnx)


def latest_successful_run(flow_name: str = TRAIN_FLOW) -> str:
    """Pathspec of the newest finished run, in any namespace."""
    namespace(None)
    for run in Flow(flow_name):
        if run.successful:
            return run.pathspec
    raise LookupError(f"no successful {flow_name} run found; run `make run-remote` first")


def load_run(pathspec: str) -> LoadedRun:
    """Load a finished ``TrainCanDetectorFlow`` run, for example ``TrainCanDetectorFlow/123``.

    Production runs live outside the default user namespace, so the global namespace is used.
    """
    namespace(None)
    run = Run(pathspec)
    if not run.successful:
        raise ValueError(f"{pathspec} did not finish successfully")
    data = run.data
    return LoadedRun(
        pathspec=run.pathspec,
        weights=data.weights,
        onnx=data.onnx_model,
        dataset_hash=data.dataset_hash,
        data_config=data.data_config,
        metrics=data.metrics,
        model_version=data.model_version,
        arch=data.arch_used,
        tags=sorted(run.tags),
    )


def encode_png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def contact_sheet(
    images: np.ndarray, labels: np.ndarray, probs: np.ndarray, columns: int = 4
) -> bytes:
    """PNG grid of images with a border (green correct, red wrong) and a true/predicted caption."""
    n, h, w, _ = images.shape
    rows = -(-n // columns)
    cap = 14
    sheet = Image.new("RGB", (columns * w, rows * (h + cap)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i in range(n):
        x, y = (i % columns) * w, (i // columns) * (h + cap)
        pred = int(probs[i].argmax())
        sheet.paste(Image.fromarray(images[i]), (x, y + cap))
        ok = pred == int(labels[i])
        colour = (0, 200, 0) if ok else (230, 0, 0)
        draw.rectangle([x, y + cap, x + w - 1, y + cap + h - 1], outline=colour)
        text = f"T:{_SHORT[int(labels[i])]} P:{_SHORT[pred]} {probs[i].max():.2f}"
        draw.text((x + 2, y + 1), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()
