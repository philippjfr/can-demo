"""ONNX export as one self-contained file, plus a PyTorch/onnxruntime parity check."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch

from .config import INPUT_NAME, OUTPUT_NAME
from .data import to_nchw_float
from .model import CanClassifier
from .onnx_check import assert_self_contained


def export_onnx(model: CanClassifier, size: int = 128) -> bytes:
    """Export to ONNX bytes with all weights embedded and a dynamic batch dimension."""
    model = model.eval().cpu()
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, size, size),),
        buf,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: "batch"},),
        dynamo=True,
        external_data=False,
    )
    data = buf.getvalue()
    assert_self_contained(data)
    return data


@dataclass(frozen=True)
class Parity:
    max_abs_diff: float
    argmax_agreement: float
    atol: float

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.atol and self.argmax_agreement == 1.0

    def to_dict(self) -> dict:
        return {
            "max_abs_diff": self.max_abs_diff,
            "argmax_agreement": self.argmax_agreement,
            "atol": self.atol,
            "passed": self.passed,
        }


def check_parity(
    model: CanClassifier, onnx_bytes: bytes, images: np.ndarray, atol: float = 1e-3
) -> Parity:
    """Compare logits from PyTorch and onnxruntime on real images (uint8 NHWC)."""
    x = to_nchw_float(images)
    with torch.no_grad():
        expected = model.eval().cpu()(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    return Parity(
        max_abs_diff=float(np.abs(expected - actual).max()),
        argmax_agreement=float((expected.argmax(1) == actual.argmax(1)).mean()),
        atol=atol,
    )
