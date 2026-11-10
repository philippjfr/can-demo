"""Models and safetensors serialisation.

Both architectures take float32 images in [0, 1] (NCHW) and normalise internally, so the exported
ONNX graph needs no preprocessing beyond ``/ 255``.
"""

from __future__ import annotations

import json
import struct

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .data import CLASS_NAMES

ARCH_MOBILENET = "mobilenet_v3_small"
ARCH_SMALL_CNN = "small_cnn"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CanClassifier(nn.Module):
    """Backbone plus fixed input normalisation. Output is raw logits for ``CLASS_NAMES``."""

    def __init__(self, backbone: nn.Module, arch: str):
        super().__init__()
        self.arch = arch
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone((image - self.mean) / self.std)


def _small_cnn(num_classes: int) -> nn.Module:
    def block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(3, 16),
        block(16, 32),
        block(32, 64),
        block(64, 96),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(96, num_classes),
    )


def build_model(arch: str = ARCH_MOBILENET, pretrained: bool = True) -> CanClassifier:
    """Build a classifier. With ``pretrained=True`` this downloads ImageNet weights and may raise
    ``OSError`` or ``RuntimeError`` when the download fails."""
    n = len(CLASS_NAMES)
    if arch == ARCH_SMALL_CNN:
        return CanClassifier(_small_cnn(n), arch)
    if arch != ARCH_MOBILENET:
        raise ValueError(f"unknown arch {arch!r}")
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    backbone = mobilenet_v3_small(weights=weights)
    head = backbone.classifier[-1]
    backbone.classifier[-1] = nn.Linear(head.in_features, n)
    return CanClassifier(backbone, arch)


def weights_to_bytes(model: CanClassifier, metadata: dict[str, str] | None = None) -> bytes:
    """Serialise to safetensors bytes. The header records the architecture."""
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    meta = {"arch": model.arch, "classes": ",".join(CLASS_NAMES), **(metadata or {})}
    return st_save(tensors, metadata=meta)


def read_metadata(data: bytes) -> dict[str, str]:
    """Read the safetensors header metadata without touching tensor data."""
    (n,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + n]).get("__metadata__", {})


def model_from_bytes(data: bytes) -> CanClassifier:
    """Rebuild a model from safetensors bytes. Never unpickles anything."""
    arch = read_metadata(data)["arch"]
    model = build_model(arch, pretrained=False)
    model.load_state_dict(st_load(data))
    return model.eval()
