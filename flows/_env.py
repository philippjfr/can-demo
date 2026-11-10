"""Environment declaration shared by the flows.

Packages resolve from the channels in CANDETECT_CONDA_CHANNELS at deploy time (default: the public
Anaconda ``main`` channel). Versions are pinned to what exists on ``main`` for linux-64, the
platform's architecture.
"""

import os

from metaflow import anaconda_base

import candetect  # noqa: F401  Imported so Metaflow ships it (METAFLOW_PACKAGE_POLICY) with the flow.

PROJECT = "can_detection_demo"
PROMOTION_EVENT = "candetect_model_promoted"
PYTHON = "3.12"

# Deploy-time defaults for the build flow's parameters.
REGISTRY = os.environ.get("CANDETECT_REGISTRY") or ""
BASE_IMAGE = os.environ.get("CANDETECT_BASE_IMAGE") or "python:3.12-slim-bookworm"

# Set by `make flow-local` to run the flow in the active environment without building one.
LOCAL = os.environ.get("CANDETECT_LOCAL") == "1"

CHANNELS = [
    c
    for c in (os.environ.get("CANDETECT_CONDA_CHANNELS") or "https://repo.anaconda.com/pkgs/main").split(",")
    if c
]

PACKAGES = {
    "pytorch": "2.15.0",
    "torchvision": "0.29.2",
    "safetensors": "0.8.0",
    "onnx": "1.22.0",
    "onnxscript": "0.7.1",
    "onnxruntime": "1.30.0",
    "numpy": "2.5.4",
    "pillow": "13.0.0",
}


def conda_env(flow_class):
    """Declare the conda environment, unless CANDETECT_LOCAL=1 (the decorator would otherwise
    refuse to run without --environment=anaconda, fast-bakery, conda or pypi)."""
    if LOCAL:
        return flow_class
    return anaconda_base(python=PYTHON, packages=PACKAGES, channels=CHANNELS)(flow_class)


# onnxruntime builds request an executable stack, which glibc 2.41 (the platform's Debian 13 base
# image) refuses to dlopen. Verified on gtm-01: "force" mode makes the import work.
ONNXRUNTIME_ENV = {"GLIBC_TUNABLES": "glibc.rtld.execstack=2"}
