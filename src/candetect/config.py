"""Deployment-specific settings, read from environment variables.

Values are loaded from ``config/demo.env`` by pixi (config/activate.sh) and by the GitHub workflows.
Nothing here is secret.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# @project names allow only lowercase letters, digits and underscores.
PROJECT_NAME = "can_detection_demo"
TRAIN_FLOW = "TrainCanDetectorFlow"
BUILD_FLOW = "BuildEdgeImageFlow"
PRODUCTION_TAG = "production"
PROMOTION_EVENT = "candetect_model_promoted"
# Tensor names of the exported ONNX graph.
INPUT_NAME = "image"
OUTPUT_NAME = "logits"


def project_branch(git_branch: str) -> str:
    """The branch name obproject-deploy gives a git branch: ``-`` and ``/`` become ``_``, lowercase.

    ``main`` keeps its name (it deploys as the production branch, ``prod``, in Metaflow terms).
    """
    if git_branch in ("main", "master"):
        return git_branch
    return re.sub(r"[-/]", "_", git_branch).lower()


@dataclass(frozen=True)
class Settings:
    metaflow_env: str
    conda_channels: tuple[str, ...]
    min_f1: float
    approver: str
    base_image: str
    registry: str
    github_repo: str
    perimeter_experiment: str
    perimeter_prod: str


def settings() -> Settings:
    def env(key: str, default: str = "") -> str:
        # CI passes unset repository variables as empty strings.
        return os.environ.get(key) or default

    channels = env("CANDETECT_CONDA_CHANNELS", "https://repo.anaconda.com/pkgs/main")
    return Settings(
        metaflow_env=env("CANDETECT_METAFLOW_ENV", "fast-bakery"),
        conda_channels=tuple(c for c in channels.split(",") if c),
        min_f1=float(env("CANDETECT_MIN_F1", "0.90")),
        approver=env("CANDETECT_APPROVER", ""),
        base_image=env("CANDETECT_BASE_IMAGE", "python:3.12-slim-bookworm"),
        registry=env("CANDETECT_REGISTRY", ""),
        github_repo=env("CANDETECT_GITHUB_REPO", ""),
        perimeter_experiment=env("CANDETECT_PERIMETER_EXPERIMENT", "default"),
        perimeter_prod=env("CANDETECT_PERIMETER_PROD", "default"),
    )
