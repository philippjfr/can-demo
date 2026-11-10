"""Helpers for build-image.yml: fetch the promoted model and record the built image.

    python -m candetect.publish fetch TrainCanDetectorFlow/73007 build --expect-hash <sha256>
    python -m candetect.publish tag-image TrainCanDetectorFlow/73007 ghcr.io/acme/edge:v1
"""

import argparse
import hashlib
import json
from pathlib import Path

from metaflow import Run, namespace

from .config import PRODUCTION_TAG
from .inference import load_run
from .onnx_check import assert_self_contained
from .promotion import image_tag


def fetch(
    pathspec: str, out_dir: str, require_tag: str | None = PRODUCTION_TAG, expect_hash: str = ""
) -> dict:
    """Write ``model.onnx`` and ``metadata.json`` for a run, refusing runs that are not promoted."""
    loaded = load_run(pathspec)
    if require_tag and require_tag not in loaded.tags:
        raise SystemExit(f"{pathspec} does not have the {require_tag!r} tag; refusing to build")
    if expect_hash and loaded.dataset_hash != expect_hash:
        raise SystemExit(f"dataset hash mismatch: {loaded.dataset_hash} != {expect_hash}")
    assert_self_contained(loaded.onnx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.onnx").write_bytes(loaded.onnx)
    meta = {
        "pathspec": loaded.pathspec,
        "model_version": loaded.model_version,
        "dataset_hash": loaded.dataset_hash,
        "arch": loaded.arch,
        "metrics": loaded.metrics,
        "image_tag": image_tag(loaded.model_version),
        "onnx_sha256": hashlib.sha256(loaded.onnx).hexdigest(),
        "tags": [t for t in loaded.tags if t.startswith(("approved_", "eval_", PRODUCTION_TAG))],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def tag_image(pathspec: str, image_ref: str) -> None:
    """Record the built edge image on the model run so the platform can trace it."""
    namespace(None)
    Run(pathspec).add_tag(f"edge_image:{image_ref}")


def main(argv: list[str] | None = None) -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("pathspec")
    f.add_argument("out_dir")
    f.add_argument("--expect-hash", default="")
    f.add_argument("--no-require-tag", action="store_true")
    t = sub.add_parser("tag-image")
    t.add_argument("pathspec")
    t.add_argument("image_ref")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        require = None if args.no_require_tag else PRODUCTION_TAG
        print(json.dumps(fetch(args.pathspec, args.out_dir, require, args.expect_hash), indent=2))
    else:
        tag_image(args.pathspec, args.image_ref)
        print(f"tagged {args.pathspec} with edge_image:{args.image_ref}")


if __name__ == "__main__":
    main()
