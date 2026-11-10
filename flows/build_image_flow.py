"""Validate a promoted model and prepare its edge image build.

Started by the event that scripts/promote.py publishes after tagging a run `production`. The
platform cannot build a standalone image with a model and labels baked in, so the image itself is
built by .github/workflows/build-image.yml (see README, "Image build fallback"). This flow checks
provenance on the platform and records the manifest the build must match.
"""

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    namespaced_event_name,
    project,
    resources,
    step,
    trigger,
)
from metaflow import environment as step_environment
from metaflow.cards import Markdown, Table

from _env import BASE_IMAGE, ONNXRUNTIME_ENV, PROJECT, PROMOTION_EVENT, REGISTRY, conda_env

IMAGE_NAME = "can-detection-edge"


@project(name=PROJECT)
@trigger(
    event={
        "name": namespaced_event_name(PROMOTION_EVENT),
        "parameters": {
            "pathspec": "pathspec",
            "model_version": "model_version",
            "dataset_hash": "dataset_hash",
            "approved_by": "approved_by",
        },
    }
)
@conda_env
class BuildEdgeImageFlow(FlowSpec):
    pathspec = Parameter("pathspec", default="", help="Promoted run, e.g. TrainCanDetectorFlow/123")
    model_version = Parameter("model_version", default="", help="Expected model version")
    dataset_hash = Parameter("dataset_hash", default="", help="Expected dataset sha256")
    approved_by = Parameter("approved_by", default="", help="Approver from the event")
    registry = Parameter("registry", default=REGISTRY, help="Image registry namespace")
    base_image = Parameter("base_image", default=BASE_IMAGE, help="Base image for the edge image")

    @step_environment(vars=ONNXRUNTIME_ENV)
    @step
    def start(self):
        """Check that the run is production tagged, approved, and matches the event."""
        from candetect.config import PRODUCTION_TAG
        from candetect.inference import load_run
        from candetect.promotion import APPROVER_PREFIX

        assert self.pathspec, "pass --pathspec or trigger through the promotion event"
        run = load_run(self.pathspec)
        approvers = [t[len(APPROVER_PREFIX) :] for t in run.tags if t.startswith(APPROVER_PREFIX)]
        checks = {
            "production tag present": PRODUCTION_TAG in run.tags,
            "approver recorded": bool(approvers),
            "model version matches event": not self.model_version
            or run.model_version == self.model_version,
            "dataset hash matches event": not self.dataset_hash
            or run.dataset_hash == self.dataset_hash,
        }
        failed = [name for name, ok in checks.items() if not ok]
        assert not failed, f"provenance checks failed: {failed}"
        self.checks = checks
        self.run_model_version = run.model_version
        self.run_dataset_hash = run.dataset_hash
        self.run_approvers = approvers
        self.next(self.verify)

    @step_environment(vars=ONNXRUNTIME_ENV)
    @resources(cpu=2, memory=4096)
    @step
    def verify(self):
        """Regenerate the data, compare hashes, and reproduce the recorded metrics via ONNX."""
        import numpy as np

        from candetect.data import DataConfig, generate_dataset, make_splits
        from candetect.export import assert_self_contained, check_parity
        from candetect.inference import load_run
        from candetect.model import model_from_bytes
        from candetect.train import confusion_matrix, metrics_from_confusion

        run = load_run(self.pathspec)
        data = generate_dataset(DataConfig.from_dict(run.data_config))
        assert data.sha256 == run.dataset_hash, "regenerated data does not match the recorded hash"
        splits = make_splits(data.labels, seed=data.config.seed)

        assert_self_contained(run.onnx)
        model = model_from_bytes(run.weights)
        parity = check_parity(model, run.onnx, data.images[splits.test][:64])
        assert parity.passed, f"ONNX parity failed: {parity}"

        probs = run.onnx_model().predict_proba(data.images[splits.test])
        cm = confusion_matrix(data.labels[splits.test], probs.argmax(1))
        metrics = metrics_from_confusion(cm)
        assert np.isclose(metrics["f1"], run.metrics["f1"], atol=1e-3), "metrics do not reproduce"
        self.reproduced_f1 = metrics["f1"]
        self.parity = parity.to_dict()
        self.next(self.end)

    @card(type="blank", id="manifest")
    @step
    def end(self):
        from candetect.promotion import image_tag

        tag = image_tag(self.run_model_version)
        registry = self.registry.rstrip("/")
        image = f"{registry}/{IMAGE_NAME}:{tag}" if registry else f"{IMAGE_NAME}:{tag}"
        self.build_manifest = {
            "image": image,
            "base_image": self.base_image,
            "labels": {
                "com.candetect.run-id": self.pathspec,
                "com.candetect.model-version": self.run_model_version,
                "com.candetect.dataset-hash": self.run_dataset_hash,
            },
            "approved_by": self.run_approvers,
            "built_by": ".github/workflows/build-image.yml",
        }
        current.card.append(Markdown(f"# Edge image build manifest\n\n`{image}`"))
        current.card.append(
            Table(
                [[name, "yes" if ok else "NO"] for name, ok in self.checks.items()]
                + [["data hash regenerated and equal", "yes"], ["ONNX parity", "yes"]],
                headers=["check", "passed"],
            )
        )
        current.card.append(
            Table(
                [[k, v] for k, v in self.build_manifest["labels"].items()]
                + [["base image", self.base_image], ["approved by", ", ".join(self.run_approvers)]],
                headers=["label", "value"],
            )
        )
        current.card.append(
            Markdown(
                "The image is built by GitHub Actions (`build-image.yml`), started by "
                f"`scripts/promote.py`. Run `{current.pathspec}` records what it must match."
            )
        )
        print(f"manifest: {image} for {self.pathspec}")


if __name__ == "__main__":
    BuildEdgeImageFlow()
