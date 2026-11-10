"""Train the can classifier on the platform.

Mirrors notebooks/01_train_locally.ipynb: the same candetect functions in the same order. The flow
only adds steps, decorators, artifacts and a card.
"""

from datetime import UTC, datetime

from metaflow import FlowSpec, Parameter, card, current, project, resources, step
from metaflow import environment as step_environment
from metaflow.cards import Image, Markdown, Table

from _env import ONNXRUNTIME_ENV, PROJECT, conda_env


@project(name=PROJECT)
@conda_env
class TrainCanDetectorFlow(FlowSpec):
    n_images = Parameter("n-images", default=2000, type=int, help="Synthetic images to generate")
    seed = Parameter("seed", default=7, type=int, help="Dataset and training seed")
    epochs = Parameter("epochs", default=3, type=int, help="Fine-tuning epochs")
    arch = Parameter(
        "arch", default="mobilenet_v3_small", help="mobilenet_v3_small or small_cnn (from scratch)"
    )
    pretrained = Parameter("pretrained", default=True, type=bool, help="Use ImageNet weights")
    fallback = Parameter(
        "fallback", default=True, type=bool, help="Train small_cnn if the weights cannot be fetched"
    )

    @step
    def start(self):
        """Generate the dataset once to fix its content hash, the key for every artifact below."""
        from candetect.data import DataConfig, generate_dataset

        data_config = DataConfig(n_images=self.n_images, seed=self.seed)
        data = generate_dataset(data_config)
        self.data_config = data_config.to_dict()
        self.dataset_hash = data.sha256
        print(f"dataset sha256: {self.dataset_hash}")
        self.next(self.train_model)

    @step_environment(vars=ONNXRUNTIME_ENV)
    @resources(cpu=4, memory=8192)
    @step
    def train_model(self):
        from candetect.data import DataConfig, generate_dataset, make_splits
        from candetect.export import check_parity, export_onnx
        from candetect.inference import OnnxModel, contact_sheet
        from candetect.model import weights_to_bytes
        from candetect.report import short_id
        from candetect.train import TrainConfig, train

        data = generate_dataset(DataConfig.from_dict(self.data_config))
        assert data.sha256 == self.dataset_hash, "dataset changed between steps"
        splits = make_splits(data.labels, seed=self.seed)

        train_config = TrainConfig(
            arch=self.arch,
            pretrained=self.pretrained,
            fallback_to_cnn=self.fallback,
            epochs=self.epochs,
            seed=self.seed,
        )
        result = train(data, splits, train_config)

        weights = weights_to_bytes(result.model, {"dataset_sha256": data.sha256})
        onnx_bytes = export_onnx(result.model)
        parity = check_parity(result.model, onnx_bytes, data.images[splits.test][:64])
        assert parity.passed, f"ONNX parity failed: {parity}"

        shown = splits.test[:8]
        probs = OnnxModel(onnx_bytes).predict_proba(data.images[shown])

        # Plain bytes and dicts only, so artifacts never unpickle torch objects.
        self.weights = weights
        self.onnx_model = onnx_bytes
        self.arch_used = result.arch
        self.pretrained_used = result.pretrained
        self.fallback_used = result.fallback_used
        self.metrics = result.test_metrics
        self.val_metrics = result.val_metrics
        self.confusion_matrix = result.confusion_matrix
        self.history = result.history
        self.train_seconds = result.train_seconds
        self.parity = parity.to_dict()
        self.sample_sheet = contact_sheet(data.images[shown], data.labels[shown], probs)
        day = datetime.now(UTC).strftime("%Y%m%d")
        self.model_version = f"{day}-{short_id(current.run_id)}"
        self.next(self.end)

    @card(type="blank", id="summary")
    @step
    def end(self):
        from metaflow import Flow, namespace

        from candetect.config import PRODUCTION_TAG

        namespace(None)
        production = next(iter(Flow(current.flow_name).runs(PRODUCTION_TAG)), None)
        self.compared_to = production.pathspec if production else ""

        current.card.append(Markdown(f"# Can detector {self.model_version}"))
        current.card.append(
            Markdown(
                f"**Run** `{current.pathspec}`  \n"
                f"**Dataset sha256** `{self.dataset_hash}`  \n"
                f"**Architecture** {self.arch_used} (pretrained: {self.pretrained_used}, "
                f"fell back to small CNN: {self.fallback_used})  \n"
                f"**ONNX parity** max abs diff {self.parity['max_abs_diff']:.2e}, "
                f"passed: {self.parity['passed']}"
            )
        )
        current.card.append(Markdown("## Test metrics"))
        current.card.append(
            Table(
                [[f"{self.metrics[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1")]],
                headers=["accuracy", "precision", "recall", "f1"],
            )
        )
        current.card.append(Markdown("## Compared with the production model"))
        if production is None:
            current.card.append(Markdown("No run is tagged `production` yet."))
        else:
            reference = production.data
            same_data = reference.dataset_hash == self.dataset_hash
            current.card.append(
                Markdown(
                    f"Production run `{production.pathspec}` "
                    f"({'same data version' if same_data else '**different data version**'})"
                )
            )
            current.card.append(
                Table(
                    [
                        [
                            name,
                            f"{self.metrics[name]:.4f}",
                            f"{reference.metrics[name]:.4f}",
                            f"{self.metrics[name] - reference.metrics[name]:+.4f}",
                        ]
                        for name in ("f1", "accuracy", "precision", "recall")
                    ],
                    headers=["metric", "this run", "production", "change"],
                )
            )
        current.card.append(Markdown("## Confusion matrix (rows: true, columns: predicted)"))
        current.card.append(
            Table(
                [
                    ["normal", *self.confusion_matrix[0]],
                    ["fallen_can", *self.confusion_matrix[1]],
                ],
                headers=["", "pred normal", "pred fallen_can"],
            )
        )
        current.card.append(Markdown("## Training history"))
        current.card.append(
            Table(
                [[h["epoch"], f"{h['loss']:.4f}", f"{h['val_f1']:.4f}"] for h in self.history],
                headers=["epoch", "loss", "val f1"],
            )
        )
        current.card.append(Markdown("## Sample test predictions"))
        current.card.append(Image(self.sample_sheet, "samples"))
        print(f"model {self.model_version} f1={self.metrics['f1']:.4f}")


if __name__ == "__main__":
    TrainCanDetectorFlow()
