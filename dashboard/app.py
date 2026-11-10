# pyright: reportAssignmentType=false
"""Monitoring dashboard: production model, metrics across runs, recent predictions, input drift.

Serve locally:   panel serve dashboard/app.py --dev --show
Without platform access, set DASHBOARD_MOCK=1 to read the artifacts that notebook 01 wrote.
"""

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn
import param
from metaflow import Flow, namespace

from candetect.config import PRODUCTION_TAG, TRAIN_FLOW, settings
from candetect.data import CLASS_NAMES, DataConfig, generate_dataset, shifted_config
from candetect.drift import FEATURES, drift_report, image_features, overall_status
from candetect.inference import LoadedRun, contact_sheet, load_run
from candetect.promotion import summarize_run

log = logging.getLogger("candetect.dashboard")
log.setLevel(logging.INFO)

pn.extension("tabulator", throttled=True)
pn.widgets.Tabulator.param.theme.default = "materialize"
pn.indicators.Number.param.font_size.default = "26pt"
pn.indicators.Number.param.title_size.default = "11pt"

SAME = "same as training"
SHIFTED = "shifted camera"
SEVERITY = {"ok": "success", "warn": "warning", "drift": "error"}
RUN_LIMIT = 6
SAMPLE_SIZE = 12
# PSI is noisy on small samples: at these sizes a same-distribution batch stays below 0.10.
REFERENCE_SIZE = 1000
BATCH_SIZE = 400


class PlatformSource:
    """Reads runs and artifacts through the Metaflow Client API."""

    label = "platform"

    def runs(self, min_f1: float) -> list[dict]:
        namespace(None)
        rows = []
        for run in Flow(TRAIN_FLOW):
            if run.successful and hasattr(run.data, "metrics"):
                rows.append(summarize_run(run, min_f1))
            if len(rows) == RUN_LIMIT:
                break
        return rows

    def model_run(self) -> tuple[LoadedRun, bool]:
        """The production run, or the newest run when nothing is promoted yet."""
        namespace(None)
        flow = Flow(TRAIN_FLOW)
        for run in flow.runs(PRODUCTION_TAG):
            return load_run(run.pathspec), True
        for run in flow:
            if run.successful:
                return load_run(run.pathspec), False
        raise LookupError("no successful runs yet; run `make run-remote`")


class MockSource:
    """Reads notebook 01 output from artifacts/local so the UI works without the platform."""

    label = "mock (local notebook artifacts)"

    def __init__(self, root: str | None = None):
        self.root = Path(root or os.environ.get("CANDETECT_ARTIFACTS", "artifacts/local"))

    def _summary(self) -> dict:
        return json.loads((self.root / "summary.json").read_text())

    def runs(self, min_f1: float) -> list[dict]:
        summary = self._summary()
        f1 = summary["metrics"]["f1"]
        return [
            {
                "run": "local",
                "created": "-",
                "arch": summary["arch"],
                "epochs": None,
                "images": None,
                "f1": f1,
                "accuracy": summary["metrics"]["accuracy"],
                "recall": summary["metrics"]["recall"],
                "dataset": summary["dataset_sha256"][:10],
                "production": True,
                "approved_by": "mock",
                "gate": "ok" if f1 >= min_f1 else "refused",
                "gate_reasons": [],
            }
        ]

    def model_run(self) -> tuple[LoadedRun, bool]:
        summary = self._summary()
        return (
            LoadedRun(
                pathspec="local/notebook",
                weights=b"",
                onnx=(self.root / "model.onnx").read_bytes(),
                dataset_hash=summary["dataset_sha256"],
                data_config=DataConfig().to_dict(),
                metrics=summary["metrics"],
                model_version="local",
                arch=summary["arch"],
                tags=[PRODUCTION_TAG, "mock"],
            ),
            True,
        )


def default_source():
    return MockSource() if os.environ.get("DASHBOARD_MOCK") == "1" else PlatformSource()


def _hide_inner_ticks(plot, element):
    """Grouped bars repeat the metric names under every run; the legend already names them."""
    plot.state.xaxis[0].major_label_text_alpha = 0


class Monitor(pn.viewable.Viewer):
    batch = param.Selector(default=SHIFTED, objects=[SHIFTED, SAME], doc="Simulated camera batch")
    refresh = param.Event(doc="Reload runs from the platform and draw a new batch")
    min_f1 = param.Number(default=settings().min_f1, constant=True)

    runs = param.DataFrame(default=pd.DataFrame(columns=["label", "f1", "accuracy", "recall"]))
    predictions = param.DataFrame(default=pd.DataFrame(columns=["image", "true", "predicted"]))
    drift_values = param.DataFrame(default=pd.DataFrame(columns=["feature", "source", "value"]))
    drift_table = param.DataFrame(default=pd.DataFrame(columns=["feature", "psi", "status"]))
    sheet = param.Parameter(default=None)

    production_md = param.String(default="Loading...")
    drift_message = param.String(default="")
    drift_severity = param.Selector(default="success", objects=list(SEVERITY.values()))
    notice = param.String(default="")
    notice_visible = param.Boolean(default=False)
    prod_f1 = param.Number(default=0.0)
    prod_accuracy = param.Number(default=0.0)
    prod_recall = param.Number(default=0.0)

    def __init__(self, source=None, **params):
        super().__init__(**params)
        self._source = source or default_source()
        self._batch_seed = 100
        self._run: LoadedRun | None = None
        self._model = None
        self._reference: pd.DataFrame | None = None

        # from_param widgets come after super().__init__ so their watchers are live.
        self._batch_select = pn.ui.RadioButtonGroup.from_param(
            self.param.batch, orientation="vertical", size="small", sizing_mode="stretch_width"
        )
        self._refresh_button = pn.ui.Button.from_param(
            self.param.refresh, label="Reload from platform", icon="refresh", color="primary",
            sizing_mode="stretch_width",
        )  # fmt: skip

        notice = pn.ui.Alert(
            object=self.param.notice, severity="warning", visible=self.param.notice_visible,
            sizing_mode="stretch_width",
        )  # fmt: skip
        production = pn.ui.Markdown(self.param.production_md, sizing_mode="stretch_width")
        kpis = pn.ui.Row(
            pn.indicators.Number(
                name="F1 (test)", value=self.param.prod_f1, format="{value:.3f}",
                sizing_mode="stretch_width",
            ),
            pn.indicators.Number(
                name="Accuracy", value=self.param.prod_accuracy, format="{value:.3f}",
                sizing_mode="stretch_width",
            ),
            pn.indicators.Number(
                name="Recall", value=self.param.prod_recall, format="{value:.3f}",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )  # fmt: skip

        runs_plot = hv.DynamicMap(self._runs_plot) * hv.DynamicMap(
            pn.bind(self._threshold_line, self.param.min_f1)
        )
        runs_pane = pn.pane.HoloViews(runs_plot, sizing_mode="stretch_width")
        sheet = pn.ui.PNG(self.param.sheet, sizing_mode="stretch_width")
        table = pn.ui.Tabulator(
            self.param.predictions, disabled=True, show_index=False, layout="fit_columns",
            pagination=None, height=330, sizing_mode="stretch_width",
            hidden_columns=["image"],
        )  # fmt: skip

        drift_alert = pn.ui.Alert(
            object=self.param.drift_message, severity=self.param.drift_severity,
            sizing_mode="stretch_width",
        )  # fmt: skip
        drift_table = pn.ui.Tabulator(
            self.param.drift_table, disabled=True, show_index=False, layout="fit_columns",
            sizing_mode="stretch_width",
        )  # fmt: skip
        hists = [
            pn.pane.HoloViews(
                hv.DynamicMap(pn.bind(self._feature_hist, name, self.param.drift_values)),
                sizing_mode="stretch_width",
            )
            for name in FEATURES
        ]

        def card(*items, title):
            return pn.ui.Paper(
                pn.ui.Typography(title, variant="h6"), *items,
                sx={"p": 2, "height": "100%"}, sizing_mode="stretch_width",
            )  # fmt: skip

        grid = pn.ui.Grid(
            pn.ui.Grid(notice, size={"xs": 12}),
            pn.ui.Grid(card(production, title="Production model"), size={"xs": 12, "md": 7}),
            pn.ui.Grid(card(kpis, title="Held-out test metrics"), size={"xs": 12, "md": 5}),
            pn.ui.Grid(card(runs_pane, title="Metrics across runs"), size={"xs": 12, "md": 7}),
            pn.ui.Grid(
                card(sheet, table, title="Recent predictions"), size={"xs": 12, "md": 5}
            ),
            pn.ui.Grid(
                card(
                    drift_alert,
                    pn.ui.Row(*hists, sizing_mode="stretch_width"),
                    drift_table,
                    title="Input drift against the training distribution",
                ),
                size={"xs": 12},
            ),
            container=True, spacing=2, sizing_mode="stretch_width",
        )  # fmt: skip
        self._page = pn.ui.Page(
            title="Can detection monitor",
            sidebar=[
                pn.ui.Typography("Incoming batch", variant="subtitle2"),
                self._batch_select,
                pn.ui.Typography(
                    "The shifted camera is darker, noisier and blurrier than the training data.",
                    variant="caption",
                ),
                self._refresh_button,
            ],
            main=[pn.ui.Container(grid, width_option="xl")],
        )

        self._load_platform()
        self._load_batch()

    def _load_platform(self):
        try:
            rows = self._source.runs(self.min_f1)
            run, is_production = self._source.model_run()
        except Exception as err:  # platform, network and auth errors all end up in the UI
            log.exception("could not read from %s", self._source.label)
            self.param.update(
                notice=f"Could not read from {self._source.label}: {err}", notice_visible=True
            )
            return
        log.info("read %d runs from %s; model run %s", len(rows), self._source.label, run.pathspec)
        self._run = run
        self._model = run.onnx_model()
        cfg = DataConfig.from_dict(run.data_config)
        reference = generate_dataset(replace(cfg, n_images=REFERENCE_SIZE))
        self._reference = image_features(reference.images)

        frame = pd.DataFrame(rows)
        marks = frame["production"].map({True: " *", False: ""})
        frame["label"] = frame["run"].astype(str) + marks
        shown = ("approved_", "eval_", "edge_", PRODUCTION_TAG)
        tags = [t for t in run.tags if t.startswith(shown)]
        edge_tags = [t.split(":", 1)[1] for t in run.tags if t.startswith("edge_image:")]
        edge = edge_tags[0] if edge_tags else "not built"
        self.param.update(
            runs=frame,
            production_md="\n\n".join(
                [
                    f"**Run** `{run.pathspec}`  \n**Model version** `{run.model_version}`  \n"
                    f"**Architecture** {run.arch}",
                    f"**Dataset sha256** `{run.dataset_hash}`",
                    f"**Edge image** `{edge}`",
                    "**Tags** " + (" ".join(f"`{t}`" for t in sorted(tags)) or "none"),
                ]
            ),
            prod_f1=run.metrics["f1"],
            prod_accuracy=run.metrics["accuracy"],
            prod_recall=run.metrics["recall"],
            notice=""
            if is_production
            else "No run is tagged production yet, so the newest run is shown.",
            notice_visible=not is_production,
        )

    def _load_batch(self):
        if self._model is None:
            return
        cfg = DataConfig.from_dict(self._run.data_config)
        if self.batch == SHIFTED:
            batch_cfg = shifted_config(cfg, seed=self._batch_seed, n_images=BATCH_SIZE)
        else:
            batch_cfg = replace(cfg, seed=self._batch_seed, n_images=BATCH_SIZE)
        batch = generate_dataset(batch_cfg)

        probs = self._model.predict_proba(batch.images)
        pred = probs.argmax(1)
        sample = slice(0, SAMPLE_SIZE)
        predictions = pd.DataFrame(
            {
                "image": np.arange(SAMPLE_SIZE),
                "true": [CLASS_NAMES[i] for i in batch.labels[sample]],
                "predicted": [CLASS_NAMES[i] for i in pred[sample]],
                "confidence": probs[sample].max(1).round(3),
                "correct": pred[sample] == batch.labels[sample],
            }
        )
        features = image_features(batch.images)
        report = drift_report(self._reference, features)
        status = overall_status(report)
        accuracy = float((pred == batch.labels).mean())
        values = pd.concat(
            [
                self._reference.melt(var_name="feature").assign(source="training"),
                features.melt(var_name="feature").assign(source="batch"),
            ]
        )
        with pn.io.hold():
            self.param.update(
                predictions=predictions,
                sheet=contact_sheet(
                    batch.images[sample], batch.labels[sample], probs[sample], columns=3
                ),
                drift_values=values,
                drift_table=report.round({"psi": 3}),
                drift_severity=SEVERITY[status],
                drift_message=(
                    f"{status.upper()}: worst PSI {report['psi'].max():.2f} "
                    f"({self.batch}); accuracy on this batch {accuracy:.1%}. "
                    "PSI below 0.10 is stable, above 0.25 is a major shift."
                ),
            )

    @param.depends("refresh", watch=True)
    def _on_refresh(self):
        self._batch_seed += 1
        self._load_platform()
        self._load_batch()

    @param.depends("batch", watch=True)
    def _on_batch(self):
        self._load_batch()

    @param.depends("runs")
    def _runs_plot(self):
        if self.runs.empty:
            return hv.Bars([], ["label", "Variable"], "value").opts(responsive=True, height=320)
        return self.runs.hvplot.bar(
            x="label", y=["f1", "accuracy", "recall"], ylim=(0, 1.05), rot=0, legend="bottom",
            xlabel="run (* = production)", ylabel="", responsive=True, height=320,
            title="Test metrics per run, dashed line is the promotion threshold",
        ).opts(hooks=[_hide_inner_ticks])  # fmt: skip

    @staticmethod
    def _threshold_line(min_f1):
        return hv.HLine(min_f1).opts(color="crimson", line_dash="dashed", line_width=2)

    @staticmethod
    def _feature_hist(feature, values):
        data = values[values["feature"] == feature]
        if data.empty:
            return hv.NdOverlay({}, kdims=["source"])
        return data.hvplot.hist(
            "value", by="source", bins=20, alpha=0.55, normed=True, legend="bottom",
            xlabel="", ylabel="density", title=feature, responsive=True, height=220,
        )  # fmt: skip

    def __panel__(self):
        return self._page


if pn.state.served:
    Monitor().servable()
elif os.environ.get("DASHBOARD_WARM_CHECK") == "1":
    # `panel serve --warm` runs the script once at startup: reading from the platform here puts
    # access problems in the app log before anyone opens the page.
    Monitor()
