import importlib.util
import io
import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest
from PIL import Image

from candetect.export import export_onnx
from candetect.model import ARCH_SMALL_CNN, build_model

EDGE = Path(__file__).parents[1] / "edge"


@pytest.fixture(scope="module")
def serve(tmp_path_factory):
    path = tmp_path_factory.mktemp("edge") / "model.onnx"
    path.write_bytes(export_onnx(build_model(ARCH_SMALL_CNN, pretrained=False)))
    spec = importlib.util.spec_from_file_location("edge_serve", EDGE / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_PATH = str(path)
    return module


@pytest.fixture()
def png(small_dataset):
    buf = io.BytesIO()
    Image.fromarray(small_dataset.images[0]).save(buf, format="PNG")
    return buf.getvalue()


def test_classify_returns_label_and_probabilities(serve, png):
    result = serve.classify(serve.load_session(), png)
    assert result["label"] in serve.CLASS_NAMES
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_other_image_sizes_are_resized(serve):
    buf = io.BytesIO()
    Image.new("RGB", (64, 200), (120, 120, 120)).save(buf, format="JPEG")
    assert serve.preprocess(buf.getvalue()).shape == (1, 3, 128, 128)


def test_edge_matches_package_inference(serve, png, small_dataset):
    from candetect.inference import OnnxModel

    session = serve.load_session()
    package = OnnxModel(Path(serve.MODEL_PATH).read_bytes()).predict_proba(small_dataset.images[:1])
    edge = serve.classify(session, png)["probabilities"]
    assert [edge[n] for n in serve.CLASS_NAMES] == pytest.approx(package[0].tolist(), abs=1e-3)


def test_cli_predict_and_info(serve, png, tmp_path, capsys, monkeypatch):
    image = tmp_path / "x.png"
    image.write_bytes(png)
    monkeypatch.setenv("MODEL_VERSION", "v-test")
    assert serve.main(["predict", str(image)]) == 0
    assert json.loads(capsys.readouterr().out)["image"] == str(image)
    assert serve.main(["info"]) == 0
    assert json.loads(capsys.readouterr().out)["model_version"] == "v-test"


def test_http_endpoints(serve, png, monkeypatch):
    monkeypatch.setenv("RUN_ID", "Flow/1")
    server = HTTPServer(("127.0.0.1", 0), serve.make_handler(serve.load_session()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.load(urllib.request.urlopen(f"{base}/healthz"))
        assert health["status"] == "ok" and health["run_id"] == "Flow/1"
        request = urllib.request.Request(f"{base}/predict", data=png, method="POST")
        assert "label" in json.load(urllib.request.urlopen(request))
        bad = urllib.request.Request(f"{base}/predict", data=b"not an image", method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        server.shutdown()


def test_dockerfile_declares_config_and_provenance_labels():
    text = (EDGE / "Dockerfile").read_text()
    assert "ARG BASE_IMAGE" in text and "FROM ${BASE_IMAGE}" in text
    assert "onnxruntime==" in text and "COPY model.onnx" in text
    for name in ("run-id", "model-version", "dataset-hash"):
        assert f"com.candetect.{name}" in text
