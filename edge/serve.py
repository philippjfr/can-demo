"""Edge inference entrypoint: onnxruntime, numpy and Pillow only.

    python serve.py info                    # model provenance from the image labels
    python serve.py predict image.png ...   # classify image files
    python serve.py serve --port 8080       # POST a PNG or JPEG to /predict, GET /healthz
"""

import argparse
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import onnxruntime as ort
from PIL import Image

CLASS_NAMES = ("normal", "fallen_can")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model.onnx")
SIZE = 128
PROVENANCE_VARS = ("RUN_ID", "MODEL_VERSION", "DATASET_HASH")


def load_session(path: str | None = None) -> ort.InferenceSession:
    return ort.InferenceSession(path or MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess(data: bytes) -> np.ndarray:
    """PNG or JPEG bytes to a float32 NCHW batch of one, scaled to [0, 1]."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if image.size != (SIZE, SIZE):
        image = image.resize((SIZE, SIZE), Image.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def classify(session: ort.InferenceSession, data: bytes) -> dict:
    logits = session.run(["logits"], {"image": preprocess(data)})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    best = int(probs.argmax())
    return {
        "label": CLASS_NAMES[best],
        "confidence": round(float(probs[best]), 4),
        "probabilities": {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, probs, strict=True)},
    }


def provenance() -> dict:
    return {k.lower(): os.environ.get(k, "unknown") for k in PROVENANCE_VARS}


def make_handler(session: ort.InferenceSession):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"status": "ok", **provenance()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, {"error": "not found"})
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                result = classify(session, body)
            except Exception as err:  # malformed image
                return self._send(400, {"error": str(err)})
            self._send(200, {**result, "model_version": provenance()["model_version"]})

        def log_message(self, *args):
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    predict = sub.add_parser("predict")
    predict.add_argument("images", nargs="+")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    if args.command == "info":
        print(json.dumps({**provenance(), "model_path": MODEL_PATH}, indent=2))
        return 0
    session = load_session()
    if args.command == "predict":
        for path in args.images:
            with open(path, "rb") as f:
                print(json.dumps({"image": path, **classify(session, f.read())}))
        return 0
    print(f"serving on :{args.port}", file=sys.stderr, flush=True)
    HTTPServer(("0.0.0.0", args.port), make_handler(session)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
