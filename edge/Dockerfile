# Edge inference image: onnxruntime, the promoted ONNX model and a small entrypoint.
#
# Build (build-image.yml does this; model.onnx comes from the promoted platform run):
#   docker build -f edge/Dockerfile \
#     --build-arg BASE_IMAGE=python:3.12-slim-bookworm \
#     --build-arg RUN_ID=TrainCanDetectorFlow/73007 \
#     --build-arg MODEL_VERSION=20261103-73007 \
#     --build-arg DATASET_HASH=<sha256> \
#     -t can-detection-edge:test build/
#
# The base image must have glibc older than 2.41 or onnxruntime will not import (its extension
# module requests an executable stack). Debian 12 (bookworm) works; Debian 13 (trixie) needs
# GLIBC_TUNABLES=glibc.rtld.execstack=2, which is set below so either base runs.
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG ORT_VERSION=1.30.0
RUN pip install --no-cache-dir "onnxruntime==${ORT_VERSION}" numpy pillow

ARG RUN_ID=unknown
ARG MODEL_VERSION=unknown
ARG DATASET_HASH=unknown
ARG BASE_IMAGE
LABEL org.opencontainers.image.title="can-detection-edge" \
      org.opencontainers.image.description="Dead plate detection, ONNX inference" \
      com.candetect.run-id="${RUN_ID}" \
      com.candetect.model-version="${MODEL_VERSION}" \
      com.candetect.dataset-hash="${DATASET_HASH}" \
      com.candetect.base-image="${BASE_IMAGE}"

ENV RUN_ID=${RUN_ID} \
    MODEL_VERSION=${MODEL_VERSION} \
    DATASET_HASH=${DATASET_HASH} \
    MODEL_PATH=/app/model.onnx \
    GLIBC_TUNABLES=glibc.rtld.execstack=2

WORKDIR /app
COPY model.onnx /app/model.onnx
COPY serve.py /app/serve.py

EXPOSE 8080
ENTRYPOINT ["python", "/app/serve.py"]
CMD ["serve", "--port", "8080"]
