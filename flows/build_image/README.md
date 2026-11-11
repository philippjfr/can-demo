# BuildEdgeImageFlow

Starts when `scripts/promote.py` publishes the promotion event. It checks that the run is tagged `production` and approved, regenerates the dataset and compares its hash, and reproduces the recorded F1 through the ONNX file.

The platform cannot build a standalone image with a model and labels, so `.github/workflows/build-image.yml` builds the edge image. This flow records the manifest that build must match.
