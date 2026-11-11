# TrainCanDetectorFlow

Generates the synthetic can conveyor dataset, fine-tunes MobileNetV3-Small on CPU, exports a single-file ONNX model, and checks that onnxruntime matches PyTorch.

Every run records the dataset sha256, test metrics, the confusion matrix, the safetensors weights and the ONNX model. The `summary` card shows them next to the current production model.

A run becomes the production model when `scripts/promote.py` tags it `production`, which requires the test F1 gate and a passing ONNX parity check.
