"""ONNX self-containment check. Needs only the onnx package, not torch."""

import onnx


def assert_self_contained(onnx_bytes: bytes) -> None:
    """Fail if the graph is invalid or any tensor points at an external data file."""
    model = onnx.load_from_string(onnx_bytes)
    onnx.checker.check_model(model)
    external = [
        t.name for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external:
        raise ValueError(f"ONNX model references external data for {len(external)} tensors")
