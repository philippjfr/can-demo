import io

import onnx
import onnxruntime as ort
import pytest
import torch

from candetect.export import assert_self_contained, check_parity, export_onnx
from candetect.model import ARCH_MOBILENET, ARCH_SMALL_CNN, build_model

ARCHS = [ARCH_MOBILENET, ARCH_SMALL_CNN]


def _model_with_nontrivial_logits(arch):
    # Random heads give near-zero logits, which would make a parity check vacuous.
    torch.manual_seed(0)
    model = build_model(arch, pretrained=False).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2) if p.ndim == 1 else None
    last = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.normal_(last.weight, std=1.0)
    return model


@pytest.mark.parametrize("arch", ARCHS)
def test_onnx_parity_and_single_file(arch, small_dataset):
    model = _model_with_nontrivial_logits(arch)
    data = export_onnx(model)
    assert_self_contained(data)
    parity = check_parity(model, data, small_dataset.images[:32])
    assert parity.passed, parity
    with torch.no_grad():
        logits = model(torch.rand(8, 3, 128, 128))
    assert logits.abs().max() > 0.05


def test_onnx_has_no_external_data_and_loads_from_memory():
    data = export_onnx(build_model(ARCH_SMALL_CNN, pretrained=False))
    proto = onnx.load_model(io.BytesIO(data), load_external_data=False)
    assert all(t.data_location != onnx.TensorProto.EXTERNAL for t in proto.graph.initializer)
    ort.InferenceSession(data, providers=["CPUExecutionProvider"])


def test_onnx_accepts_dynamic_batch():
    data = export_onnx(build_model(ARCH_SMALL_CNN, pretrained=False))
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    for n in (1, 5):
        out = session.run(None, {"image": torch.rand(n, 3, 128, 128).numpy()})[0]
        assert out.shape == (n, 2)


def test_self_contained_check_rejects_external_data():
    model = onnx.load_model_from_string(export_onnx(build_model(ARCH_SMALL_CNN, pretrained=False)))
    tensor = model.graph.initializer[0]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    with pytest.raises((ValueError, onnx.checker.ValidationError)):
        assert_self_contained(model.SerializeToString())
