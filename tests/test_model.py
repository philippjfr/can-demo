import pytest
import torch

from candetect.model import (
    ARCH_MOBILENET,
    ARCH_SMALL_CNN,
    build_model,
    model_from_bytes,
    read_metadata,
    weights_to_bytes,
)

ARCHS = [ARCH_MOBILENET, ARCH_SMALL_CNN]


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_pass_shape(arch):
    model = build_model(arch, pretrained=False).eval()
    out = model(torch.rand(3, 3, 128, 128))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ARCHS)
def test_safetensors_round_trip(arch):
    model = build_model(arch, pretrained=False).eval()
    data = weights_to_bytes(model, {"run": "x"})
    assert read_metadata(data)["arch"] == arch
    assert read_metadata(data)["run"] == "x"
    restored = model_from_bytes(data)
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))


def test_weights_are_not_a_pickle():
    data = weights_to_bytes(build_model(ARCH_SMALL_CNN, pretrained=False))
    # safetensors starts with a little-endian header length followed by a JSON header.
    assert data[8:9] == b"{"
    assert b"\x80\x04" not in data[:16]


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_model("resnet9000", pretrained=False)
