import pytest
import mindtorch_v2 as torch


def test_to_device_roundtrip():
    x = torch.tensor([1.0, 2.0])
    y = x.to("cpu")
    assert y.device.type == "cpu"

def test_tensor_cuda_targets_reserved_cuda_device():
    x = torch.tensor([1.0, 2.0])
    with pytest.raises(NotImplementedError, match="Unsupported device: .* -> cuda"):
        x.cuda()


def test_tensor_cuda_int_device_preserves_cuda_index():
    x = torch.tensor([1.0, 2.0])
    with pytest.raises(NotImplementedError, match="Unsupported device: .* -> cuda:1"):
        x.cuda(1)

