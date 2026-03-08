import pytest
import mindtorch_v2 as torch


def _require_cuda_runtime():
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    if not cuda_runtime.is_available():
        pytest.skip("CUDA not available")
    return cuda_runtime


def test_to_device_roundtrip():
    x = torch.tensor([1.0, 2.0])
    y = x.to("cpu")
    assert y.device.type == "cpu"


def test_tensor_cuda_moves_tensor_to_cuda_device():
    _require_cuda_runtime()
    x = torch.tensor([1.0, 2.0])
    y = x.cuda()
    assert y.device.type == "cuda"
    assert y.tolist() == [1.0, 2.0]


def test_tensor_to_cuda_and_back_roundtrip():
    _require_cuda_runtime()
    x = torch.tensor([1.0, 2.0])
    y = x.to("cuda")
    z = y.to("cpu")
    assert y.device.type == "cuda"
    assert z.device.type == "cpu"
    assert z.tolist() == [1.0, 2.0]


def test_tensor_cuda_int_device_zero_preserves_cuda_index():
    _require_cuda_runtime()
    x = torch.tensor([1.0, 2.0])
    y = x.cuda(0)
    assert y.device.type == "cuda"
    assert y.device.index == 0


def test_tensor_to_cuda_and_back_roundtrip_with_fake_runtime(monkeypatch):
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 1)

    x = torch.tensor([1.0, 2.0])
    y = x.to("cuda")
    z = y.to("cpu")

    assert y.device.type == "cuda"
    assert z.device.type == "cpu"
    assert z.tolist() == [1.0, 2.0]
