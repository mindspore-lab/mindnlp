import pytest


def test_cuda_runtime_availability_probe_returns_bool():
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    assert isinstance(cuda_runtime.is_available(), bool)


def test_cuda_runtime_device_count_is_non_negative():
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    count = cuda_runtime.device_count()
    assert isinstance(count, int)
    assert count >= 0


def test_cuda_runtime_current_device_roundtrip_when_available():
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    if not cuda_runtime.is_available():
        pytest.skip("CUDA runtime not available")

    current = cuda_runtime.current_device()
    cuda_runtime.set_device(current)
    assert cuda_runtime.current_device() == current


def test_cuda_runtime_synchronize_succeeds_when_available():
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    if not cuda_runtime.is_available():
        pytest.skip("CUDA runtime not available")

    cuda_runtime.synchronize()


def test_public_cuda_module_exposes_basic_api():
    import mindtorch_v2 as torch

    assert hasattr(torch, "cuda")
    assert isinstance(torch.cuda.is_available(), bool)
    assert isinstance(torch.cuda.device_count(), int)


def test_public_cuda_module_set_device_roundtrip(monkeypatch):
    import mindtorch_v2 as torch
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    state = {"current": 0}
    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 2)
    monkeypatch.setattr(cuda_runtime, "current_device", lambda: state["current"])
    monkeypatch.setattr(cuda_runtime, "set_device", lambda idx: state.__setitem__("current", int(idx)))

    torch.cuda.set_device(1)
    assert torch.cuda.current_device() == 1


def test_tensor_creation_on_cuda_with_fake_runtime(monkeypatch):
    import mindtorch_v2 as torch
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 1)

    x = torch.tensor([1.0, 2.0], device="cuda")
    assert x.device.type == "cuda"
    assert x.tolist() == [1.0, 2.0]


def test_zeros_creation_on_cuda_with_fake_runtime(monkeypatch):
    import mindtorch_v2 as torch
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 1)

    x = torch.zeros(2, device="cuda")
    assert x.device.type == "cuda"
    assert x.tolist() == [0.0, 0.0]


def test_public_cuda_stream_api(monkeypatch):
    import mindtorch_v2 as torch
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "create_stream", lambda: 123)
    monkeypatch.setattr(cuda_runtime, "destroy_stream", lambda stream: None)
    seen = {}
    monkeypatch.setattr(cuda_runtime, "synchronize_stream", lambda stream: seen.setdefault("stream", stream))

    stream = torch.cuda.Stream()
    stream.synchronize()
    assert seen["stream"] == 123


def test_public_cuda_event_api(monkeypatch):
    import mindtorch_v2 as torch
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "create_event", lambda: 456)
    monkeypatch.setattr(cuda_runtime, "destroy_event", lambda event: None)
    seen = {}
    monkeypatch.setattr(cuda_runtime, "record_event", lambda event, stream=None: seen.setdefault("record", (event, stream)))
    monkeypatch.setattr(cuda_runtime, "synchronize_event", lambda event: seen.setdefault("sync", event))

    event = torch.cuda.Event()
    event.record()
    event.synchronize()
    assert seen["record"] == (456, None)
    assert seen["sync"] == 456


def test_ones_and_full_creation_on_cuda_with_fake_runtime(monkeypatch):
    import mindtorch_v2 as torch
    import mindtorch_v2._backends.cuda.runtime as cuda_runtime

    monkeypatch.setattr(cuda_runtime, "is_available", lambda: True)
    monkeypatch.setattr(cuda_runtime, "device_count", lambda: 1)

    ones = torch.ones(2, device="cuda")
    full = torch.full((2,), 3.0, device="cuda")

    assert ones.device.type == "cuda"
    assert full.device.type == "cuda"
    assert ones.tolist() == [1.0, 1.0]
    assert full.tolist() == [3.0, 3.0]
