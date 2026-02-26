import pytest

import mindtorch_v2 as torch
from mindtorch_v2._dispatch.dispatcher import dispatch_with_keyset
from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet


def test_dispatch_prefers_meta_when_input_meta():
    a = torch.ones((2,), device="meta")
    b = torch.ones((2,), device="meta")
    c = torch.add(a, b)
    assert c.device.type == "meta"


def test_dispatch_prefers_npu_over_cpu():
    a = torch.ones((2,), device="npu")
    b = torch.ones((2,), device="npu")
    c = torch.add(a, b)
    assert c.device.type == "npu"


def test_dispatch_rejects_cross_npu_device_index():
    class _FakeTensor:
        def __init__(self, dev):
            self.device = torch.Device(dev)
            self.dtype = torch.float32

    a = _FakeTensor("npu:0")
    b = _FakeTensor("npu:1")
    keyset = DispatchKeySet(int(DispatchKey.NPU))

    with pytest.raises(RuntimeError, match=r"npu:1.*npu:0"):
        dispatch_with_keyset("add", keyset, None, a, b)
