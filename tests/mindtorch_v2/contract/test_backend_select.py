from mindtorch_v2._device import get_default_device, set_default_device
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dtype import float32
from tests.mindtorch_v2.contract.helpers import assert_torch_error
import mindtorch_v2 as torch
import torch as pt


def test_dispatch_uses_default_device_when_no_tensors():
    prev = get_default_device()
    try:
        set_default_device("meta")
        out = dispatch("empty", None, (2, 2), dtype=float32)
        assert out.device.type == "meta"
    finally:
        set_default_device(prev)


def test_dispatch_respects_explicit_device_hint():
    out = dispatch("empty", "meta", (2, 2), dtype=float32)
    assert out.device.type == "meta"


def test_dispatch_mixed_meta_cpu_device_error_matches_torch():
    def mt():
        torch.add(torch.ones((2, 2), device="meta"), torch.ones((2, 2), device="cpu"))

    def th():
        pt.add(pt.ones((2, 2), device="meta"), pt.ones((2, 2), device="cpu"))

    assert_torch_error(mt, th)


def test_dispatch_mixed_cpu_meta_device_error_matches_torch():
    def mt():
        torch.add(torch.ones((2, 2), device="cpu"), torch.ones((2, 2), device="meta"))

    def th():
        pt.add(pt.ones((2, 2), device="cpu"), pt.ones((2, 2), device="meta"))

    assert_torch_error(mt, th)
