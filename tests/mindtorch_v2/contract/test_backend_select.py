from mindtorch_v2._device import get_default_device, set_default_device
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dtype import float32


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
