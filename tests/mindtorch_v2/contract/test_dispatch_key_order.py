from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.dispatcher import _key_order


def test_dispatch_key_order_prefix_matches_torch_cuda():
    keys = _key_order(set(DispatchKey))
    names = [key.name for key in keys]
    prefix = [
        "Functionalize",
        "Autograd",
        "Meta",
        "NPU",
        "CPU",
    ]
    assert names[: len(prefix)] == prefix


def test_dispatch_key_includes_autograd_npu():
    assert DispatchKey.AutogradNPU in DispatchKey
    assert not hasattr(DispatchKey, "AutogradCUDA")
