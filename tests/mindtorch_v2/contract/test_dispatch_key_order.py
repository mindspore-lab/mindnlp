from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.dispatcher import _key_order


def test_dispatch_key_order_prefix_matches_torch_cuda():
    keys = _key_order(set(DispatchKey))
    names = [key.name for key in keys]
    # Skip infrastructure keys at the front
    skip = {"BackendSelect", "Pipeline", "Python"}
    names = [n for n in names if n not in skip]
    prefix = [
        "Functionalize",
        "ADInplaceOrView",
        "AutogradOther",
        "AutogradCPU",
        "AutogradNPU",
        "AutogradCUDA",
        "AutogradXPU",
        "AutogradMeta",
        "Autograd",
        "Meta",
        "NPU",
        "CUDA",
        "CPU",
    ]
    assert names[: len(prefix)] == prefix


def test_dispatch_key_includes_autograd_cuda():
    assert DispatchKey.AutogradNPU in DispatchKey
    assert DispatchKey.AutogradCUDA in DispatchKey
