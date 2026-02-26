from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_backend_registration_uses_keys():
    entry = registry.get("aten::add")
    assert DispatchKey.CPU in entry.kernels
    assert DispatchKey.NPU in entry.kernels
    assert DispatchKey.Meta in entry.kernels



def test_autograd_backend_key_registration_prefers_device_keys():
    entry = registry.get("aten::add")
    assert DispatchKey.AutogradCPU in entry.kernels
    assert DispatchKey.AutogradNPU in entry.kernels
    assert DispatchKey.AutogradMeta in entry.kernels
