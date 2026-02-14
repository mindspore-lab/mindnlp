from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_backend_registration_uses_keys():
    entry = registry.get("aten::add")
    assert DispatchKey.CPU in entry.kernels
    assert DispatchKey.NPU in entry.kernels
    assert DispatchKey.Meta in entry.kernels
