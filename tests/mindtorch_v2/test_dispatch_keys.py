import mindtorch_v2 as torch
from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet
from mindtorch_v2._dispatch.registry import registry


def test_dispatch_keyset_cpu():
    t = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((t,))
    assert DispatchKey.CPU in keyset


def test_registry_schema_and_kernel():
    registry.register_schema("aten::add", "add(Tensor a, Tensor b) -> Tensor")

    def cpu_impl(a, b):
        return a

    registry.register_kernel("aten::add", DispatchKey.CPU, cpu_impl)
    entry = registry.get("aten::add")
    assert entry.schema is not None
    assert entry.kernels[DispatchKey.CPU] is cpu_impl
