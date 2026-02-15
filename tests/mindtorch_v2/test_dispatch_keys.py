import mindtorch_v2 as torch
from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet
from mindtorch_v2._dispatch.registry import registry


def test_dispatch_keyset_cpu():
    t = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((t,))
    assert DispatchKey.CPU in keyset


def test_registry_schema_and_kernel():
    registry.register_schema("aten::test_add", "add(Tensor a, Tensor b) -> Tensor")

    def cpu_impl(a, b):
        return a

    registry.register_kernel("aten::test_add", DispatchKey.CPU, cpu_impl)
    entry = registry.get("aten::test_add")
    assert entry.schema is not None
    assert entry.kernels[DispatchKey.CPU] is cpu_impl


def test_keyset_includes_autograd_when_needed():
    a = torch.ones((2,))
    a.requires_grad = True
    keyset = DispatchKeySet.from_tensors((a,), grad_enabled=True, pipeline_enabled=False)
    assert DispatchKey.Autograd in keyset


def test_keyset_includes_pipeline_when_enabled():
    a = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((a,), grad_enabled=False, pipeline_enabled=True)
    assert DispatchKey.Pipeline in keyset


def test_dispatch_keyset_without_removes_keys():
    keyset = DispatchKeySet({DispatchKey.CPU, DispatchKey.Autograd})
    trimmed = keyset.without(DispatchKey.Autograd)
    assert DispatchKey.CPU in trimmed
    assert DispatchKey.Autograd not in trimmed
