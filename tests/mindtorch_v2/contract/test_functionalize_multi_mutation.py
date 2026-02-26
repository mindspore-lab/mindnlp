import mindtorch_v2 as torch
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_functionalize_writes_back_all_mutations():
    registry.register_schema("swap", "swap(Tensor a, Tensor b) -> (Tensor, Tensor)")
    registry.register_schema("swap_", "swap_(Tensor(a!) a, Tensor(b!) b) -> Tensor")

    def swap_kernel(a, b):
        out_a = torch.tensor(b._numpy_view().copy(), device=a.device)
        out_b = torch.tensor(a._numpy_view().copy(), device=a.device)
        return out_a, out_b

    registry.register_kernel("swap", DispatchKey.CPU, swap_kernel)
    registry.register_kernel("swap_", DispatchKey.CPU, swap_kernel)

    a = torch.tensor([1.0])
    b = torch.tensor([2.0])

    with torch.functionalize():
        dispatch("swap_", a.device.type, a, b)

    assert a.storage().data.tolist() == [2.0]
    assert b.storage().data.tolist() == [1.0]


def test_functionalize_multi_mutation_dedup_version_bump():
    registry.register_schema("alias_pair", "alias_pair(Tensor x, Tensor y) -> (Tensor, Tensor)")
    registry.register_schema(
        "alias_pair_",
        "alias_pair_(Tensor(a!) x, Tensor(a!) y) -> (Tensor, Tensor)",
    )

    def alias_pair_kernel(x, y):
        out_x = torch.tensor(x._numpy_view().copy() + 1.0, device=x.device)
        out_y = torch.tensor(y._numpy_view().copy() + 2.0, device=y.device)
        return out_x, out_y

    registry.register_kernel("alias_pair", DispatchKey.CPU, alias_pair_kernel)
    registry.register_kernel("alias_pair_", DispatchKey.CPU, alias_pair_kernel)

    x = torch.tensor([1.0])
    v0 = x._version_counter.value

    with torch.functionalize():
        dispatch("alias_pair_", x.device.type, x, x)

    assert x.storage().data.tolist() == [3.0]
    assert x._version_counter.value == v0 + 1


def test_functionalize_mutating_args_require_alias_set():
    from mindtorch_v2._dispatch.functionalize import _mutating_args
    from mindtorch_v2._dispatch.schema import OpSchema

    schema = OpSchema("foo_(Tensor(a!) x, Tensor(!) y, Tensor z) -> Tensor")
    mutated = _mutating_args(schema, (1, 2, 3), {})
    assert mutated == [1]


def test_dispatch_functionalize_ignores_mutation_without_alias_set():
    registry.register_schema("noalias", "noalias(Tensor x) -> Tensor")
    registry.register_schema("noalias_", "noalias_(Tensor(!) x) -> Tensor")

    def noalias_kernel(x):
        return torch.tensor(x._numpy_view().copy() + 1.0, device=x.device)

    registry.register_kernel("noalias", DispatchKey.CPU, noalias_kernel)
    registry.register_kernel("noalias_", DispatchKey.CPU, noalias_kernel)

    x = torch.tensor([1.0])
    v0 = x._version_counter.value

    with torch.functionalize():
        out = dispatch("noalias_", x.device.type, x)

    # no alias-set on mutating arg: functionalize should not write back to input
    assert x.storage().data.tolist() == [1.0]
    assert x._version_counter.value == v0
    assert out.storage().data.tolist() == [2.0]
