import pytest

import mindtorch_v2 as torch
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry
from mindtorch_v2._dispatch.functionalize import functionalize_context



def test_functionalize_routes_inplace_to_functional_kernel():
    called = {}

    def add_kernel(a, b):
        called["name"] = "add"
        out = torch.tensor((a._numpy_view() + b._numpy_view()).copy(), device=a.device)
        return out

    def add_inplace_kernel(a, b):
        called["name"] = "add_"
        return a

    registry.register_schema("add", "add(Tensor a, Tensor b) -> Tensor")
    registry.register_schema("add_", "add_(Tensor(a!) self, Tensor other) -> Tensor")
    registry.register_kernel("add", DispatchKey.Functionalize, add_kernel)
    registry.register_kernel("add_", DispatchKey.Functionalize, add_inplace_kernel)
    registry.register_kernel("add", DispatchKey.CPU, add_kernel)
    registry.register_kernel("add_", DispatchKey.CPU, add_inplace_kernel)

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    with functionalize_context():
        out = dispatch("add_", a.device.type, a, b)

    assert called.get("name") == "add"
    assert out is a
    assert a.storage().data.tolist() == [4.0, 6.0]



def test_functionalize_missing_rule_errors():
    registry.register_schema("mystery_", "mystery_(Tensor(a!) self) -> Tensor")
    registry.register_kernel("mystery_", DispatchKey.CPU, lambda a: a)
    registry.register_kernel("mystery_", DispatchKey.Functionalize, lambda a: a)

    a = torch.tensor([1.0])

    with functionalize_context():
        with pytest.raises(RuntimeError, match=r"functionalize: missing rule for op mystery_\(\)"):
            dispatch("mystery_", a.device.type, a)
