import pytest

import mindtorch_v2 as torch
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_alias_resolves_kernel():
    called = {}

    def kernel(a):
        called["name"] = "alias_target"
        return a

    registry.register_schema("alias_target", "alias_target(Tensor self) -> Tensor")
    registry.register_kernel("alias_target", DispatchKey.CPU, kernel)
    registry.register_alias("alias_op", "alias_target")

    a = torch.tensor([1.0])
    out = dispatch("alias_op", a.device.type, a)

    assert out is a
    assert called.get("name") == "alias_target"


def test_alias_error_uses_alias_name():
    registry.register_schema("alias_target2", "alias_target2(Tensor self) -> Tensor")
    registry.register_kernel("alias_target2", DispatchKey.CPU, lambda a: a)
    registry.register_alias("alias_op2", "alias_target2")

    with pytest.raises(TypeError, match=r'alias_op2\(\) missing 1 required positional arguments: "input"'):
        dispatch("alias_op2", "cpu")
