import pytest

import mindtorch_v2 as torch


def test_inplace_on_leaf_requires_grad_errors():
    x = torch.ones((2, 2)).requires_grad_()
    with pytest.raises(
        RuntimeError,
        match=r"a leaf Variable that requires grad is being used in an in-place operation.",
    ):
        x.add_(x)


def test_inplace_on_view_of_leaf_errors():
    x = torch.ones((2, 2)).requires_grad_()
    v = x.view((4,))
    with pytest.raises(
        RuntimeError,
        match=r"a view of a leaf Variable that requires grad is being used in an in-place operation.",
    ):
        v.add_(v)


def test_dispatch_inplace_checks_leaf():
    x = torch.ones((2, 2)).requires_grad_()
    from mindtorch_v2._dispatch.dispatcher import dispatch
    with pytest.raises(
        RuntimeError,
        match=r"a leaf Variable that requires grad is being used in an in-place operation.",
    ):
        dispatch("add_", x.device.type, x, x)


def test_dispatch_inplace_checks_view_of_leaf():
    x = torch.ones((2, 2)).requires_grad_()
    v = x.view((4,))
    from mindtorch_v2._dispatch.dispatcher import dispatch
    with pytest.raises(
        RuntimeError,
        match=r"a view of a leaf Variable that requires grad is being used in an in-place operation.",
    ):
        dispatch("add_", v.device.type, v, v)


def test_schema_parses_alias_set_for_mutation():
    from mindtorch_v2._dispatch.schema import OpSchema

    schema = OpSchema("add_(Tensor(a!) self, Tensor other) -> Tensor")
    assert schema.params[0].mutates is True
    assert schema.params[0].alias_set == "a"


def test_schema_parses_alias_set_for_non_mutation():
    from mindtorch_v2._dispatch.schema import OpSchema

    schema = OpSchema("foo(Tensor(a) x) -> Tensor")
    assert schema.params[0].mutates is False
    assert schema.params[0].alias_set == "a"


def test_dispatch_mutating_args_require_alias_set():
    from mindtorch_v2._dispatch.dispatcher import _mutating_args
    from mindtorch_v2._dispatch.schema import OpSchema

    schema = OpSchema("foo(Tensor(a!) x, Tensor(!) y, Tensor z) -> Tensor")
    mutated = _mutating_args(schema, (1, 2, 3), {})
    assert mutated == [1]
