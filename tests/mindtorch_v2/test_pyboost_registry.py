"""Test pyboost op registry."""
import pytest


def test_get_pyboost_op():
    """Can get pyboost op by name."""
    from mindtorch_v2._ops.pyboost import get_pyboost_op

    add_op = get_pyboost_op('add')
    assert add_op is not None


def test_pyboost_op_works():
    """Pyboost op can execute."""
    from mindtorch_v2._ops.pyboost import get_pyboost_op
    import mindspore

    add_op = get_pyboost_op('add')
    a = mindspore.Tensor([1.0, 2.0])
    b = mindspore.Tensor([3.0, 4.0])
    result = add_op(a, b)

    assert list(result.asnumpy()) == [4.0, 6.0]


def test_unknown_op_returns_none():
    """Unknown op name returns None."""
    from mindtorch_v2._ops.pyboost import get_pyboost_op

    result = get_pyboost_op('nonexistent_op')
    assert result is None
