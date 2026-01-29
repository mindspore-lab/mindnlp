"""Test op registry."""
import pytest
import numpy as np


def test_get_op_returns_instance():
    """get_op returns an Op instance."""
    from mindtorch_v2._ops.registry import get_op
    from mindtorch_v2._ops.math_ops import AddOp

    op = get_op('add')
    assert isinstance(op, AddOp)


def test_get_op_unknown_returns_none():
    """get_op returns None for unknown ops."""
    from mindtorch_v2._ops.registry import get_op

    assert get_op('nonexistent') is None


def test_execute_op_forward():
    """execute_op runs forward correctly."""
    from mindtorch_v2._ops.registry import execute_op
    import mindspore

    a = mindspore.Tensor([1.0, 2.0])
    b = mindspore.Tensor([3.0, 4.0])

    result = execute_op('add', a, b)
    np.testing.assert_array_equal(result.asnumpy(), [4.0, 6.0])


def test_execute_op_with_backward_info():
    """execute_op can return backward info."""
    from mindtorch_v2._ops.registry import execute_op
    import mindspore

    a = mindspore.Tensor([1.0, 2.0])
    b = mindspore.Tensor([3.0, 4.0])

    result, backward_info = execute_op('add', a, b, return_backward_info=True)

    assert backward_info is not None
    assert 'op' in backward_info
    assert 'saved' in backward_info


def test_execute_op_unknown_raises():
    """execute_op raises for unknown ops."""
    from mindtorch_v2._ops.registry import execute_op

    with pytest.raises(NotImplementedError):
        execute_op('nonexistent')


def test_all_math_ops_registered():
    """All math ops should be registered."""
    from mindtorch_v2._ops.registry import get_op

    for name in ['add', 'sub', 'mul', 'div']:
        assert get_op(name) is not None, f"{name} not registered"
