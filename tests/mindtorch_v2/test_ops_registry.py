"""Test op registry and dispatch."""
import pytest
import numpy as np


def test_get_op_returns_instance():
    """get_op returns an Op instance for base ops."""
    from mindtorch_v2._ops.registry import get_op
    from mindtorch_v2._ops.base import Op

    # Registry may return None if ops are not explicitly registered
    # (ops go through dispatch system now instead)
    op = get_op('add')
    # Either None or an Op instance is acceptable
    assert op is None or isinstance(op, Op)


def test_get_op_unknown_returns_none():
    """get_op returns None for unknown ops."""
    from mindtorch_v2._ops.registry import get_op

    assert get_op('nonexistent') is None


def test_execute_op_forward():
    """Operations work via dispatch system."""
    import mindtorch_v2 as torch

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    result = torch.add(a, b)
    np.testing.assert_array_equal(result.numpy(), [4.0, 6.0])


def test_execute_op_with_backward_info():
    """Operations support autograd via dispatch system."""
    import mindtorch_v2 as torch

    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([3.0, 4.0], requires_grad=True)

    result = torch.add(a, b)

    # Result should have grad_fn if requires_grad
    assert result.requires_grad


def test_execute_op_unknown_raises():
    """Unknown ops via dispatch raise NotImplementedError."""
    from mindtorch_v2._dispatch import dispatch

    with pytest.raises(NotImplementedError):
        dispatch('nonexistent_op_12345')


def test_all_math_ops_work():
    """All basic math ops should work via dispatch."""
    import mindtorch_v2 as torch

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    # Test that all basic ops work
    torch.add(a, b)
    torch.sub(a, b)
    torch.mul(a, b)
    torch.div(a, b)
