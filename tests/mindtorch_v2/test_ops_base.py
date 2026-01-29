"""Test Op base class."""
import pytest


def test_op_has_forward_and_backward():
    """Op base class must have forward and backward methods."""
    from mindtorch_v2._ops.base import Op

    op = Op()
    assert hasattr(op, 'forward')
    assert hasattr(op, 'backward')
    assert callable(op.forward)
    assert callable(op.backward)


def test_op_forward_not_implemented():
    """Base Op.forward should raise NotImplementedError."""
    from mindtorch_v2._ops.base import Op

    op = Op()
    with pytest.raises(NotImplementedError):
        op.forward()


def test_op_has_name():
    """Op should have a name property."""
    from mindtorch_v2._ops.base import Op

    op = Op()
    assert hasattr(op, 'name')
    assert op.name == "Op"
