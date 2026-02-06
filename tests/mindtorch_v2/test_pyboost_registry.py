"""Test pyboost backend ops work via dispatch."""
import pytest


def test_get_pyboost_op():
    """Ops are accessible via the dispatch system."""
    import mindtorch_v2 as torch

    # Test that add operation works (uses pyboost internally)
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = torch.add(a, b)
    assert list(result.numpy()) == [4.0, 6.0]


def test_pyboost_op_works():
    """Pyboost ops execute correctly via dispatch."""
    import mindtorch_v2 as torch

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = torch.add(a, b)

    assert list(result.numpy()) == [4.0, 6.0]


def test_unknown_op_raises():
    """Unknown ops raise NotImplementedError."""
    from mindtorch_v2._dispatch import dispatch

    with pytest.raises(NotImplementedError):
        dispatch('nonexistent_op_xyz')
