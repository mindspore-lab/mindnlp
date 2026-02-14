import pytest
import mindtorch_v2 as torch


def test_view_shares_version_counter():
    base = torch.tensor([1.0, 2.0])
    base.requires_grad = True
    view = base.view((2,))
    assert base._version_counter is view._version_counter


def test_inplace_on_leaf_raises():
    t = torch.tensor([1.0])
    t.requires_grad = True
    with pytest.raises(RuntimeError):
        t.add_(1.0)


def test_inplace_on_view_of_leaf_raises():
    t = torch.tensor([1.0, 2.0])
    t.requires_grad = True
    v = t.view((2,))
    with pytest.raises(RuntimeError):
        v.relu_()


def test_inplace_increments_version():
    t = torch.tensor([1.0])
    v0 = t._version_counter.value
    t.add_(torch.tensor([1.0]))
    assert t._version_counter.value == v0 + 1
