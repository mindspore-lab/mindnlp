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


def test_saved_tensor_version_mismatch_raises():
    a = torch.tensor([1.0, 2.0])
    a.requires_grad = True
    b = a.relu()
    c = b.relu()
    b.add_(torch.tensor([1.0, 1.0]))
    with pytest.raises(RuntimeError):
        c.sum().backward()


def test_inplace_npu_versioning():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    t = torch.tensor([1.0, 2.0], device="npu")
    v0 = t._version_counter.value
    t.relu_()
    assert t._version_counter.value == v0 + 1
