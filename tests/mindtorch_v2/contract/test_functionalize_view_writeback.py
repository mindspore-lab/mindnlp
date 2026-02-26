import pytest
import mindtorch_v2 as torch


def test_functionalize_writeback_respects_view():
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    view = base.view((2, 2))
    with torch.functionalize():
        view.add_(torch.ones((2, 2)))
    assert base.storage().data.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_functionalize_writeback_respects_view_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    view = base.view((2, 2))
    with torch.functionalize():
        view.add_(torch.ones((2, 2), device="npu"))
    assert base.to("cpu").storage().data.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_functionalize_writeback_respects_noncontig_view_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device="npu").view((2, 3))
    view = base.transpose(0, 1)
    with torch.functionalize():
        view.add_(torch.ones(view.shape, device="npu"))
    assert base.to("cpu").storage().data.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_functionalize_writeback_respects_view_meta():
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="meta")
    view = base.transpose(0, 1)
    with torch.functionalize():
        view.add_(torch.ones((2, 2), device="meta"))
    assert base.device.type == "meta"
    assert view.shape == (2, 2)
    assert view.stride == (1, 2)


def test_functionalize_view_writeback_bumps_shared_version_once():
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    view = base.view((2, 2))
    v0 = base._version_counter.value
    with torch.functionalize():
        view.add_(torch.ones((2, 2)))
    assert base._version_counter.value == v0 + 1
    assert view._version_counter.value == v0 + 1


def test_functionalize_base_and_view_inplace_sequence_bumps_twice():
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    view = base.view((2, 2))
    v0 = base._version_counter.value
    with torch.functionalize():
        base.add_(torch.ones((4,)))
        view.add_(torch.ones((2, 2)))
    assert base._version_counter.value == v0 + 2
    assert view._version_counter.value == v0 + 2
