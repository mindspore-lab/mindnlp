import mindtorch_v2 as torch
import torch as pt

from mindtorch_v2._dispatch.dispatcher import dispatch
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_dispatch_sum_rejects_float_dim_matches_torch():
    a = torch.tensor([1.0])

    def mt():
        dispatch("sum", a.device.type, a, dim=1.2)

    def th():
        pt.sum(pt.tensor([1.0]), dim=1.2)

    assert_torch_error(mt, th)


def test_sum_accepts_list_dim_like_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_x = pt.tensor([[1.0, 2.0], [3.0, 4.0]])

    mt_out = torch.sum(mt_x, dim=[0, 1])
    pt_out = pt.sum(pt_x, dim=[0, 1])

    assert mt_out.item() == pt_out.item()


def test_sum_accepts_empty_dim_sequence_like_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_x = pt.tensor([[1.0, 2.0], [3.0, 4.0]])

    mt_out = torch.sum(mt_x, dim=[])
    pt_out = pt.sum(pt_x, dim=[])

    assert mt_out.item() == pt_out.item()


def test_dispatch_sum_rejects_out_of_range_dim_matches_torch():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("sum", a.device.type, a, dim=2)

    def th_call():
        pt.sum(pt.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=2)

    assert_torch_error(mt_call, th_call)


def test_dispatch_sum_rejects_out_of_range_dim_list_matches_torch():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("sum", a.device.type, a, dim=[0, 2])

    def th_call():
        pt.sum(pt.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=[0, 2])

    assert_torch_error(mt_call, th_call)


def test_sum_accepts_list_dim_with_trailing_bool_like_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_x = pt.tensor([[1.0, 2.0], [3.0, 4.0]])

    mt_out = torch.sum(mt_x, dim=[0, True])
    pt_out = pt.sum(pt_x, dim=[0, True])

    assert mt_out.item() == pt_out.item()


def test_sum_accepts_tuple_dim_with_trailing_bool_like_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_x = pt.tensor([[1.0, 2.0], [3.0, 4.0]])

    mt_out = torch.sum(mt_x, dim=(0, True))
    pt_out = pt.sum(pt_x, dim=(0, True))

    assert mt_out.item() == pt_out.item()


def test_dispatch_sum_duplicate_dim_with_bool_matches_torch():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("sum", a.device.type, a, dim=[0, False])

    def th_call():
        pt.sum(pt.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=[0, False])

    assert_torch_error(mt_call, th_call)


def test_dispatch_sum_named_dim_error_matches_torch_on_rank2():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("sum", a.device.type, a, dim="x")

    def th_call():
        pt.sum(pt.tensor([[1.0, 2.0], [3.0, 4.0]]), dim="x")

    assert_torch_error(mt_call, th_call)


def test_view_accepts_single_int_shape_like_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_x = pt.tensor([[1.0, 2.0], [3.0, 4.0]])

    mt_out = mt_x.view(4)
    pt_out = pt_x.view(4)

    assert mt_out.shape == pt_out.shape
    assert mt_out.tolist() == pt_out.tolist()


def test_view_accepts_negative_one_shape_like_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_x = pt.tensor([[1.0, 2.0], [3.0, 4.0]])

    mt_out = mt_x.view(-1)
    pt_out = pt_x.view(-1)

    assert mt_out.shape == pt_out.shape
    assert mt_out.tolist() == pt_out.tolist()


def test_dispatch_view_rejects_none_shape_matches_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("view", mt_x.device.type, mt_x, None)

    def th_call():
        pt.tensor([[1.0, 2.0], [3.0, 4.0]]).view(None)

    assert_torch_error(mt_call, th_call)


def test_dispatch_view_rejects_mixed_tuple_shape_matches_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("view", mt_x.device.type, mt_x, (4, "x"))

    def th_call():
        pt.tensor([[1.0, 2.0], [3.0, 4.0]]).view((4, "x"))

    assert_torch_error(mt_call, th_call)


def test_dispatch_view_rejects_str_shape_matches_torch():
    mt_x = torch.tensor([1.0])

    def mt_call():
        dispatch("view", mt_x.device.type, mt_x, "x")

    def th_call():
        pt.tensor([1.0]).view("x")

    assert_torch_error(mt_call, th_call)


def test_dispatch_transpose_rejects_mixed_int_str_dims_matches_torch():
    mt_x = torch.tensor([1.0, 2.0])

    def mt_call():
        dispatch("transpose", mt_x.device.type, mt_x, 0, "1")

    def th_call():
        pt.tensor([1.0, 2.0]).transpose(0, "1")

    assert_torch_error(mt_call, th_call)


def test_dispatch_transpose_rejects_bool_dim_matches_torch():
    mt_x = torch.tensor([1.0, 2.0])

    def mt_call():
        dispatch("transpose", mt_x.device.type, mt_x, True, 0)

    def th_call():
        pt.tensor([1.0, 2.0]).transpose(True, 0)

    assert_torch_error(mt_call, th_call)


def test_dispatch_transpose_rejects_str_dim_matches_torch():
    mt_x = torch.tensor([1.0])

    def mt_call():
        dispatch("transpose", mt_x.device.type, mt_x, "0", 0)

    def th_call():
        pt.tensor([1.0]).transpose("0", 0)

    assert_torch_error(mt_call, th_call)


def test_dispatch_squeeze_rejects_bool_dim_matches_torch():
    mt_x = torch.tensor([1.0, 2.0])

    def mt_call():
        dispatch("squeeze", mt_x.device.type, mt_x, True)

    def th_call():
        pt.squeeze(pt.tensor([1.0, 2.0]), True)

    assert_torch_error(mt_call, th_call)


def test_dispatch_unsqueeze_rejects_bool_dim_matches_torch():
    mt_x = torch.tensor([1.0, 2.0])

    def mt_call():
        dispatch("unsqueeze", mt_x.device.type, mt_x, True)

    def th_call():
        pt.unsqueeze(pt.tensor([1.0, 2.0]), True)

    assert_torch_error(mt_call, th_call)


def test_dispatch_permute_rejects_bool_dim_sequence_matches_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("permute", mt_x.device.type, mt_x, [True, 0])

    def th_call():
        pt.permute(pt.tensor([[1.0, 2.0], [3.0, 4.0]]), [True, 0])

    assert_torch_error(mt_call, th_call)


def test_dispatch_permute_rejects_duplicate_dims_matches_torch():
    mt_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def mt_call():
        dispatch("permute", mt_x.device.type, mt_x, [0, 0])

    def th_call():
        pt.permute(pt.tensor([[1.0, 2.0], [3.0, 4.0]]), [0, 0])

    assert_torch_error(mt_call, th_call)


def test_dispatch_squeeze_accepts_list_dim_like_torch():
    mt_x = torch.tensor([1.0, 2.0])
    pt_x = pt.tensor([1.0, 2.0])

    mt_out = dispatch("squeeze", mt_x.device.type, mt_x, [0])
    pt_out = pt.squeeze(pt_x, [0])

    assert mt_out.shape == pt_out.shape
    assert mt_out.tolist() == pt_out.tolist()


def test_dispatch_squeeze_accepts_tuple_dim_like_torch():
    mt_x = torch.tensor([1.0, 2.0])
    pt_x = pt.tensor([1.0, 2.0])

    mt_out = dispatch("squeeze", mt_x.device.type, mt_x, (0,))
    pt_out = pt.squeeze(pt_x, (0,))

    assert mt_out.shape == pt_out.shape
    assert mt_out.tolist() == pt_out.tolist()
