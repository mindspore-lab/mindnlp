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


def test_dispatch_view_rejects_str_shape_matches_torch():
    mt_x = torch.tensor([1.0])

    def mt_call():
        dispatch("view", mt_x.device.type, mt_x, "x")

    def th_call():
        pt.tensor([1.0]).view("x")

    assert_torch_error(mt_call, th_call)


def test_dispatch_transpose_rejects_str_dim_matches_torch():
    mt_x = torch.tensor([1.0])

    def mt_call():
        dispatch("transpose", mt_x.device.type, mt_x, "0", 0)

    def th_call():
        pt.tensor([1.0]).transpose("0", 0)

    assert_torch_error(mt_call, th_call)
