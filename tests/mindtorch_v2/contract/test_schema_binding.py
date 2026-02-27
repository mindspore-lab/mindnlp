import mindtorch_v2 as torch
import torch as pt
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dispatch.registry import registry
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_dispatch_relu_missing_arg_matches_torch():
    registry.register_schema("relu", "relu(Tensor self) -> Tensor")

    def mt():
        dispatch("relu", "cpu")

    def th():
        pt.relu()

    assert_torch_error(mt, th)


def test_dispatch_relu_unexpected_kwarg_matches_torch():
    registry.register_schema("relu", "relu(Tensor self) -> Tensor")
    a = torch.tensor([1.0])

    def mt():
        dispatch("relu", a.device.type, a, badkw=1)

    def th():
        pt.relu(pt.tensor([1.0]), badkw=1)

    assert_torch_error(mt, th)


def test_dispatch_relu_too_many_args_matches_torch():
    registry.register_schema("relu", "relu(Tensor self) -> Tensor")
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])

    def mt():
        dispatch("relu", a.device.type, a, b)

    def th():
        pt.relu(pt.tensor([1.0]), pt.tensor([2.0]))

    assert_torch_error(mt, th)


def test_dispatch_add_rejects_unexpected_device_kw_matches_torch():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])

    def mt():
        dispatch("add", a.device.type, a, b, device="cpu")

    def th():
        pt.add(pt.tensor([1.0]), pt.tensor([2.0]), device="cpu")

    assert_torch_error(mt, th)


def test_dispatch_sum_rejects_invalid_keepdim_type_matches_torch():
    a = torch.tensor([1.0])

    def mt():
        dispatch("sum", a.device.type, a, keepdim="x")

    def th():
        pt.sum(pt.tensor([1.0]), keepdim="x")

    assert_torch_error(mt, th)
