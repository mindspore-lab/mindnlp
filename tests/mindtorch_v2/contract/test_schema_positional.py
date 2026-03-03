import mindtorch_v2 as torch
import torch as pt
from mindtorch_v2._dispatch.dispatcher import dispatch
from mindtorch_v2._dispatch.registry import registry
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_dispatch_relu_duplicate_arg_matches_torch():
    registry.register_schema("relu", "relu(Tensor self) -> Tensor")
    a = torch.tensor([1.0])

    def mt():
        dispatch("relu", a.device.type, a, self=a)

    def th():
        pt.relu(pt.tensor([1.0]), input=pt.tensor([1.0]))

    assert_torch_error(mt, th)
