import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_inplace_view_version_error_message():
    def mt():
        x = torch.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    def th():
        x = pt.tensor([1.0], requires_grad=True)
        y = x.view((1,))
        y.add_(1.0)
        y.sum().backward()

    assert_torch_error(mt, th)
