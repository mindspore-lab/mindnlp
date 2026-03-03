import pytest
import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


@pytest.mark.parametrize(
    "mt_call, torch_call",
    [
        (lambda: torch.add(), lambda: pt.add()),
        (lambda: torch.sum(), lambda: pt.sum()),
        (lambda: torch.transpose(), lambda: pt.transpose()),
        (lambda: torch.reshape(), lambda: pt.reshape()),
        (lambda: torch.tensor([1.0]).view(), lambda: pt.tensor([1.0]).view()),
    ],
)
def test_core_missing_args_matches_torch(mt_call, torch_call):
    assert_torch_error(mt_call, torch_call)


@pytest.mark.parametrize(
    "mt_call, torch_call",
    [
        (
            lambda: torch.add(torch.tensor([1.0]), torch.tensor([2.0]), badkw=1),
            lambda: pt.add(pt.tensor([1.0]), pt.tensor([2.0]), badkw=1),
        ),
        (
            lambda: torch.sum(torch.tensor([1.0]), badkw=1),
            lambda: pt.sum(pt.tensor([1.0]), badkw=1),
        ),
        (
            lambda: torch.tensor([1.0], badkw=1),
            lambda: pt.tensor([1.0], badkw=1),
        ),
    ],
)
def test_core_unexpected_kw_matches_torch(mt_call, torch_call):
    assert_torch_error(mt_call, torch_call)
