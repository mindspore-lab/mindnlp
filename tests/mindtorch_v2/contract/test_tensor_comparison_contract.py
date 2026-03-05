import operator

import numpy as np
import torch as pt

import mindtorch_v2 as torch

from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_tensor_eq_ne_return_tensor_matches_torch():
    mt_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mt_b = torch.tensor([[1.0, 0.0], [3.0, 5.0]])
    pt_a = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_b = pt.tensor([[1.0, 0.0], [3.0, 5.0]])

    mt_eq = mt_a == mt_b
    mt_ne = mt_a != 2.0
    pt_eq = pt_a == pt_b
    pt_ne = pt_a != 2.0

    assert isinstance(mt_eq, torch.Tensor)
    assert isinstance(mt_ne, torch.Tensor)
    np.testing.assert_array_equal(mt_eq.numpy(), pt_eq.numpy())
    np.testing.assert_array_equal(mt_ne.numpy(), pt_ne.numpy())


def test_tensor_ordering_return_tensor_matches_torch():
    mt_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mt_b = torch.tensor([[1.0, 0.0], [3.0, 5.0]])
    pt_a = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
    pt_b = pt.tensor([[1.0, 0.0], [3.0, 5.0]])

    for op in (operator.lt, operator.le, operator.gt, operator.ge):
        mt_out = op(mt_a, mt_b)
        pt_out = op(pt_a, pt_b)
        assert isinstance(mt_out, torch.Tensor)
        np.testing.assert_array_equal(mt_out.numpy(), pt_out.numpy())


def test_tensor_bool_ambiguous_matches_torch():
    assert_torch_error(
        lambda: bool(torch.tensor([1, 2])),
        lambda: bool(pt.tensor([1, 2])),
    )
    assert_torch_error(
        lambda: bool(torch.tensor([])),
        lambda: bool(pt.tensor([])),
    )


def test_tensor_eq_unsupported_rhs_matches_torch():
    mt_x = torch.tensor([1, 2])
    pt_x = pt.tensor([1, 2])

    mt_out = mt_x == [1, 2]
    pt_out = pt_x == [1, 2]

    assert type(mt_out) is type(pt_out)
    assert mt_out == pt_out


def test_tensor_ordering_unsupported_rhs_matches_torch():
    assert_torch_error(
        lambda: torch.tensor([1, 2]) > [1, 2],
        lambda: pt.tensor([1, 2]) > [1, 2],
    )
