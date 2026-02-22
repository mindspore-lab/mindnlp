import pytest

import mindtorch_v2 as torch


def test_inplace_on_leaf_requires_grad_errors():
    x = torch.ones((2, 2)).requires_grad_()
    with pytest.raises(
        RuntimeError,
        match=r"a leaf Variable that requires grad is being used in an in-place operation.",
    ):
        x.add_(x)


def test_inplace_on_view_of_leaf_errors():
    x = torch.ones((2, 2)).requires_grad_()
    v = x.view((4,))
    with pytest.raises(
        RuntimeError,
        match=r"a view of a leaf Variable that requires grad is being used in an in-place operation.",
    ):
        v.add_(v)
