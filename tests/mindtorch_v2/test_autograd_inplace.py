import pytest
import mindtorch_v2 as torch


def test_view_shares_version_counter():
    base = torch.tensor([1.0, 2.0])
    base.requires_grad = True
    view = base.view((2,))
    assert base._version_counter is view._version_counter
