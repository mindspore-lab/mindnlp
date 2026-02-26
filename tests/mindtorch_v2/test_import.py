import mindtorch_v2 as torch


def test_import_has_tensor():
    assert hasattr(torch, "tensor")
