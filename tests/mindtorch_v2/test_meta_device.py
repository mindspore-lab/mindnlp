import pytest
import mindtorch_v2 as torch


def test_meta_tensor_blocks_numpy():
    t = torch.tensor([1.0, 2.0], device="meta")
    with pytest.raises(RuntimeError, match="meta tensor has no data"):
        _ = t.numpy()


def test_meta_to_cpu_materializes():
    t = torch.tensor([1.0, 2.0], device="meta")
    out = t.to("cpu")
    assert out.device.type == "cpu"
    assert out.shape == (2,)


def test_meta_ops_shape_propagation():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    c = a + b
    d = c.relu()
    e = d.sum()
    assert c.device.type == "meta"
    assert d.device.type == "meta"
    assert e.device.type == "meta"
    assert c.shape == a.shape
    assert e.shape == ()
