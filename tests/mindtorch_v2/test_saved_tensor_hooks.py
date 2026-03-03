import pytest
import mindtorch_v2 as torch


def test_saved_tensor_hooks_basic_roundtrip():
    packed = []

    def pack(tensor):
        packed.append(tensor)
        return tensor.numpy().copy()

    def unpack(data):
        return torch.tensor(data)

    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = torch.mul(x, x)
        y.sum().backward()
    assert len(packed) >= 1
    assert x.grad is not None


def test_saved_tensor_hooks_disallow_no_grad():
    def pack(tensor):
        return tensor

    def unpack(tensor):
        return tensor

    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = torch.mul(x, x)
        y.sum().backward()
    assert x.grad is not None


def test_saved_tensor_hooks_pack_raises():
    def pack(_tensor):
        raise RuntimeError("pack failed")

    def unpack(tensor):
        return tensor

    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        with pytest.raises(RuntimeError):
            y = torch.mul(x, x)
            y.sum().backward()
