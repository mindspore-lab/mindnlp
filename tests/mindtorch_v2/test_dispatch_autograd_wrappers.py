import mindtorch_v2 as torch
from mindtorch_v2._dispatch.dispatcher import dispatch


def test_autograd_dispatch_add_sets_grad_fn():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    x.requires_grad = True
    y.requires_grad = True
    out = dispatch("add", x.device.type, x, y)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_mul_sets_grad_fn():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    x.requires_grad = True
    y.requires_grad = True
    out = dispatch("mul", x.device.type, x, y)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_matmul_sets_grad_fn():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0], [2.0]])
    x.requires_grad = True
    y.requires_grad = True
    out = dispatch("matmul", x.device.type, x, y)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_sum_sets_grad_fn():
    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    out = dispatch("sum", x.device.type, x)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_relu_sets_grad_fn_cpu():
    x = torch.tensor([1.0, -2.0])
    x.requires_grad = True
    out = dispatch("relu", x.device.type, x)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_reshape_sets_grad_fn():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad = True
    out = dispatch("reshape", x.device.type, x, (4,))
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_transpose_sets_grad_fn():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad = True
    out = dispatch("transpose", x.device.type, x, 0, 1)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_view_sets_grad_fn():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x.requires_grad = True
    out = dispatch("view", x.device.type, x, (2, 2))
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_add_inplace_sets_grad_fn():
    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    out = dispatch("add_", x.device.type, x, torch.tensor([1.0, 1.0]))
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_mul_inplace_sets_grad_fn():
    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    out = dispatch("mul_", x.device.type, x, torch.tensor([2.0, 3.0]))
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_relu_inplace_sets_grad_fn():
    x = torch.tensor([1.0, -2.0])
    x.requires_grad = True
    out = dispatch("relu_", x.device.type, x)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_zero_inplace_sets_grad_fn():
    x = torch.tensor([1.0, -2.0])
    x.requires_grad = True
    out = dispatch("zero_", x.device.type, x)
    assert out.requires_grad is True
    assert out.grad_fn is not None
