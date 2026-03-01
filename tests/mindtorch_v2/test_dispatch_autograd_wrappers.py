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




def test_autograd_dispatch_relu_sets_grad_fn_npu_like_tensor(monkeypatch):
    from mindtorch_v2._autograd.grad_mode import set_grad_enabled
    from mindtorch_v2._dispatch.keys import DispatchKeySet, DispatchKey
    from mindtorch_v2._dispatch.dispatcher import dispatch_with_keyset

    class _FakeNode:
        def __init__(self):
            self._saved = []

        def save_for_backward(self, *tensors):
            self._saved = list(tensors)

        def saved_tensors(self):
            return tuple(self._saved)

    class _FakeTensor:
        def __init__(self):
            self.device = type("_D", (), {"type": "npu", "index": 0})()
            self.requires_grad = True
            self.grad_fn = None
            self.shape = (2,)

        def _ones_like(self):
            out = _FakeTensor()
            out.requires_grad = False
            return out

    x = _FakeTensor()

    def _fake_redispatch(name, keyset, *args, **kwargs):
        out = _FakeTensor()
        out.requires_grad = False
        out.grad_fn = None
        return out

    monkeypatch.setattr("mindtorch_v2._backends.autograd.redispatch", _fake_redispatch)
    monkeypatch.setattr("mindtorch_v2._backends.autograd.Node", lambda backward, inputs: _FakeNode())

    keyset = DispatchKeySet(int(DispatchKey.AutogradNPU) | int(DispatchKey.NPU))
    with set_grad_enabled(True):
        out = dispatch_with_keyset("relu", keyset, "npu", x)

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
    # Use non-leaf tensor (result of an operation) to avoid inplace check error
    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    y = dispatch("add", x.device.type, x, torch.tensor([0.0, 0.0]))  # non-leaf
    out = dispatch("add_", y.device.type, y, torch.tensor([1.0, 1.0]))
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_mul_inplace_sets_grad_fn():
    # Use non-leaf tensor (result of an operation) to avoid inplace check error
    x = torch.tensor([1.0, 2.0])
    x.requires_grad = True
    y = dispatch("add", x.device.type, x, torch.tensor([0.0, 0.0]))  # non-leaf
    out = dispatch("mul_", y.device.type, y, torch.tensor([2.0, 3.0]))
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_relu_inplace_sets_grad_fn():
    # Use non-leaf tensor (result of an operation) to avoid inplace check error
    x = torch.tensor([1.0, -2.0])
    x.requires_grad = True
    y = dispatch("add", x.device.type, x, torch.tensor([0.0, 0.0]))  # non-leaf
    out = dispatch("relu_", y.device.type, y)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_autograd_dispatch_zero_inplace_sets_grad_fn():
    # Use non-leaf tensor (result of an operation) to avoid inplace check error
    x = torch.tensor([1.0, -2.0])
    x.requires_grad = True
    y = dispatch("add", x.device.type, x, torch.tensor([0.0, 0.0]))  # non-leaf
    out = dispatch("zero_", y.device.type, y)
    assert out.requires_grad is True
    assert out.grad_fn is not None


def test_contiguous_autograd_sets_grad_fn():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad = True
    y = x.transpose(0, 1).contiguous()
    assert y.grad_fn is not None

    out = y.sum()
    out.backward()
    assert x.grad is not None


def test_to_autograd_sets_grad_fn_meta():
    x = torch.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True
    y = x.to("meta")
    assert y is not x
    assert y.grad_fn is not None

    out = y.sum()
    out.backward()
    assert x.grad is not None
