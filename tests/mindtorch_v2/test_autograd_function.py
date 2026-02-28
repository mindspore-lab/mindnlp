import pytest
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2._autograd import Function
from mindtorch_v2._autograd.engine import backward, grad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allclose(t, expected, atol=1e-6):
    arr = t._numpy_view().flatten()
    return np.allclose(arr, expected, atol=atol)


# ---------------------------------------------------------------------------
# 1. Old-style Function (ctx as first param)
# ---------------------------------------------------------------------------

class _DoubleOldStyle(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.mul(x, torch.tensor([2.0]))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (torch.mul(grad_output, torch.tensor([2.0])),)


def test_old_style_function():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    assert y.grad_fn is not None
    backward(y.sum())
    # dy/dx = 2, so grad should be [2.0]
    assert _allclose(x.grad, [2.0])


# ---------------------------------------------------------------------------
# 2. New-style Function (no ctx in forward)
# ---------------------------------------------------------------------------

class _DoubleNewStyle(Function):
    @staticmethod
    def forward(x):
        return torch.mul(x, torch.tensor([2.0]))

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (torch.mul(grad_output, torch.tensor([2.0])),)


def test_new_style_function():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleNewStyle.apply(x)
    assert y.grad_fn is not None
    backward(y.sum())
    assert _allclose(x.grad, [2.0])


# ---------------------------------------------------------------------------
# 3. No-grad path — inputs without requires_grad
# ---------------------------------------------------------------------------

def test_no_grad_path():
    x = torch.tensor([3.0])  # requires_grad=False by default
    y = _DoubleOldStyle.apply(x)
    assert y.grad_fn is None


# ---------------------------------------------------------------------------
# 4. mark_non_differentiable
# ---------------------------------------------------------------------------

class _NonDiffFunc(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.mul(x, torch.tensor([2.0]))
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,)


def test_mark_non_differentiable():
    x = torch.tensor([3.0], requires_grad=True)
    y = _NonDiffFunc.apply(x)
    # Output is marked non-differentiable, so no grad_fn
    assert y.grad_fn is None


# ---------------------------------------------------------------------------
# 5. needs_input_grad
# ---------------------------------------------------------------------------

class _CheckNeedsGrad(Function):
    captured_needs = None

    @staticmethod
    def forward(ctx, x, y):
        _CheckNeedsGrad.captured_needs = ctx.needs_input_grad
        ctx.save_for_backward(x, y)
        return torch.add(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, grad_output)


def test_needs_input_grad_values():
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0])  # no grad
    out = _CheckNeedsGrad.apply(x, y)
    assert _CheckNeedsGrad.captured_needs == (True, False)


# ---------------------------------------------------------------------------
# 6. Version check — inplace modification after save_for_backward
# ---------------------------------------------------------------------------

class _SaveAndReturn(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.mul(x, torch.tensor([1.0]))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (grad_output,)


def test_version_check():
    x = torch.tensor([1.0], requires_grad=True)
    y = _SaveAndReturn.apply(x)
    # Mutate x in-place to bump its version counter
    x._version_counter.bump()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        backward(y.sum())


# ---------------------------------------------------------------------------
# 7. Double backward without retain_graph raises RuntimeError
# ---------------------------------------------------------------------------

def test_double_backward_raises():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    backward(y.sum())
    # Second backward should fail — saved tensors are released
    x.grad = None
    with pytest.raises(RuntimeError, match="Trying to backward through the graph a second time"):
        backward(y.sum())


# ---------------------------------------------------------------------------
# 8. With retain_graph — double backward works
# ---------------------------------------------------------------------------

def test_retain_graph():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    backward(y.sum(), retain_graph=True)
    first_grad = x.grad._numpy_view().copy()
    x.grad = None
    backward(y.sum())
    second_grad = x.grad._numpy_view().copy()
    assert np.allclose(first_grad, second_grad)


# ---------------------------------------------------------------------------
# 9. Gradient accumulation — .grad is accumulated correctly
# ---------------------------------------------------------------------------

def test_gradient_accumulation():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    backward(y.sum(), retain_graph=True)
    # grad should be [2.0]
    assert _allclose(x.grad, [2.0])
    # Run backward again without clearing grad -> should accumulate
    backward(y.sum())
    assert _allclose(x.grad, [4.0])


# ---------------------------------------------------------------------------
# 10. Chain two custom Functions
# ---------------------------------------------------------------------------

class _AddOne(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.add(x, torch.tensor([1.0]))

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,)


def test_chain_functions():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)  # y = 2*x
    z = _AddOne.apply(y)          # z = 2*x + 1
    backward(z.sum())
    # dz/dx = 2
    assert _allclose(x.grad, [2.0])


# ---------------------------------------------------------------------------
# 12. autograd.grad() with custom Function
# ---------------------------------------------------------------------------

def test_autograd_grad():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    (dx,) = grad(y.sum(), (x,))
    assert _allclose(dx, [2.0])
    # x.grad should NOT be set (autograd.grad doesn't accumulate by default)
    assert x.grad is None
