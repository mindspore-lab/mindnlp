from ._dispatch.dispatcher import dispatch
from ._autograd.grad_mode import GradMode, no_grad
from ._autograd.node import Node
from ._autograd.utils import reduce_grad
from ._dtype import to_numpy_dtype


def add(a, b):
    out = dispatch("add", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            grad_a = reduce_grad(grad, a.shape) if a.requires_grad else None
            grad_b = reduce_grad(grad, b.shape) if b.requires_grad else None
            return grad_a, grad_b
        out.grad_fn = Node(_backward, (a, b))
        out.requires_grad = True
    return out


def mul(a, b):
    out = dispatch("mul", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            with no_grad():
                grad_a = dispatch("mul", a.device.type, grad, b) if a.requires_grad else None
                grad_b = dispatch("mul", a.device.type, grad, a) if b.requires_grad else None
            grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
            grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
            return grad_a, grad_b
        out.grad_fn = Node(_backward, (a, b))
        out.requires_grad = True
    return out


def matmul(a, b):
    out = dispatch("matmul", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            with no_grad():
                grad_a = dispatch("matmul", a.device.type, grad, b.transpose(0, 1)) if a.requires_grad else None
                grad_b = dispatch("matmul", a.device.type, a.transpose(0, 1), grad) if b.requires_grad else None
            return grad_a, grad_b
        out.grad_fn = Node(_backward, (a, b))
        out.requires_grad = True
    return out


def relu(a):
    out = dispatch("relu", a.device.type, a)
    if GradMode.enabled and a.requires_grad and a.device.type == "cpu":
        def _backward(grad):
            with no_grad():
                mask = a._ones_like()
                mask.storage.data = (a.storage.data > 0).astype(to_numpy_dtype(a.dtype))
                return (dispatch("mul", a.device.type, grad, mask),)
        out.grad_fn = Node(_backward, (a,))
        out.requires_grad = True
    return out


def sum(a, dim=None, keepdim=False):
    out = dispatch("sum", a.device.type, a, dim=dim, keepdim=keepdim)
    if GradMode.enabled and a.requires_grad:
        def _backward(grad):
            with no_grad():
                ones = a._ones_like()
                return (dispatch("mul", a.device.type, grad, ones),)
        out.grad_fn = Node(_backward, (a,))
        out.requires_grad = True
    return out
