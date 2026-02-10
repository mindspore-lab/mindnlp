from ._dispatch.dispatcher import dispatch
from ._autograd.grad_mode import GradMode, no_grad
from ._autograd.node import Node


def add(a, b):
    out = dispatch("add", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            return grad, grad
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
            return grad_a, grad_b
        out.grad_fn = Node(_backward, (a, b))
        out.requires_grad = True
    return out


def matmul(a, b):
    return dispatch("matmul", a.device.type, a, b)


def relu(a):
    return dispatch("relu", a.device.type, a)


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
