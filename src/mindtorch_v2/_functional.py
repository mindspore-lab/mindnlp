from ._dispatch.dispatcher import dispatch
from ._autograd.grad_mode import GradMode, no_grad
from ._autograd.node import Node
from ._autograd.utils import reduce_grad
from ._device import device as Device, get_default_device
from ._dtype import to_numpy_dtype


def add(a, b):
    out = dispatch("add", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            grad_a = reduce_grad(grad, a.shape) if a.requires_grad else None
            grad_b = reduce_grad(grad, b.shape) if b.requires_grad else None
            return grad_a, grad_b
        node = Node(_backward, (a, b))
        node.save_for_backward(a, b)
        out.grad_fn = node
        out.requires_grad = True
    return out


def mul(a, b):
    out = dispatch("mul", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            saved_a, saved_b = out.grad_fn.saved_tensors()
            with no_grad():
                grad_a = dispatch("mul", saved_a.device.type, grad, saved_b) if saved_a.requires_grad else None
                grad_b = dispatch("mul", saved_b.device.type, grad, saved_a) if saved_b.requires_grad else None
            grad_a = reduce_grad(grad_a, saved_a.shape) if grad_a is not None else None
            grad_b = reduce_grad(grad_b, saved_b.shape) if grad_b is not None else None
            return grad_a, grad_b
        node = Node(_backward, (a, b))
        node.save_for_backward(a, b)
        out.grad_fn = node
        out.requires_grad = True
    return out


def matmul(a, b):
    out = dispatch("matmul", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            saved_a, saved_b = out.grad_fn.saved_tensors()
            with no_grad():
                grad_a = dispatch("matmul", saved_a.device.type, grad, saved_b.transpose(0, 1)) if saved_a.requires_grad else None
                grad_b = dispatch("matmul", saved_a.device.type, saved_a.transpose(0, 1), grad) if saved_b.requires_grad else None
            return grad_a, grad_b
        node = Node(_backward, (a, b))
        node.save_for_backward(a, b)
        out.grad_fn = node
        out.requires_grad = True
    return out


def relu(a):
    out = dispatch("relu", a.device.type, a)
    if GradMode.enabled and a.requires_grad and a.device.type == "cpu":
        def _backward(grad):
            (saved_a,) = out.grad_fn.saved_tensors()
            with no_grad():
                mask = saved_a._ones_like()
                mask.storage()._data = (saved_a.storage().data > 0).astype(to_numpy_dtype(saved_a.dtype))
                return (dispatch("mul", saved_a.device.type, grad, mask),)
        node = Node(_backward, (a,))
        node.save_for_backward(a)
        out.grad_fn = node
        out.requires_grad = True
    return out


def sum(a, dim=None, keepdim=False):
    out = dispatch("sum", a.device.type, a, dim=dim, keepdim=keepdim)
    if GradMode.enabled and a.requires_grad:
        def _backward(grad):
            (saved_a,) = out.grad_fn.saved_tensors()
            with no_grad():
                ones = saved_a._ones_like()
                return (dispatch("mul", saved_a.device.type, grad, ones),)
        node = Node(_backward, (a,))
        node.save_for_backward(a)
        out.grad_fn = node
        out.requires_grad = True
    return out


def reshape(a, shape):
    out = dispatch("reshape", a.device.type, a, shape)
    if GradMode.enabled and a.requires_grad:
        def _backward(grad):
            return (dispatch("reshape", grad.device.type, grad, a.shape),)
        out.grad_fn = Node(_backward, (a,))
        out.requires_grad = True
    return out


def view(a, shape):
    return reshape(a, shape)


def transpose(a, dim0, dim1):
    out = dispatch("transpose", a.device.type, a, dim0, dim1)
    if GradMode.enabled and a.requires_grad:
        def _backward(grad):
            return (dispatch("transpose", grad.device.type, grad, dim0, dim1),)
        out.grad_fn = Node(_backward, (a,))
        out.requires_grad = True
    return out


def tensor(data, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("tensor", dev, data, dtype=dtype)


def zeros(shape, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("zeros", dev, shape, dtype=dtype)


def ones(shape, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("ones", dev, shape, dtype=dtype)


def empty(shape, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("empty", dev, shape, dtype=dtype)


def to(a, device, non_blocking=False):
    return dispatch("to", a.device, a, device, non_blocking=non_blocking)



def _as_device(dev):
    if dev is None:
        return get_default_device()
    if isinstance(dev, str):
        return Device(dev)
    return dev
