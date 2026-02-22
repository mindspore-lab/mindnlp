from .._dispatch.keys import DispatchKey
from .._dispatch.registry import registry
from .._dispatch.dispatcher import current_dispatch_keyset, redispatch
from .._autograd.grad_mode import GradMode, no_grad
from .._autograd.node import Node
from .._autograd.utils import reduce_grad


def _autograd_binary(name, backward_impl, *, save_inputs=True):
    def wrapper(a, b):
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        out = redispatch(name, keyset, a, b)
        if GradMode.enabled and (a.requires_grad or b.requires_grad):
            node_holder = {}

            def _backward(grad):
                if save_inputs:
                    saved_a, saved_b = node_holder["node"].saved_tensors()
                else:
                    saved_a, saved_b = a, b
                return backward_impl(grad, a, b, saved_a, saved_b, keyset)

            node = Node(_backward, (a, b))
            node_holder["node"] = node
            if save_inputs:
                node.save_for_backward(a, b)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper





def _autograd_unary_args(name, backward_impl, *, cpu_only=False, save_input=True):
    def wrapper(a, *args, **kwargs):
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        out = redispatch(name, keyset, a, *args, **kwargs)
        if cpu_only and a.device.type != "cpu":
            return out
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                if save_input:
                    saved_a = node_holder["node"].saved_tensors()[0]
                else:
                    saved_a = a
                return backward_impl(grad, a, saved_a, keyset, args, kwargs)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            if save_input:
                node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper
def _autograd_unary(name, backward_impl, *, cpu_only=False, save_input=True):
    def wrapper(a, **kwargs):
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        out = redispatch(name, keyset, a, **kwargs)
        if cpu_only and a.device.type != "cpu":
            return out
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                if save_input:
                    saved_a = node_holder["node"].saved_tensors()[0]
                else:
                    saved_a = a
                return backward_impl(grad, a, saved_a, keyset)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            if save_input:
                node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_view(name, backward_impl):
    def wrapper(a, *args):
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        out = redispatch(name, keyset, a, *args)
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                return backward_impl(grad, a, saved_a, args, keyset)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_inplace(name, backward_impl, *, cpu_only=False, save_input=True):
    def wrapper(a, *args):
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        out = redispatch(name, keyset, a, *args)
        if cpu_only and a.device.type != "cpu":
            return out
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                if save_input:
                    saved = node_holder["node"].saved_tensors()[0]
                else:
                    saved = a
                return backward_impl(grad, a, saved, args, keyset)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            if save_input:
                node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _add_backward(grad, a, b, _saved_a, _saved_b, _keyset):
    grad_a = reduce_grad(grad, a.shape) if a.requires_grad else None
    grad_b = reduce_grad(grad, b.shape) if b.requires_grad else None
    return grad_a, grad_b


def _mul_backward(grad, a, b, saved_a, saved_b, keyset):
    with no_grad():
        grad_a = redispatch("mul", keyset, grad, saved_b) if a.requires_grad else None
        grad_b = redispatch("mul", keyset, grad, saved_a) if b.requires_grad else None
    grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
    grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
    return grad_a, grad_b


def _matmul_backward(grad, a, b, saved_a, saved_b, keyset):
    with no_grad():
        grad_a = redispatch("matmul", keyset, grad, saved_b.transpose(0, 1)) if a.requires_grad else None
        grad_b = redispatch("matmul", keyset, saved_a.transpose(0, 1), grad) if b.requires_grad else None
    return grad_a, grad_b


def _sum_backward(grad, _a, saved_a, keyset):
    with no_grad():
        ones = saved_a._ones_like()
        return (redispatch("mul", keyset, grad, ones),)


def _relu_backward(grad, _a, saved_a, keyset):
    with no_grad():
        mask = saved_a._ones_like()
        mask.storage()._data = (saved_a.storage().data > 0).astype(mask.storage().data.dtype)
        return (redispatch("mul", keyset, grad, mask),)


def _reshape_backward(grad, a, _saved_a, _args, keyset):
    return (redispatch("reshape", keyset, grad, a.shape),)


def _transpose_backward(grad, _a, _saved_a, args, keyset):
    dim0, dim1 = args
    return (redispatch("transpose", keyset, grad, dim0, dim1),)


def _inplace_binary_backward(grad, a, _saved_a, args, _keyset):
    b = args[0]
    grad_a = reduce_grad(grad, a.shape) if a.requires_grad else None
    grad_b = reduce_grad(grad, b.shape) if b.requires_grad else None
    return grad_a, grad_b


def _inplace_relu_backward(grad, _a, saved_a, _args, keyset):
    with no_grad():
        mask = saved_a._ones_like()
        mask.storage()._data = (saved_a.storage().data > 0).astype(mask.storage().data.dtype)
        return (redispatch("mul", keyset, grad, mask),)


def _inplace_zero_backward(_grad, _a, _saved_a, _args, _keyset):
    return (None,)


registry.register_kernel("add", DispatchKey.Autograd, _autograd_binary("add", _add_backward, save_inputs=False))
registry.register_kernel("mul", DispatchKey.Autograd, _autograd_binary("mul", _mul_backward))
registry.register_kernel("matmul", DispatchKey.Autograd, _autograd_binary("matmul", _matmul_backward))
registry.register_kernel("sum", DispatchKey.Autograd, _autograd_unary("sum", _sum_backward, save_input=False))
registry.register_kernel("relu", DispatchKey.Autograd, _autograd_unary("relu", _relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("reshape", DispatchKey.Autograd, _autograd_view("reshape", _reshape_backward))
registry.register_kernel("transpose", DispatchKey.Autograd, _autograd_view("transpose", _transpose_backward))
registry.register_kernel("view", DispatchKey.Autograd, _autograd_view("view", _reshape_backward))
registry.register_kernel("add_", DispatchKey.Autograd, _autograd_inplace("add_", _inplace_binary_backward, save_input=True))
registry.register_kernel("mul_", DispatchKey.Autograd, _autograd_inplace("mul_", _inplace_binary_backward, save_input=True))
registry.register_kernel("relu_", DispatchKey.Autograd, _autograd_inplace("relu_", _inplace_relu_backward, cpu_only=True, save_input=True))



def _contiguous_backward(grad, _a, _saved_a, _keyset):
    return (grad,)



def _to_backward(grad, a, _saved_a, keyset, args, _kwargs):
    with no_grad():
        return (redispatch("to", keyset, grad, a.device, non_blocking=False),)

registry.register_kernel("zero_", DispatchKey.Autograd, _autograd_inplace("zero_", _inplace_zero_backward, save_input=False))
registry.register_kernel("contiguous", DispatchKey.Autograd, _autograd_unary("contiguous", _contiguous_backward, save_input=False))
registry.register_kernel("to", DispatchKey.Autograd, _autograd_unary_args("to", _to_backward, save_input=True))
