from .._dispatch.keys import DispatchKey
from .._dispatch.registry import registry
from .._dispatch.dispatcher import current_dispatch_keyset, redispatch
from .._autograd.grad_mode import GradMode, no_grad
from .._autograd.node import Node
from .._autograd.utils import reduce_grad
import numpy as np


def _strip_autograd_keys(keyset):
    return keyset.without({
        DispatchKey.Autograd,
        DispatchKey.AutogradOther,
        DispatchKey.AutogradCPU,
        DispatchKey.AutogradNPU,
        DispatchKey.AutogradXPU,
        DispatchKey.AutogradMeta,
    })


def _autograd_binary(name, backward_impl, *, save_inputs=True):
    def wrapper(a, b):
        keyset = _strip_autograd_keys(current_dispatch_keyset())
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
        keyset = _strip_autograd_keys(current_dispatch_keyset())
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
        keyset = _strip_autograd_keys(current_dispatch_keyset())
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
        keyset = _strip_autograd_keys(current_dispatch_keyset())
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
        keyset = _strip_autograd_keys(current_dispatch_keyset())
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


def _mean_backward(grad, _a, saved_a, keyset):
    with no_grad():
        # For mean, gradient is 1/N where N is the number of elements
        numel = saved_a.numel()
        ones = saved_a._ones_like()
        # Scale gradient by 1/numel
        from .._creation import tensor
        scale = tensor(1.0 / numel, device=grad.device)
        scaled_grad = redispatch("mul", keyset, grad, scale)
        return (redispatch("mul", keyset, scaled_grad, ones),)


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
registry.register_kernel("add", DispatchKey.AutogradCPU, _autograd_binary("add", _add_backward, save_inputs=False))
registry.register_kernel("add", DispatchKey.AutogradNPU, _autograd_binary("add", _add_backward, save_inputs=False))
registry.register_kernel("add", DispatchKey.AutogradMeta, _autograd_binary("add", _add_backward, save_inputs=False))
registry.register_kernel("mul", DispatchKey.Autograd, _autograd_binary("mul", _mul_backward))
registry.register_kernel("mul", DispatchKey.AutogradCPU, _autograd_binary("mul", _mul_backward))
registry.register_kernel("mul", DispatchKey.AutogradNPU, _autograd_binary("mul", _mul_backward))
registry.register_kernel("mul", DispatchKey.AutogradMeta, _autograd_binary("mul", _mul_backward))
registry.register_kernel("matmul", DispatchKey.Autograd, _autograd_binary("matmul", _matmul_backward))
registry.register_kernel("matmul", DispatchKey.AutogradCPU, _autograd_binary("matmul", _matmul_backward))
registry.register_kernel("matmul", DispatchKey.AutogradNPU, _autograd_binary("matmul", _matmul_backward))
registry.register_kernel("matmul", DispatchKey.AutogradMeta, _autograd_binary("matmul", _matmul_backward))
registry.register_kernel("sum", DispatchKey.Autograd, _autograd_unary("sum", _sum_backward, save_input=False))
registry.register_kernel("sum", DispatchKey.AutogradCPU, _autograd_unary("sum", _sum_backward, save_input=False))
registry.register_kernel("sum", DispatchKey.AutogradNPU, _autograd_unary("sum", _sum_backward, save_input=False))
registry.register_kernel("sum", DispatchKey.AutogradMeta, _autograd_unary("sum", _sum_backward, save_input=False))
registry.register_kernel("mean", DispatchKey.Autograd, _autograd_unary("mean", _mean_backward, save_input=False))
registry.register_kernel("mean", DispatchKey.AutogradCPU, _autograd_unary("mean", _mean_backward, save_input=False))
registry.register_kernel("mean", DispatchKey.AutogradNPU, _autograd_unary("mean", _mean_backward, save_input=False))
registry.register_kernel("mean", DispatchKey.AutogradMeta, _autograd_unary("mean", _mean_backward, save_input=False))
registry.register_kernel("relu", DispatchKey.Autograd, _autograd_unary("relu", _relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("relu", DispatchKey.AutogradCPU, _autograd_unary("relu", _relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("relu", DispatchKey.AutogradNPU, _autograd_unary("relu", _relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("relu", DispatchKey.AutogradMeta, _autograd_unary("relu", _relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("reshape", DispatchKey.Autograd, _autograd_view("reshape", _reshape_backward))
registry.register_kernel("reshape", DispatchKey.AutogradCPU, _autograd_view("reshape", _reshape_backward))
registry.register_kernel("reshape", DispatchKey.AutogradNPU, _autograd_view("reshape", _reshape_backward))
registry.register_kernel("reshape", DispatchKey.AutogradMeta, _autograd_view("reshape", _reshape_backward))
registry.register_kernel("transpose", DispatchKey.Autograd, _autograd_view("transpose", _transpose_backward))
registry.register_kernel("transpose", DispatchKey.AutogradCPU, _autograd_view("transpose", _transpose_backward))
registry.register_kernel("transpose", DispatchKey.AutogradNPU, _autograd_view("transpose", _transpose_backward))
registry.register_kernel("transpose", DispatchKey.AutogradMeta, _autograd_view("transpose", _transpose_backward))
registry.register_kernel("view", DispatchKey.Autograd, _autograd_view("view", _reshape_backward))
registry.register_kernel("view", DispatchKey.AutogradCPU, _autograd_view("view", _reshape_backward))
registry.register_kernel("view", DispatchKey.AutogradNPU, _autograd_view("view", _reshape_backward))
registry.register_kernel("view", DispatchKey.AutogradMeta, _autograd_view("view", _reshape_backward))
registry.register_kernel("add_", DispatchKey.Autograd, _autograd_inplace("add_", _inplace_binary_backward, save_input=True))
registry.register_kernel("add_", DispatchKey.AutogradCPU, _autograd_inplace("add_", _inplace_binary_backward, save_input=True))
registry.register_kernel("add_", DispatchKey.AutogradNPU, _autograd_inplace("add_", _inplace_binary_backward, save_input=True))
registry.register_kernel("add_", DispatchKey.AutogradMeta, _autograd_inplace("add_", _inplace_binary_backward, save_input=True))
registry.register_kernel("mul_", DispatchKey.Autograd, _autograd_inplace("mul_", _inplace_binary_backward, save_input=True))
registry.register_kernel("mul_", DispatchKey.AutogradCPU, _autograd_inplace("mul_", _inplace_binary_backward, save_input=True))
registry.register_kernel("mul_", DispatchKey.AutogradNPU, _autograd_inplace("mul_", _inplace_binary_backward, save_input=True))
registry.register_kernel("mul_", DispatchKey.AutogradMeta, _autograd_inplace("mul_", _inplace_binary_backward, save_input=True))
registry.register_kernel("relu_", DispatchKey.Autograd, _autograd_inplace("relu_", _inplace_relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("relu_", DispatchKey.AutogradCPU, _autograd_inplace("relu_", _inplace_relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("relu_", DispatchKey.AutogradNPU, _autograd_inplace("relu_", _inplace_relu_backward, cpu_only=True, save_input=True))
registry.register_kernel("relu_", DispatchKey.AutogradMeta, _autograd_inplace("relu_", _inplace_relu_backward, cpu_only=True, save_input=True))



def _contiguous_backward(grad, _a, _saved_a, _keyset):
    return (grad,)



def _to_backward(grad, a, _saved_a, keyset, args, _kwargs):
    with no_grad():
        return (redispatch("to", keyset, grad, a.device, non_blocking=False),)


# --- Activation backward implementations ---

def _silu_backward(grad, _a, saved_a, keyset):
    with no_grad():
        # d/dx(x * sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        arr = saved_a.storage().data
        sig = 1.0 / (1.0 + np.exp(-arr))
        grad_arr = sig * (1.0 + arr * (1.0 - sig))
        mask = saved_a._ones_like()
        mask.storage()._data = grad_arr.astype(mask.storage().data.dtype)
        return (redispatch("mul", keyset, grad, mask),)


def _leaky_relu_backward(grad, _a, saved_a, keyset, args, kwargs):
    negative_slope = args[0] if args else kwargs.get("negative_slope", 0.01)
    with no_grad():
        mask = saved_a._ones_like()
        mask.storage()._data = np.where(
            saved_a.storage().data > 0, 1.0, negative_slope
        ).astype(mask.storage().data.dtype)
        return (redispatch("mul", keyset, grad, mask),)


def _elu_backward(grad, _a, saved_a, keyset, args, kwargs):
    alpha = args[0] if args else kwargs.get("alpha", 1.0)
    with no_grad():
        mask = saved_a._ones_like()
        arr = saved_a.storage().data
        # d/dx ELU = 1 if x > 0, else alpha * exp(x)
        mask.storage()._data = np.where(
            arr > 0, 1.0, alpha * np.exp(arr)
        ).astype(mask.storage().data.dtype)
        return (redispatch("mul", keyset, grad, mask),)


def _mish_backward(grad, _a, saved_a, keyset):
    with no_grad():
        # d/dx mish(x) = d/dx [x * tanh(softplus(x))]
        # = tanh(sp) + x * sech^2(sp) * sigmoid(x)
        # where sp = softplus(x) = log(1 + exp(x))
        arr = saved_a.storage().data
        sp = np.log1p(np.exp(arr))
        tanh_sp = np.tanh(sp)
        sig = 1.0 / (1.0 + np.exp(-arr))
        sech2 = 1.0 - tanh_sp ** 2
        grad_arr = tanh_sp + arr * sech2 * sig
        mask = saved_a._ones_like()
        mask.storage()._data = grad_arr.astype(mask.storage().data.dtype)
        return (redispatch("mul", keyset, grad, mask),)


def _prelu_backward(grad, a, b, saved_a, saved_b, keyset):
    with no_grad():
        arr = saved_a.storage().data
        w_arr = saved_b.storage().data
        # d/dx prelu = 1 if x > 0, else weight
        grad_x_arr = np.where(arr > 0, 1.0, w_arr).astype(arr.dtype)
        mask = saved_a._ones_like()
        mask.storage()._data = grad_x_arr
        grad_a = redispatch("mul", keyset, grad, mask) if a.requires_grad else None
        # d/dw prelu = x if x <= 0, else 0
        grad_w_arr = np.where(arr > 0, 0.0, arr).astype(arr.dtype)
        w_mask = saved_a._ones_like()
        w_mask.storage()._data = grad_w_arr
        grad_b = redispatch("mul", keyset, grad, w_mask) if b.requires_grad else None
        if grad_b is not None:
            grad_b = reduce_grad(grad_b, b.shape)
        return grad_a, grad_b


def _abs_backward(grad, _a, saved_a, keyset):
    with no_grad():
        # d/dx abs(x) = sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
        arr = saved_a.storage().data
        sign_arr = np.sign(arr).astype(arr.dtype)
        mask = saved_a._ones_like()
        mask.storage()._data = sign_arr
        return (redispatch("mul", keyset, grad, mask),)


def _neg_backward(grad, _a, _saved_a, keyset):
    with no_grad():
        # d/dx (-x) = -1
        return (redispatch("neg", keyset, grad),)


registry.register_kernel("zero_", DispatchKey.Autograd, _autograd_inplace("zero_", _inplace_zero_backward, save_input=False))
registry.register_kernel("zero_", DispatchKey.AutogradCPU, _autograd_inplace("zero_", _inplace_zero_backward, save_input=False))
registry.register_kernel("zero_", DispatchKey.AutogradNPU, _autograd_inplace("zero_", _inplace_zero_backward, save_input=False))
registry.register_kernel("zero_", DispatchKey.AutogradMeta, _autograd_inplace("zero_", _inplace_zero_backward, save_input=False))
registry.register_kernel("contiguous", DispatchKey.Autograd, _autograd_unary("contiguous", _contiguous_backward, save_input=False))
registry.register_kernel("contiguous", DispatchKey.AutogradCPU, _autograd_unary("contiguous", _contiguous_backward, save_input=False))
registry.register_kernel("contiguous", DispatchKey.AutogradNPU, _autograd_unary("contiguous", _contiguous_backward, save_input=False))
registry.register_kernel("contiguous", DispatchKey.AutogradMeta, _autograd_unary("contiguous", _contiguous_backward, save_input=False))
registry.register_kernel("to", DispatchKey.Autograd, _autograd_unary_args("to", _to_backward, save_input=True))
registry.register_kernel("to", DispatchKey.AutogradCPU, _autograd_unary_args("to", _to_backward, save_input=True))
registry.register_kernel("to", DispatchKey.AutogradNPU, _autograd_unary_args("to", _to_backward, save_input=True))
registry.register_kernel("to", DispatchKey.AutogradMeta, _autograd_unary_args("to", _to_backward, save_input=True))
registry.register_kernel("silu", DispatchKey.Autograd, _autograd_unary("silu", _silu_backward, cpu_only=True))
registry.register_kernel("silu", DispatchKey.AutogradCPU, _autograd_unary("silu", _silu_backward, cpu_only=True))
registry.register_kernel("silu", DispatchKey.AutogradNPU, _autograd_unary("silu", _silu_backward, cpu_only=True))
registry.register_kernel("silu", DispatchKey.AutogradMeta, _autograd_unary("silu", _silu_backward, cpu_only=True))
registry.register_kernel("leaky_relu", DispatchKey.Autograd, _autograd_unary_args("leaky_relu", _leaky_relu_backward, cpu_only=True))
registry.register_kernel("leaky_relu", DispatchKey.AutogradCPU, _autograd_unary_args("leaky_relu", _leaky_relu_backward, cpu_only=True))
registry.register_kernel("leaky_relu", DispatchKey.AutogradNPU, _autograd_unary_args("leaky_relu", _leaky_relu_backward, cpu_only=True))
registry.register_kernel("leaky_relu", DispatchKey.AutogradMeta, _autograd_unary_args("leaky_relu", _leaky_relu_backward, cpu_only=True))
registry.register_kernel("elu", DispatchKey.Autograd, _autograd_unary_args("elu", _elu_backward, cpu_only=True))
registry.register_kernel("elu", DispatchKey.AutogradCPU, _autograd_unary_args("elu", _elu_backward, cpu_only=True))
registry.register_kernel("elu", DispatchKey.AutogradNPU, _autograd_unary_args("elu", _elu_backward, cpu_only=True))
registry.register_kernel("elu", DispatchKey.AutogradMeta, _autograd_unary_args("elu", _elu_backward, cpu_only=True))
registry.register_kernel("mish", DispatchKey.Autograd, _autograd_unary("mish", _mish_backward, cpu_only=True))
registry.register_kernel("mish", DispatchKey.AutogradCPU, _autograd_unary("mish", _mish_backward, cpu_only=True))
registry.register_kernel("mish", DispatchKey.AutogradNPU, _autograd_unary("mish", _mish_backward, cpu_only=True))
registry.register_kernel("mish", DispatchKey.AutogradMeta, _autograd_unary("mish", _mish_backward, cpu_only=True))
registry.register_kernel("prelu", DispatchKey.Autograd, _autograd_binary("prelu", _prelu_backward))
registry.register_kernel("prelu", DispatchKey.AutogradCPU, _autograd_binary("prelu", _prelu_backward))
registry.register_kernel("prelu", DispatchKey.AutogradNPU, _autograd_binary("prelu", _prelu_backward))
registry.register_kernel("prelu", DispatchKey.AutogradMeta, _autograd_binary("prelu", _prelu_backward))
registry.register_kernel("abs", DispatchKey.Autograd, _autograd_unary("abs", _abs_backward, cpu_only=True))
registry.register_kernel("abs", DispatchKey.AutogradCPU, _autograd_unary("abs", _abs_backward, cpu_only=True))
registry.register_kernel("abs", DispatchKey.AutogradNPU, _autograd_unary("abs", _abs_backward, cpu_only=True))
registry.register_kernel("abs", DispatchKey.AutogradMeta, _autograd_unary("abs", _abs_backward, cpu_only=True))
registry.register_kernel("neg", DispatchKey.Autograd, _autograd_unary("neg", _neg_backward, save_input=False, cpu_only=True))
registry.register_kernel("neg", DispatchKey.AutogradCPU, _autograd_unary("neg", _neg_backward, save_input=False, cpu_only=True))
registry.register_kernel("neg", DispatchKey.AutogradNPU, _autograd_unary("neg", _neg_backward, save_input=False, cpu_only=True))
registry.register_kernel("neg", DispatchKey.AutogradMeta, _autograd_unary("neg", _neg_backward, save_input=False, cpu_only=True))
