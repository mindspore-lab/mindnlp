from contextlib import nullcontext

from .._autograd.grad_mode import GradMode, no_grad
from .._autograd.node import Node
from .._autograd.utils import reduce_grad
from .._dispatch.dispatcher import current_dispatch_keyset, redispatch
from .._dispatch.keys import DispatchKey
from .._dispatch.registry import registry
from .._dispatch.registration import register_autograd_kernels


def _strip_autograd_keys(keyset):
    if keyset is None:
        return None
    return keyset.without(
        {
            DispatchKey.Autograd,
            DispatchKey.AutogradOther,
            DispatchKey.AutogradCPU,
            DispatchKey.AutogradNPU,
            DispatchKey.AutogradXPU,
            DispatchKey.AutogradMeta,
        }
    )


def _grad_context(_keyset=None):
    from .._autograd.engine import is_create_graph_enabled

    if is_create_graph_enabled():
        return nullcontext()
    return no_grad()


def _backward_dispatch_keyset(raw_keyset, autograd_keyset):
    from .._autograd.engine import is_create_graph_enabled

    if is_create_graph_enabled() and autograd_keyset is not None:
        return autograd_keyset
    return raw_keyset


def _autograd_unary_passthrough(name):
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return (redispatch("to", backward_keyset, grad, a.device, non_blocking=False),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_binary(name, backward_impl, *, save_inputs=True):
    def wrapper(a, b):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b)
        a_requires_grad = getattr(a, "requires_grad", False)
        b_requires_grad = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_requires_grad or b_requires_grad):
            node_holder = {}

            def _backward(grad):
                if save_inputs:
                    saved_a, saved_b = node_holder["node"].saved_tensors()
                else:
                    saved_a, saved_b = a, b
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, b, saved_a, saved_b, backward_keyset)

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
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if cpu_only and a.device.type != "cpu":
            return out
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                if save_input:
                    saved_a = node_holder["node"].saved_tensors()[0]
                else:
                    saved_a = a
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, saved_a, backward_keyset, args, kwargs)

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
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, **kwargs)
        if cpu_only and a.device.type != "cpu":
            return out
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                if save_input:
                    saved_a = node_holder["node"].saved_tensors()[0]
                else:
                    saved_a = a
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, saved_a, backward_keyset)

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
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args)
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, saved_a, args, backward_keyset)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_inplace(name, backward_impl, *, cpu_only=False, save_input=True):
    def wrapper(a, *args):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args)
        if cpu_only and a.device.type != "cpu":
            return out
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                if save_input:
                    saved = node_holder["node"].saved_tensors()[0]
                else:
                    saved = a
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, saved, args, backward_keyset)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            if save_input:
                node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _scalar_tensor_like(ref, value):
    from .._creation import tensor

    return tensor(value, dtype=ref.dtype, device=ref.device)


def _add_backward(grad, a, b, _saved_a, _saved_b, _keyset):
    grad_a = reduce_grad(grad, a.shape) if getattr(a, "requires_grad", False) else None
    grad_b = reduce_grad(grad, b.shape) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


def _mul_backward(grad, a, b, saved_a, saved_b, keyset):
    with _grad_context(keyset):
        grad_a = redispatch("mul", keyset, grad, saved_b) if getattr(a, "requires_grad", False) else None
        grad_b = redispatch("mul", keyset, grad, saved_a) if getattr(b, "requires_grad", False) else None
    grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
    grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
    return grad_a, grad_b


def _div_backward(grad, a, b, saved_a, saved_b, keyset):
    grad_a = None
    grad_b = None
    with _grad_context(keyset):
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("div", keyset, grad, saved_b)
        if getattr(b, "requires_grad", False):
            denom = redispatch("mul", keyset, saved_b, saved_b)
            num = redispatch("mul", keyset, grad, saved_a)
            grad_b = redispatch("div", keyset, num, denom)
            grad_b = redispatch("neg", keyset, grad_b)
    if grad_a is not None:
        grad_a = reduce_grad(grad_a, a.shape)
    if grad_b is not None:
        grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _matmul_backward(grad, a, b, saved_a, saved_b, keyset):
    with _grad_context(keyset):
        grad_a = None
        grad_b = None

        if getattr(a, "requires_grad", False):
            grad_a = redispatch("matmul", keyset, grad, saved_b.transpose(-1, -2))
            grad_a = reduce_grad(grad_a, a.shape)

        if getattr(b, "requires_grad", False):
            grad_b = redispatch("matmul", keyset, saved_a.transpose(-1, -2), grad)
            grad_b = reduce_grad(grad_b, b.shape)

    return grad_a, grad_b


def _sum_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        return (redispatch("mul", keyset, grad, ones),)


def _mean_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        numel = saved_a.numel()
        ones = saved_a._ones_like()
        from .._creation import tensor

        scale = tensor(1.0 / numel, device=grad.device)
        scaled_grad = redispatch("mul", keyset, grad, scale)
        return (redispatch("mul", keyset, scaled_grad, ones),)


def _relu_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        mask = redispatch("sign", keyset, redispatch("relu", keyset, saved_a))
        grad_input = redispatch("mul", keyset, grad, mask)
        return (grad_input,)


def _reshape_backward(grad, a, _saved_a, _args, keyset):
    return (redispatch("reshape", keyset, grad, a.shape),)


def _transpose_backward(grad, _a, _saved_a, args, keyset):
    dim0, dim1 = args
    return (redispatch("transpose", keyset, grad, dim0, dim1),)


def _getitem_backward(grad, a, _saved_a, keyset, args, _kwargs):
    key = args[0]
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        redispatch("setitem", keyset, grad_input, key, grad)
    return (grad_input,)


def _inplace_binary_backward(grad, a, _saved_a, args, _keyset):
    b = args[0]
    grad_a = reduce_grad(grad, a.shape) if getattr(a, "requires_grad", False) else None
    grad_b = reduce_grad(grad, b.shape) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


def _inplace_relu_backward(grad, _a, saved_a, _args, keyset):
    with _grad_context(keyset):
        mask = redispatch("sign", keyset, redispatch("relu", keyset, saved_a))
        grad_input = redispatch("mul", keyset, grad, mask)
        return (grad_input,)


def _inplace_zero_backward(_grad, _a, _saved_a, _args, _keyset):
    return (None,)




def _contiguous_backward(grad, _a, _saved_a, _keyset):
    return (grad,)


def _to_backward(grad, a, _saved_a, keyset, args, _kwargs):
    with _grad_context(keyset):
        return (redispatch("to", keyset, grad, a.device, non_blocking=False),)


def _silu_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        sig = redispatch("sigmoid", keyset, saved_a)
        ones = saved_a._ones_like()
        one_minus_sig = redispatch("add", keyset, ones, redispatch("neg", keyset, sig))
        x_mul = redispatch("mul", keyset, saved_a, one_minus_sig)
        factor = redispatch("mul", keyset, sig, redispatch("add", keyset, ones, x_mul))
        return (redispatch("mul", keyset, grad, factor),)


def _leaky_relu_backward(grad, _a, saved_a, keyset, args, kwargs):
    negative_slope = args[0] if args else kwargs.get("negative_slope", 0.01)
    with _grad_context(keyset):
        pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, saved_a))
        ones = saved_a._ones_like()
        nonpos_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
        slope = _scalar_tensor_like(saved_a, negative_slope)
        factor = redispatch(
            "add",
            keyset,
            pos_mask,
            redispatch("mul", keyset, nonpos_mask, slope),
        )
        return (redispatch("mul", keyset, grad, factor),)


def _elu_backward(grad, _a, saved_a, keyset, args, kwargs):
    alpha = args[0] if args else kwargs.get("alpha", 1.0)
    with _grad_context(keyset):
        pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, saved_a))
        ones = saved_a._ones_like()
        nonpos_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
        alpha_tensor = _scalar_tensor_like(saved_a, alpha)
        exp_x = redispatch("exp", keyset, saved_a)
        neg_branch = redispatch("mul", keyset, alpha_tensor, exp_x)
        factor = redispatch(
            "add",
            keyset,
            pos_mask,
            redispatch("mul", keyset, nonpos_mask, neg_branch),
        )
        return (redispatch("mul", keyset, grad, factor),)


def _mish_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        sp = redispatch("softplus", keyset, saved_a)
        tanh_sp = redispatch("tanh", keyset, sp)
        tanh_sp_sq = redispatch("mul", keyset, tanh_sp, tanh_sp)
        sech2 = redispatch("add", keyset, ones, redispatch("neg", keyset, tanh_sp_sq))
        sig = redispatch("sigmoid", keyset, saved_a)
        tail = redispatch("mul", keyset, saved_a, redispatch("mul", keyset, sech2, sig))
        factor = redispatch("add", keyset, tanh_sp, tail)
        return (redispatch("mul", keyset, grad, factor),)


def _prelu_backward(grad, a, b, saved_a, saved_b, keyset):
    with _grad_context(keyset):
        pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, saved_a))
        ones = saved_a._ones_like()
        nonpos_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))

        grad_a = None
        if getattr(a, "requires_grad", False):
            factor_x = redispatch(
                "add",
                keyset,
                pos_mask,
                redispatch("mul", keyset, nonpos_mask, saved_b),
            )
            grad_a = redispatch("mul", keyset, grad, factor_x)

        grad_b = None
        if getattr(b, "requires_grad", False):
            w_input = redispatch("mul", keyset, nonpos_mask, saved_a)
            grad_b = redispatch("mul", keyset, grad, w_input)
            grad_b = reduce_grad(grad_b, b.shape)

        return grad_a, grad_b


def _abs_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        sign = redispatch("sign", keyset, saved_a)
        return (redispatch("mul", keyset, grad, sign),)


def _neg_backward(grad, _a, _saved_a, keyset):
    with _grad_context(keyset):
        return (redispatch("neg", keyset, grad),)

def _register_autograd_op(name, factory, *, include_meta=True):
    kwargs = {
        "default": factory(),
        "cpu": factory(),
        "npu": factory(),
    }
    if include_meta:
        kwargs["meta"] = factory()
    register_autograd_kernels(name, **kwargs)


for _entry in (
    ("add", lambda: _autograd_binary("add", _add_backward, save_inputs=False)),
    ("mul", lambda: _autograd_binary("mul", _mul_backward)),
    ("matmul", lambda: _autograd_binary("matmul", _matmul_backward)),
    ("div", lambda: _autograd_binary("div", _div_backward)),
    ("true_divide", lambda: _autograd_binary("true_divide", _div_backward)),
    ("sum", lambda: _autograd_unary("sum", _sum_backward, save_input=False)),
    ("mean", lambda: _autograd_unary("mean", _mean_backward, save_input=False)),
    ("relu", lambda: _autograd_unary("relu", _relu_backward, save_input=True)),
    ("reshape", lambda: _autograd_view("reshape", _reshape_backward)),
    ("transpose", lambda: _autograd_view("transpose", _transpose_backward)),
    ("view", lambda: _autograd_view("view", _reshape_backward)),
    ("getitem", lambda: _autograd_unary_args("getitem", _getitem_backward, save_input=False), False),
    ("add_", lambda: _autograd_inplace("add_", _inplace_binary_backward, save_input=True)),
    ("mul_", lambda: _autograd_inplace("mul_", _inplace_binary_backward, save_input=True)),
    ("relu_", lambda: _autograd_inplace("relu_", _inplace_relu_backward, save_input=True)),
    ("zero_", lambda: _autograd_inplace("zero_", _inplace_zero_backward, save_input=False)),
    ("contiguous", lambda: _autograd_unary("contiguous", _contiguous_backward, save_input=False)),
    ("to", lambda: _autograd_unary_args("to", _to_backward, save_input=True)),
    ("silu", lambda: _autograd_unary("silu", _silu_backward)),
    ("leaky_relu", lambda: _autograd_unary_args("leaky_relu", _leaky_relu_backward)),
    ("elu", lambda: _autograd_unary_args("elu", _elu_backward)),
    ("mish", lambda: _autograd_unary("mish", _mish_backward)),
    ("prelu", lambda: _autograd_binary("prelu", _prelu_backward)),
    ("abs", lambda: _autograd_unary("abs", _abs_backward)),
    ("neg", lambda: _autograd_unary("neg", _neg_backward, save_input=False)),
    ("softmax", lambda: _autograd_unary_passthrough("softmax")),
    ("dropout", lambda: _autograd_unary_passthrough("dropout")),
    ("gelu", lambda: _autograd_unary_passthrough("gelu")),
    ("layer_norm", lambda: _autograd_unary_passthrough("layer_norm")),
):
    if len(_entry) == 2:
        _name, _factory = _entry
        _register_autograd_op(_name, _factory)
    else:
        _name, _factory, _include_meta = _entry
        _register_autograd_op(_name, _factory, include_meta=_include_meta)

