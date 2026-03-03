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


# ---------------------------------------------------------------------------
# Phase 1: Basic unary/binary math backward ops
# ---------------------------------------------------------------------------

def _exp_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        exp_x = redispatch("exp", keyset, saved_a)
        return (redispatch("mul", keyset, grad, exp_x),)


def _log_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        return (redispatch("div", keyset, grad, saved_a),)


def _log2_backward(grad, _a, saved_a, keyset):
    import math
    with _grad_context(keyset):
        ln2 = _scalar_tensor_like(saved_a, math.log(2.0))
        denom = redispatch("mul", keyset, saved_a, ln2)
        return (redispatch("div", keyset, grad, denom),)


def _exp2_backward(grad, _a, saved_a, keyset):
    import math
    with _grad_context(keyset):
        ln2 = _scalar_tensor_like(saved_a, math.log(2.0))
        exp2_x = redispatch("exp2", keyset, saved_a)
        factor = redispatch("mul", keyset, exp2_x, ln2)
        return (redispatch("mul", keyset, grad, factor),)


def _sqrt_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        two = _scalar_tensor_like(saved_a, 2.0)
        sqrt_x = redispatch("sqrt", keyset, saved_a)
        denom = redispatch("mul", keyset, two, sqrt_x)
        return (redispatch("div", keyset, grad, denom),)


def _rsqrt_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        neg_half = _scalar_tensor_like(saved_a, -0.5)
        rsqrt_x = redispatch("rsqrt", keyset, saved_a)
        rsqrt_cubed = redispatch("mul", keyset, rsqrt_x, redispatch("mul", keyset, rsqrt_x, rsqrt_x))
        factor = redispatch("mul", keyset, neg_half, rsqrt_cubed)
        return (redispatch("mul", keyset, grad, factor),)


def _sigmoid_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        sig = redispatch("sigmoid", keyset, saved_a)
        ones = saved_a._ones_like()
        one_minus_sig = redispatch("add", keyset, ones, redispatch("neg", keyset, sig))
        factor = redispatch("mul", keyset, sig, one_minus_sig)
        return (redispatch("mul", keyset, grad, factor),)


def _tanh_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        tanh_x = redispatch("tanh", keyset, saved_a)
        tanh_sq = redispatch("mul", keyset, tanh_x, tanh_x)
        ones = saved_a._ones_like()
        factor = redispatch("add", keyset, ones, redispatch("neg", keyset, tanh_sq))
        return (redispatch("mul", keyset, grad, factor),)


def _sin_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        cos_x = redispatch("cos", keyset, saved_a)
        return (redispatch("mul", keyset, grad, cos_x),)


def _cos_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        sin_x = redispatch("sin", keyset, saved_a)
        neg_sin = redispatch("neg", keyset, sin_x)
        return (redispatch("mul", keyset, grad, neg_sin),)


def _erf_backward(grad, _a, saved_a, keyset):
    import math
    with _grad_context(keyset):
        coeff = _scalar_tensor_like(saved_a, 2.0 / math.sqrt(math.pi))
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        neg_x_sq = redispatch("neg", keyset, x_sq)
        exp_val = redispatch("exp", keyset, neg_x_sq)
        factor = redispatch("mul", keyset, coeff, exp_val)
        return (redispatch("mul", keyset, grad, factor),)


def _softplus_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        sig = redispatch("sigmoid", keyset, saved_a)
        return (redispatch("mul", keyset, grad, sig),)


def _pow_backward(grad, a, b, saved_a, saved_b, keyset):
    with _grad_context(keyset):
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            ones = saved_a._ones_like()
            b_minus_1 = redispatch("add", keyset, saved_b, redispatch("neg", keyset, ones))
            a_pow = redispatch("pow", keyset, saved_a, b_minus_1)
            grad_a = redispatch("mul", keyset, grad, redispatch("mul", keyset, saved_b, a_pow))
        if getattr(b, "requires_grad", False):
            a_pow_b = redispatch("pow", keyset, saved_a, saved_b)
            log_a = redispatch("log", keyset, saved_a)
            grad_b = redispatch("mul", keyset, grad, redispatch("mul", keyset, a_pow_b, log_a))
    if grad_a is not None:
        grad_a = reduce_grad(grad_a, a.shape)
    if grad_b is not None:
        grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


# ---------------------------------------------------------------------------
# Phase 2: Fix incorrect passthrough ops
# ---------------------------------------------------------------------------

def _softmax_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = args[0] if args else kwargs.get("dim", -1)
    with _grad_context(keyset):
        s = redispatch("softmax", keyset, saved_a, dim)
        gs = redispatch("mul", keyset, grad, s)
        gs_sum = redispatch("sum", keyset, gs, dim=dim, keepdim=True)
        return (redispatch("mul", keyset, s, redispatch("add", keyset, grad, redispatch("neg", keyset, gs_sum))),)


def _gelu_backward(grad, _a, saved_a, keyset):
    import math
    with _grad_context(keyset):
        sqrt2 = _scalar_tensor_like(saved_a, math.sqrt(2.0))
        x_over_sqrt2 = redispatch("div", keyset, saved_a, sqrt2)
        erf_val = redispatch("erf", keyset, x_over_sqrt2)
        ones = saved_a._ones_like()
        half = _scalar_tensor_like(saved_a, 0.5)
        cdf = redispatch("mul", keyset, half, redispatch("add", keyset, ones, erf_val))
        coeff = _scalar_tensor_like(saved_a, 1.0 / math.sqrt(2.0 * math.pi))
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        neg_half_x_sq = redispatch("mul", keyset, _scalar_tensor_like(saved_a, -0.5), x_sq)
        pdf = redispatch("mul", keyset, coeff, redispatch("exp", keyset, neg_half_x_sq))
        x_pdf = redispatch("mul", keyset, saved_a, pdf)
        factor = redispatch("add", keyset, cdf, x_pdf)
        return (redispatch("mul", keyset, grad, factor),)


def _layer_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    normalized_shape = args[0] if args else kwargs.get("normalized_shape")
    weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
    eps = args[3] if len(args) > 3 else kwargs.get("eps", 1e-5)

    with _grad_context(keyset):
        norm_shape = tuple(normalized_shape)
        ndim = len(saved_a.shape)
        n_norm = len(norm_shape)
        axis_dims = tuple(range(ndim - n_norm, ndim))

        mean = redispatch("mean", keyset, saved_a, dim=axis_dims, keepdim=True)
        diff = redispatch("add", keyset, saved_a, redispatch("neg", keyset, mean))
        var = redispatch("mean", keyset, redispatch("mul", keyset, diff, diff), dim=axis_dims, keepdim=True)
        eps_t = _scalar_tensor_like(saved_a, eps)
        inv_std = redispatch("rsqrt", keyset, redispatch("add", keyset, var, eps_t))
        x_hat = redispatch("mul", keyset, diff, inv_std)

        if weight is not None:
            dl_dxhat = redispatch("mul", keyset, grad, weight)
        else:
            dl_dxhat = grad

        n = 1
        for d in axis_dims:
            n *= saved_a.shape[d]
        n_t = _scalar_tensor_like(saved_a, float(n))

        mean_dl_dxhat = redispatch("div", keyset, redispatch("sum", keyset, dl_dxhat, dim=axis_dims, keepdim=True), n_t)
        mean_dl_dxhat_xhat = redispatch("div", keyset, redispatch("sum", keyset, redispatch("mul", keyset, dl_dxhat, x_hat), dim=axis_dims, keepdim=True), n_t)

        grad_input = redispatch("mul", keyset, inv_std,
            redispatch("add", keyset,
                redispatch("add", keyset, dl_dxhat, redispatch("neg", keyset, mean_dl_dxhat)),
                redispatch("neg", keyset, redispatch("mul", keyset, x_hat, mean_dl_dxhat_xhat))))

        return (grad_input,)


# ---------------------------------------------------------------------------
# Phase 3: View/shape ops backward
# ---------------------------------------------------------------------------

def _squeeze_backward(grad, a, _saved_a, _args, keyset):
    return (redispatch("reshape", keyset, grad, a.shape),)


def _unsqueeze_backward(grad, a, _saved_a, _args, keyset):
    return (redispatch("reshape", keyset, grad, a.shape),)


def _expand_backward(grad, a, _saved_a, _args, keyset):
    return (reduce_grad(grad, a.shape),)


def _permute_backward(grad, _a, _saved_a, args, keyset):
    dims = args[0]
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    return (redispatch("permute", keyset, grad, inv),)


def _narrow_backward(grad, a, _saved_a, keyset, args, _kwargs):
    dim, start, _length = args[0], args[1], args[2]
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        slices = [slice(None)] * len(a.shape)
        d = dim if dim >= 0 else dim + len(a.shape)
        slices[d] = slice(int(start), int(start) + int(_length))
        redispatch("setitem", keyset, grad_input, tuple(slices), grad)
    return (grad_input,)


def _select_backward(grad, a, _saved_a, keyset, args, _kwargs):
    dim, index = args[0], args[1]
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        slices = [slice(None)] * len(a.shape)
        d = dim if dim >= 0 else dim + len(a.shape)
        slices[d] = int(index)
        redispatch("setitem", keyset, grad_input, tuple(slices), redispatch("unsqueeze", keyset, grad, d))
    return (grad_input,)


def _autograd_multi_input(name, backward_impl, *, save_inputs=True):
    def wrapper(tensors, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, tensors, *args, **kwargs)
        any_requires_grad = any(getattr(t, "requires_grad", False) for t in tensors)
        if GradMode.enabled and any_requires_grad:
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                if save_inputs:
                    saved = node_holder["node"].saved_tensors()
                else:
                    saved = tensors
                return backward_impl(grad, tensors, saved, backward_keyset, args, kwargs)

            node = Node(_backward, tuple(tensors))
            node_holder["node"] = node
            if save_inputs:
                node.save_for_backward(*tensors)
            if isinstance(out, tuple):
                for o in out:
                    if hasattr(o, "grad_fn"):
                        o.grad_fn = node
                        o.requires_grad = True
            else:
                out.grad_fn = node
                out.requires_grad = True
        return out

    return wrapper


def _cat_backward(grad, tensors, _saved, keyset, args, _kwargs):
    dim = args[0] if args else _kwargs.get("dim", 0)
    with _grad_context(keyset):
        sizes = [t.shape[dim] for t in tensors]
        grads = redispatch("split", keyset, grad, sizes, dim)
        result = []
        for i, t in enumerate(tensors):
            if getattr(t, "requires_grad", False):
                result.append(grads[i])
            else:
                result.append(None)
        return tuple(result)


def _stack_backward(grad, tensors, _saved, keyset, args, _kwargs):
    dim = args[0] if args else _kwargs.get("dim", 0)
    with _grad_context(keyset):
        grads = redispatch("unbind", keyset, grad, dim)
        result = []
        for i, t in enumerate(tensors):
            if getattr(t, "requires_grad", False):
                result.append(grads[i])
            else:
                result.append(None)
        return tuple(result)


def _autograd_multi_output(name, backward_impl, *, save_input=True):
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        outs = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            new_outs = []
            for idx, o in enumerate(outs):
                node_holder = {}

                def _make_backward(i, holder):
                    def _backward(grad):
                        backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                        if save_input:
                            saved_a = holder["node"].saved_tensors()[0]
                        else:
                            saved_a = a
                        return backward_impl(grad, i, a, saved_a, backward_keyset, args, kwargs)
                    return _backward

                node = Node(_make_backward(idx, node_holder), (a,))
                node_holder["node"] = node
                if save_input:
                    node.save_for_backward(a)
                o.grad_fn = node
                o.requires_grad = True
                new_outs.append(o)
            outs = tuple(new_outs)
        return outs

    return wrapper


def _split_backward(grad, idx, a, _saved_a, keyset, args, _kwargs):
    split_size_or_sections = args[0]
    dim = args[1] if len(args) > 1 else _kwargs.get("dim", 0)
    d = dim if dim >= 0 else dim + len(a.shape)
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        if isinstance(split_size_or_sections, (list, tuple)):
            start = sum(split_size_or_sections[:idx])
        else:
            start = idx * split_size_or_sections
        slices = [slice(None)] * len(a.shape)
        slices[d] = slice(start, start + grad.shape[d])
        redispatch("setitem", keyset, grad_input, tuple(slices), grad)
    return (grad_input,)


def _unbind_backward(grad, idx, a, _saved_a, keyset, args, _kwargs):
    dim = args[0] if args else _kwargs.get("dim", 0)
    d = dim if dim >= 0 else dim + len(a.shape)
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        slices = [slice(None)] * len(a.shape)
        slices[d] = idx
        redispatch("setitem", keyset, grad_input, tuple(slices), redispatch("unsqueeze", keyset, grad, d))
    return (grad_input,)


# ---------------------------------------------------------------------------
# Phase 4: Advanced ops backward
# ---------------------------------------------------------------------------

def _clamp_backward(grad, _a, saved_a, keyset, args, _kwargs):
    min_val = args[0] if len(args) > 0 else _kwargs.get("min_val", None)
    max_val = args[1] if len(args) > 1 else _kwargs.get("max_val", None)
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zeros = redispatch("zeros", keyset, saved_a.shape, dtype=saved_a.dtype, device=saved_a.device)
        mask = ones
        if min_val is not None:
            min_t = _scalar_tensor_like(saved_a, float(min_val))
            ge_min = redispatch("sign", keyset, redispatch("relu", keyset,
                redispatch("add", keyset, saved_a, redispatch("neg", keyset, min_t))))
            eq_min = redispatch("sign", keyset, redispatch("relu", keyset,
                redispatch("add", keyset, min_t, redispatch("neg", keyset, saved_a))))
            eq_min = redispatch("add", keyset, ones, redispatch("neg", keyset,
                redispatch("add", keyset, ge_min, eq_min)))
            ge_min_full = redispatch("add", keyset, ge_min, eq_min)
            mask = redispatch("mul", keyset, mask, ge_min_full)
        if max_val is not None:
            max_t = _scalar_tensor_like(saved_a, float(max_val))
            le_max = redispatch("sign", keyset, redispatch("relu", keyset,
                redispatch("add", keyset, max_t, redispatch("neg", keyset, saved_a))))
            eq_max = redispatch("sign", keyset, redispatch("relu", keyset,
                redispatch("add", keyset, saved_a, redispatch("neg", keyset, max_t))))
            eq_max = redispatch("add", keyset, ones, redispatch("neg", keyset,
                redispatch("add", keyset, le_max, eq_max)))
            le_max_full = redispatch("add", keyset, le_max, eq_max)
            mask = redispatch("mul", keyset, mask, le_max_full)
        return (redispatch("mul", keyset, grad, mask),)


def _hardtanh_backward(grad, _a, saved_a, keyset, args, _kwargs):
    min_val = args[0] if len(args) > 0 else _kwargs.get("min_val", -1.0)
    max_val = args[1] if len(args) > 1 else _kwargs.get("max_val", 1.0)
    with _grad_context(keyset):
        min_t = _scalar_tensor_like(saved_a, float(min_val))
        max_t = _scalar_tensor_like(saved_a, float(max_val))
        ge_min = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, saved_a, redispatch("neg", keyset, min_t))))
        le_max = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, max_t, redispatch("neg", keyset, saved_a))))
        mask = redispatch("mul", keyset, ge_min, le_max)
        ones = saved_a._ones_like()
        eq_min = redispatch("add", keyset, ones, redispatch("neg", keyset,
            redispatch("add", keyset, ge_min,
                redispatch("sign", keyset, redispatch("relu", keyset,
                    redispatch("add", keyset, min_t, redispatch("neg", keyset, saved_a)))))))
        eq_max = redispatch("add", keyset, ones, redispatch("neg", keyset,
            redispatch("add", keyset, le_max,
                redispatch("sign", keyset, redispatch("relu", keyset,
                    redispatch("add", keyset, saved_a, redispatch("neg", keyset, max_t)))))))
        mask = redispatch("add", keyset, mask, redispatch("add", keyset, eq_min, eq_max))
        return (redispatch("mul", keyset, grad, mask),)


def _relu6_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        six = _scalar_tensor_like(saved_a, 6.0)
        pos_mask = redispatch("sign", keyset, redispatch("relu", keyset, saved_a))
        le6_mask = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, six, redispatch("neg", keyset, saved_a))))
        ones = saved_a._ones_like()
        eq6 = redispatch("add", keyset, ones, redispatch("neg", keyset,
            redispatch("add", keyset, le6_mask,
                redispatch("sign", keyset, redispatch("relu", keyset,
                    redispatch("add", keyset, saved_a, redispatch("neg", keyset, six)))))))
        mask = redispatch("mul", keyset, pos_mask, redispatch("add", keyset, le6_mask, eq6))
        eq0 = redispatch("add", keyset, ones, redispatch("neg", keyset,
            redispatch("add", keyset, pos_mask,
                redispatch("sign", keyset, redispatch("relu", keyset, redispatch("neg", keyset, saved_a))))))
        mask = redispatch("add", keyset, mask, eq0)
        return (redispatch("mul", keyset, grad, mask),)


def _log_softmax_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = args[0] if args else kwargs.get("dim", -1)
    with _grad_context(keyset):
        log_s = redispatch("log_softmax", keyset, saved_a, dim)
        s = redispatch("exp", keyset, log_s)
        grad_sum = redispatch("sum", keyset, grad, dim=dim, keepdim=True)
        return (redispatch("add", keyset, grad, redispatch("neg", keyset, redispatch("mul", keyset, s, grad_sum))),)


def _batch_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    weight = args[3] if len(args) > 3 else kwargs.get("weight", None)
    training = args[5] if len(args) > 5 else kwargs.get("training", False)
    eps = args[7] if len(args) > 7 else kwargs.get("eps", 1e-5)

    with _grad_context(keyset):
        ndim = len(saved_a.shape)
        axes = (0,) + tuple(range(2, ndim))
        shape_for_stats = [1, saved_a.shape[1]] + [1] * (ndim - 2)

        mean = redispatch("mean", keyset, saved_a, dim=axes, keepdim=True)
        diff = redispatch("add", keyset, saved_a, redispatch("neg", keyset, mean))
        var = redispatch("mean", keyset, redispatch("mul", keyset, diff, diff), dim=axes, keepdim=True)
        eps_t = _scalar_tensor_like(saved_a, eps)
        inv_std = redispatch("rsqrt", keyset, redispatch("add", keyset, var, eps_t))
        x_hat = redispatch("mul", keyset, diff, inv_std)

        if weight is not None:
            dl_dxhat = redispatch("mul", keyset, grad, weight.reshape(shape_for_stats))
        else:
            dl_dxhat = grad

        n = 1
        for ax in axes:
            n *= saved_a.shape[ax]
        n_t = _scalar_tensor_like(saved_a, float(n))

        mean_dl_dxhat = redispatch("div", keyset, redispatch("sum", keyset, dl_dxhat, dim=axes, keepdim=True), n_t)
        mean_dl_dxhat_xhat = redispatch("div", keyset, redispatch("sum", keyset, redispatch("mul", keyset, dl_dxhat, x_hat), dim=axes, keepdim=True), n_t)

        grad_input = redispatch("mul", keyset, inv_std,
            redispatch("add", keyset,
                redispatch("add", keyset, dl_dxhat, redispatch("neg", keyset, mean_dl_dxhat)),
                redispatch("neg", keyset, redispatch("mul", keyset, x_hat, mean_dl_dxhat_xhat))))

        return (grad_input,)


def _autograd_embedding(name, backward_impl):
    def wrapper(weight, indices, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, weight, indices, *args, **kwargs)
        if GradMode.enabled and getattr(weight, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_w, saved_idx = node_holder["node"].saved_tensors()
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, weight, indices, saved_w, saved_idx, backward_keyset)

            node = Node(_backward, (weight, indices))
            node_holder["node"] = node
            node.save_for_backward(weight, indices)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _embedding_backward(grad, weight, _indices, saved_weight, saved_indices, keyset):
    with _grad_context(keyset):
        grad_weight = redispatch("zeros", keyset, saved_weight.shape, dtype=saved_weight.dtype, device=saved_weight.device)
        num_indices = saved_indices.numel()
        emb_dim = saved_weight.shape[-1]
        flat_idx = redispatch("reshape", keyset, saved_indices, (num_indices,))
        flat_grad = redispatch("reshape", keyset, grad, (num_indices, emb_dim))
        redispatch("index_add_", keyset, grad_weight, 0, flat_idx, flat_grad)
    return (grad_weight, None)


def _autograd_where(name, backward_impl):
    def wrapper(cond, x, y):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, cond, x, y)
        x_rg = getattr(x, "requires_grad", False)
        y_rg = getattr(y, "requires_grad", False)
        if GradMode.enabled and (x_rg or y_rg):
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, cond, x, y, saved[0], backward_keyset)

            node = Node(_backward, (cond, x, y))
            node_holder["node"] = node
            node.save_for_backward(cond)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _where_backward(grad, cond, x, y, saved_cond, keyset):
    with _grad_context(keyset):
        zeros = redispatch("zeros", keyset, grad.shape, dtype=grad.dtype, device=grad.device)
        grad_x = None
        grad_y = None
        if getattr(x, "requires_grad", False):
            grad_x = redispatch("where", keyset, saved_cond, grad, zeros)
            grad_x = reduce_grad(grad_x, x.shape)
        if getattr(y, "requires_grad", False):
            grad_y = redispatch("where", keyset, saved_cond, zeros, grad)
            grad_y = reduce_grad(grad_y, y.shape)
    return None, grad_x, grad_y


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
    ("pow", lambda: _autograd_binary("pow", _pow_backward)),
    ("sum", lambda: _autograd_unary("sum", _sum_backward, save_input=False)),
    ("mean", lambda: _autograd_unary("mean", _mean_backward, save_input=False)),
    ("relu", lambda: _autograd_unary("relu", _relu_backward, save_input=True)),
    ("reshape", lambda: _autograd_view("reshape", _reshape_backward)),
    ("transpose", lambda: _autograd_view("transpose", _transpose_backward)),
    ("view", lambda: _autograd_view("view", _reshape_backward)),
    ("squeeze", lambda: _autograd_view("squeeze", _squeeze_backward)),
    ("unsqueeze", lambda: _autograd_view("unsqueeze", _unsqueeze_backward)),
    ("expand", lambda: _autograd_view("expand", _expand_backward)),
    ("permute", lambda: _autograd_view("permute", _permute_backward)),
    ("getitem", lambda: _autograd_unary_args("getitem", _getitem_backward, save_input=False), False),
    ("narrow", lambda: _autograd_unary_args("narrow", _narrow_backward, save_input=False)),
    ("select", lambda: _autograd_unary_args("select", _select_backward, save_input=False)),
    ("add_", lambda: _autograd_inplace("add_", _inplace_binary_backward, save_input=True)),
    ("mul_", lambda: _autograd_inplace("mul_", _inplace_binary_backward, save_input=True)),
    ("relu_", lambda: _autograd_inplace("relu_", _inplace_relu_backward, save_input=True)),
    ("zero_", lambda: _autograd_inplace("zero_", _inplace_zero_backward, save_input=False)),
    ("contiguous", lambda: _autograd_unary("contiguous", _contiguous_backward, save_input=False)),
    ("to", lambda: _autograd_unary_args("to", _to_backward, save_input=True)),
    ("exp", lambda: _autograd_unary("exp", _exp_backward)),
    ("log", lambda: _autograd_unary("log", _log_backward)),
    ("log2", lambda: _autograd_unary("log2", _log2_backward)),
    ("exp2", lambda: _autograd_unary("exp2", _exp2_backward)),
    ("sqrt", lambda: _autograd_unary("sqrt", _sqrt_backward)),
    ("rsqrt", lambda: _autograd_unary("rsqrt", _rsqrt_backward)),
    ("sigmoid", lambda: _autograd_unary("sigmoid", _sigmoid_backward)),
    ("tanh", lambda: _autograd_unary("tanh", _tanh_backward)),
    ("sin", lambda: _autograd_unary("sin", _sin_backward)),
    ("cos", lambda: _autograd_unary("cos", _cos_backward)),
    ("erf", lambda: _autograd_unary("erf", _erf_backward)),
    ("softplus", lambda: _autograd_unary("softplus", _softplus_backward)),
    ("silu", lambda: _autograd_unary("silu", _silu_backward)),
    ("leaky_relu", lambda: _autograd_unary_args("leaky_relu", _leaky_relu_backward)),
    ("elu", lambda: _autograd_unary_args("elu", _elu_backward)),
    ("mish", lambda: _autograd_unary("mish", _mish_backward)),
    ("prelu", lambda: _autograd_binary("prelu", _prelu_backward)),
    ("abs", lambda: _autograd_unary("abs", _abs_backward)),
    ("neg", lambda: _autograd_unary("neg", _neg_backward, save_input=False)),
    ("relu6", lambda: _autograd_unary("relu6", _relu6_backward)),
    ("gelu", lambda: _autograd_unary("gelu", _gelu_backward)),
    ("softmax", lambda: _autograd_unary_args("softmax", _softmax_backward)),
    ("log_softmax", lambda: _autograd_unary_args("log_softmax", _log_softmax_backward)),
    ("layer_norm", lambda: _autograd_unary_args("layer_norm", _layer_norm_backward)),
    ("batch_norm", lambda: _autograd_unary_args("batch_norm", _batch_norm_backward)),
    ("clamp", lambda: _autograd_unary_args("clamp", _clamp_backward)),
    ("hardtanh", lambda: _autograd_unary_args("hardtanh", _hardtanh_backward)),
    ("dropout", lambda: _autograd_unary_passthrough("dropout")),
    ("cat", lambda: _autograd_multi_input("cat", _cat_backward, save_inputs=False)),
    ("stack", lambda: _autograd_multi_input("stack", _stack_backward, save_inputs=False)),
    ("embedding", lambda: _autograd_embedding("embedding", _embedding_backward)),
    ("where", lambda: _autograd_where("where", _where_backward)),
    ("split", lambda: _autograd_multi_output("split", _split_backward, save_input=False)),
    ("unbind", lambda: _autograd_multi_output("unbind", _unbind_backward, save_input=False)),
):
    if len(_entry) == 2:
        _name, _factory = _entry
        _register_autograd_op(_name, _factory)
    else:
        _name, _factory, _include_meta = _entry
        _register_autograd_op(_name, _factory, include_meta=_include_meta)

