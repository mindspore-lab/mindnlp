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


# ---- Dropout backward (uses saved mask from forward) ----

def _autograd_dropout():
    """Proper dropout backward using the mask saved during forward."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch("dropout", raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            backward_data = getattr(out, "_backward_data", None)
            node_holder = {}

            def _backward(grad):
                if backward_data is not None and grad.device.type == "npu":
                    # NPU path: use aclnnDropoutDoMask with saved mask
                    from .npu import aclnn, runtime as npu_runtime, state as npu_state
                    runtime = npu_runtime.get_runtime((grad.device.index or 0))
                    stream = npu_state.current_stream((grad.device.index or 0))
                    out_shape = grad.shape
                    out_stride = npu_runtime._contiguous_stride(out_shape)
                    out_numel = 1
                    for d in out_shape:
                        out_numel *= d
                    from .npu.ops import _dtype_itemsize, _unwrap_storage
                    itemsize = _dtype_itemsize(grad.dtype)
                    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)
                    grad_ptr = _unwrap_storage(grad).data_ptr()
                    aclnn.dropout_do_mask(
                        grad_ptr,
                        backward_data["mask_ptr"],
                        out_ptr,
                        out_shape, grad.stride, grad.dtype,
                        backward_data["mask_numel"],
                        backward_data["p"],
                        runtime, stream=stream.stream,
                    )
                    from .npu.ops import npu_typed_storage_from_ptr, _wrap_tensor
                    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1),
                                                            grad.dtype, device=grad.device)
                    return (_wrap_tensor(out_storage, out_shape, out_stride),)
                # CPU fallback: apply mask * scale
                from .cpu.ops import _to_numpy, _from_numpy
                import numpy as _np
                g_np = _to_numpy(grad)
                p = args[0] if args else kwargs.get("p", 0.5)
                training = args[1] if len(args) > 1 else kwargs.get("training", True)
                if not training or p == 0:
                    return (grad,)
                # Without the mask we cannot compute the exact backward;
                # use the output to infer which elements were zeroed
                out_np = _to_numpy(out)
                a_np = _to_numpy(a)
                mask_np = _np.where(_np.abs(a_np) > 0, (out_np != 0).astype(g_np.dtype), 1.0)
                scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
                grad_np = g_np * mask_np * scale
                return (_from_numpy(grad_np, grad.dtype, grad.device),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


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


# ---------------------------------------------------------------------------
# Phase 5: Transformer-critical ops backward
# ---------------------------------------------------------------------------

def _masked_fill_backward(grad, _a, saved_a, keyset, args, kwargs):
    mask = args[0]
    with _grad_context(keyset):
        zeros = redispatch("zeros", keyset, grad.shape, dtype=grad.dtype, device=grad.device)
        from .._creation import tensor
        mask_bool = mask
        grad_input = redispatch("where", keyset, mask_bool, zeros, grad)
        return (grad_input,)


def _var_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = kwargs.get("dim", None)
    unbiased = kwargs.get("unbiased", True)
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        if dim is not None:
            mean_val = redispatch("mean", keyset, saved_a, dim=dim, keepdim=True)
        else:
            mean_val = redispatch("mean", keyset, saved_a)
        diff = redispatch("add", keyset, saved_a, redispatch("neg", keyset, mean_val))
        two = _scalar_tensor_like(saved_a, 2.0)
        if dim is not None:
            if isinstance(dim, (list, tuple)):
                n = 1
                for d in dim:
                    n *= saved_a.shape[d if d >= 0 else d + len(saved_a.shape)]
            else:
                n = saved_a.shape[dim if dim >= 0 else dim + len(saved_a.shape)]
        else:
            n = saved_a.numel()
        correction = 1 if unbiased else 0
        denom = _scalar_tensor_like(saved_a, float(n - correction))
        factor = redispatch("div", keyset, redispatch("mul", keyset, two, diff), denom)
        if not keepdim and dim is not None:
            if isinstance(dim, int):
                grad = redispatch("unsqueeze", keyset, grad, dim)
            else:
                for d in sorted(dim if isinstance(dim, (list, tuple)) else [dim]):
                    grad = redispatch("unsqueeze", keyset, grad, d)
        ones = saved_a._ones_like()
        grad_expanded = redispatch("mul", keyset, grad, ones)
        return (redispatch("mul", keyset, grad_expanded, factor),)


def _std_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)
    unbiased = kwargs.get("unbiased", True)
    with _grad_context(keyset):
        if dim is not None:
            mean_val = redispatch("mean", keyset, saved_a, dim=dim, keepdim=True)
            std_val = redispatch("std", keyset, saved_a, dim=dim, keepdim=True, unbiased=unbiased)
        else:
            mean_val = redispatch("mean", keyset, saved_a)
            std_val = redispatch("std", keyset, saved_a, unbiased=unbiased)
        diff = redispatch("add", keyset, saved_a, redispatch("neg", keyset, mean_val))
        if dim is not None:
            if isinstance(dim, (list, tuple)):
                n = 1
                for d in dim:
                    n *= saved_a.shape[d if d >= 0 else d + len(saved_a.shape)]
            else:
                n = saved_a.shape[dim if dim >= 0 else dim + len(saved_a.shape)]
        else:
            n = saved_a.numel()
        correction = 1 if unbiased else 0
        n_corr = _scalar_tensor_like(saved_a, float(n - correction))
        factor = redispatch("div", keyset, diff, redispatch("mul", keyset, n_corr, std_val))
        if not keepdim and dim is not None:
            if isinstance(dim, int):
                grad = redispatch("unsqueeze", keyset, grad, dim)
            else:
                for d in sorted(dim if isinstance(dim, (list, tuple)) else [dim]):
                    grad = redispatch("unsqueeze", keyset, grad, d)
        ones = saved_a._ones_like()
        grad_expanded = redispatch("mul", keyset, grad, ones)
        return (redispatch("mul", keyset, grad_expanded, factor),)


def _gather_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = args[0]
    index = args[1]
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, saved_a.shape, dtype=saved_a.dtype, device=saved_a.device)
        redispatch("scatter_add_", keyset, grad_input, dim, index, grad)
        return (grad_input,)


def _index_select_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = args[0]
    index = args[1]
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, saved_a.shape, dtype=saved_a.dtype, device=saved_a.device)
        redispatch("index_add_", keyset, grad_input, dim, index, grad)
        return (grad_input,)


def _repeat_backward(grad, _a, saved_a, keyset, args, kwargs):
    repeats = args[0]
    with _grad_context(keyset):
        input_shape = saved_a.shape
        ndim_input = len(input_shape)
        ndim_repeat = len(repeats)
        ndim = max(ndim_input, ndim_repeat)
        padded_shape = [1] * (ndim - ndim_input) + list(input_shape)
        padded_repeats = [1] * (ndim - ndim_repeat) + list(repeats)
        reshape_dims = []
        for s, r in zip(padded_shape, padded_repeats):
            reshape_dims.extend([r, s])
        grad_reshaped = redispatch("reshape", keyset, grad, tuple(reshape_dims))
        sum_dims = tuple(range(0, 2 * ndim, 2))
        for d in sum_dims:
            grad_reshaped = redispatch("sum", keyset, grad_reshaped, dim=d, keepdim=True)
        grad_out = redispatch("reshape", keyset, grad_reshaped, input_shape)
        return (grad_out,)


def _tril_backward(grad, _a, _saved_a, keyset, args, kwargs):
    with _grad_context(keyset):
        return (redispatch("tril", keyset, grad),)


def _triu_backward(grad, _a, _saved_a, keyset, args, kwargs):
    with _grad_context(keyset):
        return (redispatch("triu", keyset, grad),)


def _group_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    num_groups = args[0] if args else kwargs.get("num_groups")
    weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
    eps = args[3] if len(args) > 3 else kwargs.get("eps", 1e-5)

    with _grad_context(keyset):
        N = saved_a.shape[0]
        C = saved_a.shape[1]
        spatial = saved_a.shape[2:]
        channels_per_group = C // num_groups
        group_size = channels_per_group
        for s in spatial:
            group_size *= s

        reshaped = redispatch("reshape", keyset, saved_a, (N, num_groups, channels_per_group, *spatial))
        axes = tuple(range(2, len(reshaped.shape)))
        mean = redispatch("mean", keyset, reshaped, dim=axes, keepdim=True)
        diff = redispatch("add", keyset, reshaped, redispatch("neg", keyset, mean))
        var = redispatch("mean", keyset, redispatch("mul", keyset, diff, diff), dim=axes, keepdim=True)
        eps_t = _scalar_tensor_like(saved_a, eps)
        inv_std = redispatch("rsqrt", keyset, redispatch("add", keyset, var, eps_t))
        x_hat = redispatch("mul", keyset, diff, inv_std)

        grad_reshaped = redispatch("reshape", keyset, grad, (N, num_groups, channels_per_group, *spatial))
        if weight is not None:
            w_shape = [1, num_groups, channels_per_group] + [1] * len(spatial)
            w_reshaped = redispatch("reshape", keyset, weight, tuple(w_shape))
            dl_dxhat = redispatch("mul", keyset, grad_reshaped, w_reshaped)
        else:
            dl_dxhat = grad_reshaped

        n_t = _scalar_tensor_like(saved_a, float(group_size))
        mean_dl_dxhat = redispatch("div", keyset, redispatch("sum", keyset, dl_dxhat, dim=axes, keepdim=True), n_t)
        mean_dl_dxhat_xhat = redispatch("div", keyset, redispatch("sum", keyset, redispatch("mul", keyset, dl_dxhat, x_hat), dim=axes, keepdim=True), n_t)

        grad_reshaped_out = redispatch("mul", keyset, inv_std,
            redispatch("add", keyset,
                redispatch("add", keyset, dl_dxhat, redispatch("neg", keyset, mean_dl_dxhat)),
                redispatch("neg", keyset, redispatch("mul", keyset, x_hat, mean_dl_dxhat_xhat))))

        grad_input = redispatch("reshape", keyset, grad_reshaped_out, saved_a.shape)
        return (grad_input,)


def _rms_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    normalized_shape = args[0] if args else kwargs.get("normalized_shape")
    weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
    eps = args[2] if len(args) > 2 else kwargs.get("eps", 1e-6)

    with _grad_context(keyset):
        norm_shape = tuple(normalized_shape)
        ndim = len(saved_a.shape)
        n_norm = len(norm_shape)
        axis_dims = tuple(range(ndim - n_norm, ndim))

        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        variance = redispatch("mean", keyset, x_sq, dim=axis_dims, keepdim=True)
        eps_t = _scalar_tensor_like(saved_a, eps)
        rms = redispatch("sqrt", keyset, redispatch("add", keyset, variance, eps_t))
        x_hat = redispatch("div", keyset, saved_a, rms)

        if weight is not None:
            dl_dxhat = redispatch("mul", keyset, grad, weight)
        else:
            dl_dxhat = grad

        n = 1
        for d in axis_dims:
            n *= saved_a.shape[d]
        n_t = _scalar_tensor_like(saved_a, float(n))

        dot = redispatch("sum", keyset, redispatch("mul", keyset, dl_dxhat, x_hat), dim=axis_dims, keepdim=True)
        grad_input = redispatch("div", keyset,
            redispatch("add", keyset, dl_dxhat,
                redispatch("neg", keyset, redispatch("mul", keyset, x_hat,
                    redispatch("div", keyset, dot, n_t)))),
            rms)
        return (grad_input,)


def _flip_backward(grad, _a, _saved_a, keyset, args, kwargs):
    dims = args[0]
    with _grad_context(keyset):
        return (redispatch("flip", keyset, grad, dims),)


def _cumsum_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = args[0] if args else kwargs.get("dim", 0)
    with _grad_context(keyset):
        flipped = redispatch("flip", keyset, grad, [dim])
        cum = redispatch("cumsum", keyset, flipped, dim)
        return (redispatch("flip", keyset, cum, [dim]),)


def _pad_backward(grad, _a, saved_a, keyset, args, kwargs):
    pad_widths = args[0]
    with _grad_context(keyset):
        ndim = len(saved_a.shape)
        n_pairs = len(pad_widths) // 2
        slices = [slice(None)] * ndim
        for i in range(n_pairs):
            dim = ndim - 1 - i
            left = int(pad_widths[2 * i])
            right = int(pad_widths[2 * i + 1])
            dim_size = grad.shape[dim]
            start = max(left, 0)
            end = dim_size - max(right, 0)
            slices[dim] = slice(start, end)
        grad_input = redispatch("getitem", keyset, grad, tuple(slices))
        return (grad_input,)


def _prod_backward(grad, _a, saved_a, keyset, args, kwargs):
    with _grad_context(keyset):
        dim = kwargs.get("dim", None)
        keepdim = kwargs.get("keepdim", False)
        # Compute prod with keepdim=True for proper broadcasting
        if dim is not None:
            prod_val = redispatch("prod", keyset, saved_a, dim=dim, keepdim=True)
        else:
            prod_val = redispatch("prod", keyset, saved_a)
        factor = redispatch("div", keyset, prod_val, saved_a)
        if not keepdim and dim is not None:
            if isinstance(dim, int):
                grad = redispatch("unsqueeze", keyset, grad, dim)
        ones = saved_a._ones_like()
        grad_expanded = redispatch("mul", keyset, grad, ones)
        return (redispatch("mul", keyset, grad_expanded, factor),)


def _norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    p = args[0] if args else kwargs.get("p", 2)
    dim = kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        norm_val = redispatch("norm", keyset, saved_a, p, dim=dim, keepdim=True)
        eps_t = _scalar_tensor_like(saved_a, 1e-12)
        safe_norm = redispatch("add", keyset, norm_val, eps_t)
        if p == 2 or p == 2.0:
            factor = redispatch("div", keyset, saved_a, safe_norm)
        else:
            p_t = _scalar_tensor_like(saved_a, float(p))
            ones = saved_a._ones_like()
            p_minus_1 = _scalar_tensor_like(saved_a, float(p - 1))
            abs_x = redispatch("abs", keyset, saved_a)
            sign_x = redispatch("sign", keyset, saved_a)
            abs_pow = redispatch("pow", keyset, abs_x, p_minus_1)
            norm_pow_p_minus_1 = redispatch("pow", keyset, safe_norm, p_minus_1)
            factor = redispatch("div", keyset, redispatch("mul", keyset, sign_x, abs_pow), norm_pow_p_minus_1)
        if not keepdim and dim is not None:
            if isinstance(dim, int):
                grad = redispatch("unsqueeze", keyset, grad, dim)
            elif isinstance(dim, (list, tuple)):
                for d in sorted(dim):
                    grad = redispatch("unsqueeze", keyset, grad, d)
        ones = saved_a._ones_like()
        grad_expanded = redispatch("mul", keyset, grad, ones)
        return (redispatch("mul", keyset, grad_expanded, factor),)


# ---------------------------------------------------------------------------
# Phase 6: Conv/pool backward ops
# ---------------------------------------------------------------------------

def _autograd_conv(name, *, has_bias=True):
    """Autograd wrapper for convolution ops that need weight + optional bias gradients."""
    def wrapper(input, weight, bias=None, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, input, weight, bias, *args, **kwargs)
        any_rg = (getattr(input, "requires_grad", False) or
                  getattr(weight, "requires_grad", False) or
                  (bias is not None and getattr(bias, "requires_grad", False)))
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                saved_input, saved_weight = saved[0], saved[1]
                saved_bias = saved[2] if len(saved) > 2 else None
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _conv_backward(name, grad, input, weight, bias,
                                     saved_input, saved_weight, saved_bias,
                                     backward_keyset, args, kwargs)

            inputs = (input, weight) if bias is None else (input, weight, bias)
            node = Node(_backward, inputs)
            node_holder["node"] = node
            node.save_for_backward(*inputs)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _conv_backward(name, grad, input, weight, bias, saved_input, saved_weight, saved_bias, keyset, args, kwargs):
    with _grad_context(keyset):
        # Use numpy for backward — simpler and always correct on CPU
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        weight_np = _to_numpy(saved_weight)

        is_1d = name in ("conv1d", "conv_transpose1d")
        is_transpose = name in ("conv_transpose2d", "conv_transpose1d")

        stride = kwargs.get("stride", args[0] if args else (1, 1))
        padding = kwargs.get("padding", args[1] if len(args) > 1 else (0, 0))
        dilation = kwargs.get("dilation", args[2] if len(args) > 2 else (1, 1))
        groups = kwargs.get("groups", args[3] if len(args) > 3 else 1)

        if is_1d:
            # Unsqueeze to 2D
            grad_np = grad_np[:, :, np.newaxis, :]
            input_np = input_np[:, :, np.newaxis, :]
            weight_np = weight_np[:, :, np.newaxis, :]
            stride = (1, stride[0]) if isinstance(stride, tuple) else (1, stride)
            padding = (0, padding[0]) if isinstance(padding, tuple) else (0, padding)
            dilation = (1, dilation[0]) if isinstance(dilation, tuple) else (1, dilation)

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        sH, sW = stride
        pH, pW = padding
        dH, dW = dilation

        grad_input_np = None
        grad_weight_np = None
        grad_bias_np = None

        if is_transpose:
            # conv_transpose forward: output = conv_transpose(input, weight)
            # conv_transpose backward for input: grad_input = conv2d(grad, weight)
            # conv_transpose backward for weight: grad_weight = conv2d(input.T, grad)
            N, C_in, H_in, W_in = input_np.shape
            C_in_w, C_out_g, kH, kW = weight_np.shape
            C_out = C_out_g * groups

            if getattr(input, "requires_grad", False):
                # grad_input = conv2d(grad_output, weight, stride, padding, dilation, groups)
                # This is just the forward conv with the grad as input
                grad_input_np = np.zeros_like(input_np)
                if pH > 0 or pW > 0:
                    grad_padded = np.pad(grad_np, ((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
                else:
                    grad_padded = grad_np
                for g in range(groups):
                    c_in_per_g = C_in // groups
                    c_out_per_g = C_out_g
                    for ci_local in range(c_in_per_g):
                        ci = g * c_in_per_g + ci_local
                        for co_local in range(c_out_per_g):
                            co = g * c_out_per_g + co_local
                            kernel = weight_np[ci_local, co_local]
                            for ih in range(H_in):
                                for iw in range(W_in):
                                    val = 0.0
                                    for kh in range(kH):
                                        for kw in range(kW):
                                            oh = ih * sH + kh * dH
                                            ow = iw * sW + kw * dW
                                            if oh < grad_padded.shape[2] and ow < grad_padded.shape[3]:
                                                val += grad_padded[:, co, oh, ow] * kernel[kh, kw]
                                    grad_input_np[:, ci, ih, iw] += val

            if getattr(weight, "requires_grad", False):
                grad_weight_np = np.zeros_like(weight_np)
                if pH > 0 or pW > 0:
                    grad_padded = np.pad(grad_np, ((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
                else:
                    grad_padded = grad_np
                for g in range(groups):
                    c_in_per_g = C_in // groups
                    c_out_per_g = C_out_g
                    for ci_local in range(c_in_per_g):
                        ci = g * c_in_per_g + ci_local
                        for co_local in range(c_out_per_g):
                            co = g * c_out_per_g + co_local
                            for kh in range(kH):
                                for kw in range(kW):
                                    val = 0.0
                                    for ih in range(H_in):
                                        for iw in range(W_in):
                                            oh = ih * sH + kh * dH
                                            ow = iw * sW + kw * dW
                                            if oh < grad_padded.shape[2] and ow < grad_padded.shape[3]:
                                                val += (input_np[:, ci, ih, iw] * grad_padded[:, co, oh, ow]).sum()
                                    grad_weight_np[ci_local, co_local, kh, kw] = val
        else:
            # Standard conv2d backward
            N, C_in, H_in, W_in = input_np.shape
            C_out, C_in_g, kH, kW = weight_np.shape
            _, _, H_out, W_out = grad_np.shape

            if pH > 0 or pW > 0:
                input_padded = np.pad(input_np, ((0,0),(0,0),(pH,pH),(pW,pW)), mode='constant')
            else:
                input_padded = input_np

            if getattr(input, "requires_grad", False):
                grad_input_padded = np.zeros_like(input_padded)
                for g in range(groups):
                    c_out_per_g = C_out // groups
                    for co_local in range(c_out_per_g):
                        co = g * c_out_per_g + co_local
                        for ci_local in range(C_in_g):
                            ci = g * C_in_g + ci_local
                            kernel = weight_np[co, ci_local]
                            for oh in range(H_out):
                                for ow in range(W_out):
                                    for kh in range(kH):
                                        for kw in range(kW):
                                            ih = oh * sH + kh * dH
                                            iw = ow * sW + kw * dW
                                            grad_input_padded[:, ci, ih, iw] += grad_np[:, co, oh, ow] * kernel[kh, kw]
                if pH > 0 or pW > 0:
                    grad_input_np = grad_input_padded[:, :, pH:pH+H_in, pW:pW+W_in]
                else:
                    grad_input_np = grad_input_padded

            if getattr(weight, "requires_grad", False):
                grad_weight_np = np.zeros_like(weight_np)
                for g in range(groups):
                    c_out_per_g = C_out // groups
                    for co_local in range(c_out_per_g):
                        co = g * c_out_per_g + co_local
                        for ci_local in range(C_in_g):
                            ci = g * C_in_g + ci_local
                            for kh in range(kH):
                                for kw in range(kW):
                                    val = 0.0
                                    for oh in range(H_out):
                                        for ow in range(W_out):
                                            ih = oh * sH + kh * dH
                                            iw = ow * sW + kw * dW
                                            val += (input_padded[:, ci, ih, iw] * grad_np[:, co, oh, ow]).sum()
                                    grad_weight_np[co, ci_local, kh, kw] = val

        if bias is not None and getattr(bias, "requires_grad", False):
            if is_1d:
                grad_bias_np = grad_np[:, :, 0, :].sum(axis=(0, 2))
            else:
                grad_bias_np = grad_np.sum(axis=(0, 2, 3))

        # Convert back
        grad_input_t = None
        if grad_input_np is not None:
            if is_1d:
                grad_input_np = grad_input_np[:, :, 0, :]
            grad_input_t = _from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)), input.dtype, input.device)

        grad_weight_t = None
        if grad_weight_np is not None:
            if is_1d:
                grad_weight_np = grad_weight_np[:, :, 0, :]
            grad_weight_t = _from_numpy(np.ascontiguousarray(grad_weight_np.astype(weight_np.dtype)), weight.dtype, weight.device)

        grad_bias_t = None
        if grad_bias_np is not None:
            grad_bias_t = _from_numpy(np.ascontiguousarray(grad_bias_np.astype(input_np.dtype)), bias.dtype, bias.device)

        if bias is not None:
            return grad_input_t, grad_weight_t, grad_bias_t
        return grad_input_t, grad_weight_t


def _autograd_pool(name, backward_impl):
    """Autograd wrapper for pooling ops."""
    def wrapper(input, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, input, *args, **kwargs)
        if GradMode.enabled and getattr(input, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_input = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, input, saved_input, out, backward_keyset, args, kwargs)

            node = Node(_backward, (input,))
            node_holder["node"] = node
            node.save_for_backward(input)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _max_pool2d_backward(grad, input, saved_input, out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        import math

        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        out_np = _to_numpy(out)

        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        dilation = args[3] if len(args) > 3 else kwargs.get("dilation", 1)

        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
        pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)
        dH, dW = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

        N, C, H, W = input_np.shape
        _, _, H_out, W_out = grad_np.shape

        if pH > 0 or pW > 0:
            input_padded = np.pad(input_np, ((0,0),(0,0),(pH,pH),(pW,pW)),
                                  mode='constant', constant_values=-np.inf)
        else:
            input_padded = input_np

        grad_input_padded = np.zeros_like(input_padded)
        for oh in range(H_out):
            for ow in range(W_out):
                for kh in range(kH):
                    for kw in range(kW):
                        ih = oh * sH + kh * dH
                        iw = ow * sW + kw * dW
                        if ih < input_padded.shape[2] and iw < input_padded.shape[3]:
                            mask = (input_padded[:, :, ih, iw] == out_np[:, :, oh, ow])
                            grad_input_padded[:, :, ih, iw] += grad_np[:, :, oh, ow] * mask

        if pH > 0 or pW > 0:
            grad_input_np = grad_input_padded[:, :, pH:pH+H, pW:pW+W]
        else:
            grad_input_np = grad_input_padded

        grad_input = _from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                                 input.dtype, input.device)
        return (grad_input,)


def _avg_pool2d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)

        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        count_include_pad = args[4] if len(args) > 4 else kwargs.get("count_include_pad", True)
        divisor_override = args[5] if len(args) > 5 else kwargs.get("divisor_override", None)

        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
        pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)

        N, C, H, W = input_np.shape
        _, _, H_out, W_out = grad_np.shape

        if pH > 0 or pW > 0:
            grad_input_padded = np.zeros((N, C, H + 2*pH, W + 2*pW), dtype=input_np.dtype)
        else:
            grad_input_padded = np.zeros_like(input_np)

        for oh in range(H_out):
            for ow in range(W_out):
                h_start = oh * sH
                w_start = ow * sW
                h_end = min(h_start + kH, grad_input_padded.shape[2])
                w_end = min(w_start + kW, grad_input_padded.shape[3])
                if divisor_override is not None:
                    count = divisor_override
                elif count_include_pad:
                    count = kH * kW
                else:
                    actual_h = min(h_end, H + pH) - max(h_start, pH)
                    actual_w = min(w_end, W + pW) - max(w_start, pW)
                    count = max(actual_h * actual_w, 1)
                grad_input_padded[:, :, h_start:h_end, w_start:w_end] += grad_np[:, :, oh:oh+1, ow:ow+1] / count

        if pH > 0 or pW > 0:
            grad_input_np = grad_input_padded[:, :, pH:pH+H, pW:pW+W]
        else:
            grad_input_np = grad_input_padded

        grad_input = _from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                                 input.dtype, input.device)
        return (grad_input,)


def _adaptive_avg_pool2d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        N, C, H, W = input_np.shape
        output_size = args[0]
        if isinstance(output_size, int):
            oH = oW = output_size
        else:
            oH, oW = output_size

        grad_input_np = np.zeros_like(input_np)
        for oh in range(oH):
            h_start = oh * H // oH
            h_end = (oh + 1) * H // oH
            for ow in range(oW):
                w_start = ow * W // oW
                w_end = (ow + 1) * W // oW
                count = (h_end - h_start) * (w_end - w_start)
                grad_input_np[:, :, h_start:h_end, w_start:w_end] += grad_np[:, :, oh:oh+1, ow:ow+1] / count

        grad_input = _from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                                 input.dtype, input.device)
        return (grad_input,)


# ---------------------------------------------------------------------------
# Phase 7: Utility ops backward
# ---------------------------------------------------------------------------

def _roll_backward(grad, _a, _saved_a, keyset, args, kwargs):
    shifts = args[0]
    dims = args[1] if len(args) > 1 else kwargs.get("dims", None)
    with _grad_context(keyset):
        if isinstance(shifts, int):
            neg_shifts = -shifts
        else:
            neg_shifts = tuple(-s for s in shifts)
        return (redispatch("roll", keyset, grad, neg_shifts, dims),)


def _tile_backward(grad, _a, saved_a, keyset, args, kwargs):
    dims = args[0]
    with _grad_context(keyset):
        input_shape = saved_a.shape
        ndim_input = len(input_shape)
        ndim_tile = len(dims) if isinstance(dims, (list, tuple)) else 1
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        ndim = max(ndim_input, ndim_tile)
        padded_shape = [1] * (ndim - ndim_input) + list(input_shape)
        padded_dims = [1] * (ndim - ndim_tile) + list(dims)
        reshape_dims = []
        for s, r in zip(padded_shape, padded_dims):
            reshape_dims.extend([r, s])
        grad_reshaped = redispatch("reshape", keyset, grad, tuple(reshape_dims))
        sum_dims = tuple(range(0, 2 * ndim, 2))
        for d in sum_dims:
            grad_reshaped = redispatch("sum", keyset, grad_reshaped, dim=d, keepdim=True)
        grad_out = redispatch("reshape", keyset, grad_reshaped, input_shape)
        return (grad_out,)


def _autograd_sort_like(name, backward_impl):
    """Autograd wrapper for sort/topk that returns (values, indices)."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        outs = redispatch(name, raw_keyset, a, *args, **kwargs)
        values, indices = outs
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, indices, backward_keyset, args, kwargs)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            values.grad_fn = node
            values.requires_grad = True
        return values, indices

    return wrapper


def _sort_backward(grad, a, indices, keyset, args, kwargs):
    dim = args[0] if args else kwargs.get("dim", -1)
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        redispatch("scatter_", keyset, grad_input, dim, indices, grad)
        return (grad_input,)


def _topk_backward(grad, a, indices, keyset, args, kwargs):
    dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        redispatch("scatter_", keyset, grad_input, dim, indices, grad)
        return (grad_input,)


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
    ("dropout", _autograd_dropout),
    ("cat", lambda: _autograd_multi_input("cat", _cat_backward, save_inputs=False)),
    ("stack", lambda: _autograd_multi_input("stack", _stack_backward, save_inputs=False)),
    ("embedding", lambda: _autograd_embedding("embedding", _embedding_backward)),
    ("where", lambda: _autograd_where("where", _where_backward)),
    ("split", lambda: _autograd_multi_output("split", _split_backward, save_input=False)),
    ("unbind", lambda: _autograd_multi_output("unbind", _unbind_backward, save_input=False)),
    # Phase 5: Transformer-critical ops
    ("masked_fill", lambda: _autograd_unary_args("masked_fill", _masked_fill_backward, save_input=False)),
    ("var", lambda: _autograd_unary_args("var", _var_backward)),
    ("std", lambda: _autograd_unary_args("std", _std_backward)),
    ("gather", lambda: _autograd_unary_args("gather", _gather_backward, save_input=False)),
    ("index_select", lambda: _autograd_unary_args("index_select", _index_select_backward, save_input=False)),
    ("repeat", lambda: _autograd_unary_args("repeat", _repeat_backward, save_input=False)),
    ("tril", lambda: _autograd_unary_args("tril", _tril_backward, save_input=False)),
    ("triu", lambda: _autograd_unary_args("triu", _triu_backward, save_input=False)),
    ("group_norm", lambda: _autograd_unary_args("group_norm", _group_norm_backward)),
    ("rms_norm", lambda: _autograd_unary_args("rms_norm", _rms_norm_backward)),
    ("flip", lambda: _autograd_unary_args("flip", _flip_backward, save_input=False)),
    ("cumsum", lambda: _autograd_unary_args("cumsum", _cumsum_backward, save_input=False)),
    ("pad", lambda: _autograd_unary_args("pad", _pad_backward, save_input=False)),
    ("prod", lambda: _autograd_unary_args("prod", _prod_backward)),
    ("norm", lambda: _autograd_unary_args("norm", _norm_backward)),
    # Phase 6: Conv/pool ops
    ("conv2d", lambda: _autograd_conv("conv2d")),
    ("conv1d", lambda: _autograd_conv("conv1d")),
    ("conv_transpose2d", lambda: _autograd_conv("conv_transpose2d")),
    ("conv_transpose1d", lambda: _autograd_conv("conv_transpose1d")),
    ("max_pool2d", lambda: _autograd_pool("max_pool2d", _max_pool2d_backward)),
    ("avg_pool2d", lambda: _autograd_pool("avg_pool2d", _avg_pool2d_backward)),
    ("adaptive_avg_pool2d", lambda: _autograd_pool("adaptive_avg_pool2d", _adaptive_avg_pool2d_backward)),
    # Phase 7: Utility ops
    ("roll", lambda: _autograd_unary_args("roll", _roll_backward, save_input=False)),
    ("tile", lambda: _autograd_unary_args("tile", _tile_backward, save_input=False)),
    ("sort", lambda: _autograd_sort_like("sort", _sort_backward)),
    ("topk", lambda: _autograd_sort_like("topk", _topk_backward)),
):
    if len(_entry) == 2:
        _name, _factory = _entry
        _register_autograd_op(_name, _factory)
    else:
        _name, _factory, _include_meta = _entry
        _register_autograd_op(_name, _factory, include_meta=_include_meta)

