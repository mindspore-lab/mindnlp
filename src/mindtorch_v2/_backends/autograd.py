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


def _conv_backward(name, grad, input, weight, bias, saved_input, saved_weight, saved_bias, keyset, args, kwargs):  # pylint: disable=too-many-branches,too-many-nested-blocks
    with _grad_context(keyset):
        # Use numpy for backward — simpler and always correct on CPU
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        weight_np = _to_numpy(saved_weight)

        is_1d = name in ("conv1d", "conv_transpose1d")
        is_3d = name in ("conv3d", "conv_transpose3d")
        is_transpose = name in ("conv_transpose2d", "conv_transpose1d", "conv_transpose3d")
        # pylint: disable=too-many-nested-blocks
        if is_3d:
            stride = kwargs.get("stride", args[0] if args else (1, 1, 1))
            padding = kwargs.get("padding", args[1] if len(args) > 1 else (0, 0, 0))
            dilation = kwargs.get("dilation", args[2] if len(args) > 2 else (1, 1, 1))
            groups = kwargs.get("groups", args[3] if len(args) > 3 else 1)
        else:
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

        if is_3d:
            # Handle 3D convolutions directly
            if isinstance(stride, int):
                stride = (stride, stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation, dilation)
            sD, sH, sW = stride
            pD, pH, pW = padding
            dD, dH, dW = dilation

            grad_input_np = None
            grad_weight_np = None
            grad_bias_np = None

            if is_transpose:
                N, C_in, D_in, H_in, W_in = input_np.shape
                C_in_w, C_out_g, kD, kH, kW = weight_np.shape
                C_out = C_out_g * groups

                if getattr(input, "requires_grad", False):
                    grad_input_np = np.zeros_like(input_np)
                    if pD > 0 or pH > 0 or pW > 0:
                        grad_padded = np.pad(grad_np, ((0,0),(0,0),(pD,pD),(pH,pH),(pW,pW)), mode='constant')
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
                                for id_ in range(D_in):
                                    for ih in range(H_in):
                                        for iw in range(W_in):
                                            val = 0.0
                                            for kd in range(kD):
                                                for kh in range(kH):
                                                    for kw in range(kW):
                                                        od = id_ * sD + kd * dD
                                                        oh = ih * sH + kh * dH
                                                        ow = iw * sW + kw * dW
                                                        if od < grad_padded.shape[2] and oh < grad_padded.shape[3] and ow < grad_padded.shape[4]:
                                                            val += grad_padded[:, co, od, oh, ow] * kernel[kd, kh, kw]
                                            grad_input_np[:, ci, id_, ih, iw] += val

                if getattr(weight, "requires_grad", False):
                    grad_weight_np = np.zeros_like(weight_np)
                    if pD > 0 or pH > 0 or pW > 0:
                        grad_padded = np.pad(grad_np, ((0,0),(0,0),(pD,pD),(pH,pH),(pW,pW)), mode='constant')
                    else:
                        grad_padded = grad_np
                    for g in range(groups):
                        c_in_per_g = C_in // groups
                        c_out_per_g = C_out_g
                        for ci_local in range(c_in_per_g):
                            ci = g * c_in_per_g + ci_local
                            for co_local in range(c_out_per_g):
                                co = g * c_out_per_g + co_local
                                for kd in range(kD):
                                    for kh in range(kH):
                                        for kw in range(kW):
                                            val = 0.0
                                            for id_ in range(D_in):
                                                for ih in range(H_in):
                                                    for iw in range(W_in):
                                                        od = id_ * sD + kd * dD
                                                        oh = ih * sH + kh * dH
                                                        ow = iw * sW + kw * dW
                                                        if od < grad_padded.shape[2] and oh < grad_padded.shape[3] and ow < grad_padded.shape[4]:
                                                            val += (input_np[:, ci, id_, ih, iw] * grad_padded[:, co, od, oh, ow]).sum()
                                            grad_weight_np[ci_local, co_local, kd, kh, kw] = val
            else:
                # Standard conv3d backward
                N, C_in, D_in, H_in, W_in = input_np.shape
                C_out, C_in_g, kD, kH, kW = weight_np.shape
                _, _, D_out, H_out, W_out = grad_np.shape

                if pD > 0 or pH > 0 or pW > 0:
                    input_padded = np.pad(input_np, ((0,0),(0,0),(pD,pD),(pH,pH),(pW,pW)), mode='constant')
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
                                for od in range(D_out):
                                    for oh in range(H_out):
                                        for ow in range(W_out):
                                            for kd in range(kD):
                                                for kh in range(kH):
                                                    for kw in range(kW):
                                                        id_ = od * sD + kd * dD
                                                        ih = oh * sH + kh * dH
                                                        iw = ow * sW + kw * dW
                                                        grad_input_padded[:, ci, id_, ih, iw] += grad_np[:, co, od, oh, ow] * kernel[kd, kh, kw]
                    if pD > 0 or pH > 0 or pW > 0:
                        grad_input_np = grad_input_padded[:, :, pD:pD+D_in, pH:pH+H_in, pW:pW+W_in]
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
                                for kd in range(kD):
                                    for kh in range(kH):
                                        for kw in range(kW):
                                            val = 0.0
                                            for od in range(D_out):
                                                for oh in range(H_out):
                                                    for ow in range(W_out):
                                                        id_ = od * sD + kd * dD
                                                        ih = oh * sH + kh * dH
                                                        iw = ow * sW + kw * dW
                                                        val += (input_padded[:, ci, id_, ih, iw] * grad_np[:, co, od, oh, ow]).sum()
                                            grad_weight_np[co, ci_local, kd, kh, kw] = val

            if bias is not None and getattr(bias, "requires_grad", False):
                grad_bias_np = grad_np.sum(axis=(0, 2, 3, 4))

            grad_input_t = None
            if grad_input_np is not None:
                grad_input_t = _from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)), input.dtype, input.device)
            grad_weight_t = None
            if grad_weight_np is not None:
                grad_weight_t = _from_numpy(np.ascontiguousarray(grad_weight_np.astype(weight_np.dtype)), weight.dtype, weight.device)
            grad_bias_t = None
            if grad_bias_np is not None:
                grad_bias_t = _from_numpy(np.ascontiguousarray(grad_bias_np.astype(input_np.dtype)), bias.dtype, bias.device)

            if bias is not None:
                return grad_input_t, grad_weight_t, grad_bias_t
            return grad_input_t, grad_weight_t

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


# ---------------------------------------------------------------------------
# Phase 8: Missing backward ops (Part A: existing ops, Part B: new ops)
# ---------------------------------------------------------------------------

# --- Part A: Missing backward for existing registered ops ---

def _sub_backward(grad, a, b, _saved_a, _saved_b, keyset):
    with _grad_context(keyset):
        neg_grad = redispatch("neg", keyset, grad)
    grad_a = reduce_grad(grad, a.shape) if getattr(a, "requires_grad", False) else None
    grad_b = reduce_grad(neg_grad, b.shape) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


def _mm_backward(grad, a, b, saved_a, saved_b, keyset):
    with _grad_context(keyset):
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            # grad_a = grad @ saved_b.T  (n,k) @ (m,k).T = (n,m)
            grad_a = redispatch("mm", keyset, grad, redispatch("transpose", keyset, saved_b, 0, 1))
        if getattr(b, "requires_grad", False):
            # grad_b = saved_a.T @ grad  (m,n) @ (n,k) = (m,k)
            grad_b = redispatch("mm", keyset, redispatch("transpose", keyset, saved_a, 0, 1), grad)
    return grad_a, grad_b


def _bmm_backward(grad, a, b, saved_a, saved_b, keyset):
    with _grad_context(keyset):
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            # grad_a = grad @ saved_b.transpose(-1,-2)
            grad_a = redispatch("matmul", keyset, grad, redispatch("transpose", keyset, saved_b, -1, -2))
        if getattr(b, "requires_grad", False):
            # grad_b = saved_a.transpose(-1,-2) @ grad
            grad_b = redispatch("matmul", keyset, redispatch("transpose", keyset, saved_a, -1, -2), grad)
    return grad_a, grad_b


def _lerp_backward(grad, a, b, weight, keyset):
    """Backward for lerp: out = a + weight * (b - a)."""
    with _grad_context(keyset):
        if isinstance(weight, (int, float)):
            w_t = _scalar_tensor_like(a, float(weight))
        else:
            w_t = weight
        one = _scalar_tensor_like(a, 1.0)
        one_minus_w = redispatch("add", keyset, one, redispatch("neg", keyset, w_t))
        grad_a = redispatch("mul", keyset, grad, one_minus_w) if getattr(a, "requires_grad", False) else None
        grad_b = redispatch("mul", keyset, grad, w_t) if getattr(b, "requires_grad", False) else None
    if grad_a is not None:
        grad_a = reduce_grad(grad_a, a.shape)
    if grad_b is not None:
        grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _autograd_lerp(name):
    """Autograd wrapper for lerp (3-input: a, b, weight)."""
    def wrapper(a, b, weight):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, weight)
        a_rg = getattr(a, "requires_grad", False)
        b_rg = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_rg or b_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _lerp_backward(grad, a, b, weight, backward_keyset)

            inputs = [t for t in (a, b) if hasattr(t, "requires_grad")]
            node = Node(_backward, tuple(inputs))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _addcmul_backward(grad, a, b, c, value, keyset):
    """Backward for addcmul: out = a + value * b * c."""
    with _grad_context(keyset):
        val_t = _scalar_tensor_like(a, float(value))
        grad_a = reduce_grad(grad, a.shape) if getattr(a, "requires_grad", False) else None
        grad_b = None
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, redispatch("mul", keyset, c, val_t))
            grad_b = reduce_grad(grad_b, b.shape)
        grad_c = None
        if getattr(c, "requires_grad", False):
            grad_c = redispatch("mul", keyset, grad, redispatch("mul", keyset, b, val_t))
            grad_c = reduce_grad(grad_c, c.shape)
    return grad_a, grad_b, grad_c


def _autograd_addcmul(name):
    """Autograd wrapper for addcmul (input, tensor1, tensor2, value=1)."""
    def wrapper(a, b, c, value=1):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, c, value)
        any_rg = (getattr(a, "requires_grad", False) or
                  getattr(b, "requires_grad", False) or
                  getattr(c, "requires_grad", False))
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _addcmul_backward(grad, a, b, c, value, backward_keyset)

            inputs = tuple(t for t in (a, b, c) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _addcdiv_backward(grad, a, b, c, value, keyset):
    """Backward for addcdiv: out = a + value * b / c."""
    with _grad_context(keyset):
        val_t = _scalar_tensor_like(a, float(value))
        grad_a = reduce_grad(grad, a.shape) if getattr(a, "requires_grad", False) else None
        grad_b = None
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("div", keyset, redispatch("mul", keyset, grad, val_t), c)
            grad_b = reduce_grad(grad_b, b.shape)
        grad_c = None
        if getattr(c, "requires_grad", False):
            c_sq = redispatch("mul", keyset, c, c)
            neg_grad_val_b = redispatch("neg", keyset,
                redispatch("mul", keyset, grad,
                    redispatch("mul", keyset, val_t, b)))
            grad_c = redispatch("div", keyset, neg_grad_val_b, c_sq)
            grad_c = reduce_grad(grad_c, c.shape)
    return grad_a, grad_b, grad_c


def _autograd_addcdiv(name):
    """Autograd wrapper for addcdiv (input, tensor1, tensor2, value=1)."""
    def wrapper(a, b, c, value=1):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, c, value)
        any_rg = (getattr(a, "requires_grad", False) or
                  getattr(b, "requires_grad", False) or
                  getattr(c, "requires_grad", False))
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _addcdiv_backward(grad, a, b, c, value, backward_keyset)

            inputs = tuple(t for t in (a, b, c) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _amax_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for amax: scatter gradient to positions where input == amax."""
    dim = kwargs.get("dim", None) if not args else args[0] if args else None
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        eps = _scalar_tensor_like(saved_a, 1e-12)
        if dim is None:
            max_val = redispatch("amax", keyset, saved_a)
            mask = redispatch("eq", keyset, saved_a, max_val)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad),)
        else:
            if not keepdim:
                grad_expanded = redispatch("unsqueeze", keyset, grad, dim)
            else:
                grad_expanded = grad
            max_val = redispatch("amax", keyset, saved_a, dim=dim, keepdim=True)
            mask = redispatch("eq", keyset, saved_a, max_val)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f, dim=dim, keepdim=True)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad_expanded),)


def _amin_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for amin: scatter gradient to positions where input == amin."""
    dim = kwargs.get("dim", None) if not args else args[0] if args else None
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        eps = _scalar_tensor_like(saved_a, 1e-12)
        if dim is None:
            min_val = redispatch("amin", keyset, saved_a)
            mask = redispatch("eq", keyset, saved_a, min_val)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad),)
        else:
            if not keepdim:
                grad_expanded = redispatch("unsqueeze", keyset, grad, dim)
            else:
                grad_expanded = grad
            min_val = redispatch("amin", keyset, saved_a, dim=dim, keepdim=True)
            mask = redispatch("eq", keyset, saved_a, min_val)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f, dim=dim, keepdim=True)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad_expanded),)


# --- Part B: Backward for new ops ---

def _log1p_backward(grad, _a, saved_a, keyset):
    """Backward for log1p: out = log(1 + x), grad_input = grad / (1 + x)."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        denom = redispatch("add", keyset, one, saved_a)
        return (redispatch("div", keyset, grad, denom),)


def _expm1_backward(grad, _a, saved_a, keyset):
    """Backward for expm1: out = exp(x) - 1, grad_input = grad * exp(x)."""
    with _grad_context(keyset):
        exp_x = redispatch("exp", keyset, saved_a)
        return (redispatch("mul", keyset, grad, exp_x),)


def _reciprocal_backward(grad, _a, saved_a, keyset):
    """Backward for reciprocal: out = 1/x, grad_input = -grad / x^2."""
    with _grad_context(keyset):
        sq = redispatch("mul", keyset, saved_a, saved_a)
        neg_grad = redispatch("neg", keyset, grad)
        return (redispatch("div", keyset, neg_grad, sq),)


def _maximum_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for element-wise maximum."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        # mask_a is 1 where a >= b, 0 otherwise
        ge_mask = redispatch("ge", keyset, saved_a, saved_b)
        mask_a = redispatch("where", keyset, ge_mask, ones, redispatch("mul", keyset, ones, zero))
        # mask_b = 1 - mask_a (where b > a)
        mask_b = redispatch("add", keyset, ones, redispatch("neg", keyset, mask_a))
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, mask_a)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, mask_b)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _minimum_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for element-wise minimum."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        # mask_a is 1 where a <= b, 0 otherwise
        le_mask = redispatch("le", keyset, saved_a, saved_b)
        mask_a = redispatch("where", keyset, le_mask, ones, redispatch("mul", keyset, ones, zero))
        # mask_b = 1 - mask_a
        mask_b = redispatch("add", keyset, ones, redispatch("neg", keyset, mask_a))
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, mask_a)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, mask_b)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _dot_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for dot (1D dot product, scalar output)."""
    with _grad_context(keyset):
        grad_a = redispatch("mul", keyset, grad, saved_b) if getattr(a, "requires_grad", False) else None
        grad_b = redispatch("mul", keyset, grad, saved_a) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


def _outer_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for outer product: out[i,j] = a[i]*b[j]."""
    with _grad_context(keyset):
        # grad shape: (n, m)
        # grad_a[i] = sum_j grad[i,j] * b[j]  → (n,) = grad @ b
        grad_a = redispatch("matmul", keyset, grad, saved_b) if getattr(a, "requires_grad", False) else None
        # grad_b[j] = sum_i grad[i,j] * a[i]  → (m,) = grad.T @ a
        grad_b = redispatch("matmul", keyset, redispatch("transpose", keyset, grad, 0, 1), saved_a) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


def _mv_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for mv (matrix-vector product): out = a @ b."""
    with _grad_context(keyset):
        # grad shape: (n,)
        # grad_a[i,j] = grad[i] * b[j]  → outer product  (n,m)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("outer", keyset, grad, saved_b)
        if getattr(b, "requires_grad", False):
            # grad_b = a.T @ grad  → (m,n) @ (n,) = (m,)
            grad_b = redispatch("matmul", keyset, redispatch("transpose", keyset, saved_a, 0, 1), grad)
    return grad_a, grad_b


def _flatten_backward(grad, a, _saved_a, keyset, args, kwargs):
    """Backward for flatten: reshape gradient back to original shape."""
    return (redispatch("reshape", keyset, grad, a.shape),)


def _unflatten_backward(grad, a, _saved_a, keyset, args, kwargs):
    """Backward for unflatten: reshape gradient back to original shape."""
    return (redispatch("reshape", keyset, grad, a.shape),)


def _movedim_backward(grad, _a, _saved_a, keyset, args, kwargs):
    """Backward for movedim: apply the inverse permutation."""
    source = args[0]
    destination = args[1]
    with _grad_context(keyset):
        # Inverse: move destination back to source
        return (redispatch("movedim", keyset, grad, destination, source),)


def _diagonal_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for diagonal: scatter grad back along the diagonal using numpy."""
    offset = args[0] if args else kwargs.get("offset", 0)
    dim1 = args[1] if len(args) > 1 else kwargs.get("dim1", 0)
    dim2 = args[2] if len(args) > 2 else kwargs.get("dim2", 1)
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        shape = saved_a.shape
        ndim = len(shape)
        d1 = dim1 if dim1 >= 0 else dim1 + ndim
        d2 = dim2 if dim2 >= 0 else dim2 + ndim

        # Move d1 and d2 to the last two positions
        perm = [i for i in range(ndim) if i not in (d1, d2)] + [d1, d2]
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i

        out_np = np.zeros([shape[i] for i in range(ndim)], dtype=grad_np.dtype)
        arr_moved = np.transpose(out_np, perm)
        # grad has batch dims + diagonal dim at the end
        # The number of diagonal elements
        n_d1 = shape[d1]
        n_d2 = shape[d2]
        if offset >= 0:
            diag_len = min(n_d1, n_d2 - offset)
        else:
            diag_len = min(n_d1 + offset, n_d2)
        diag_len = max(diag_len, 0)

        for k in range(diag_len):
            if offset >= 0:
                i1, i2 = k, k + offset
            else:
                i1, i2 = k - offset, k
            arr_moved[..., i1, i2] = grad_np[..., k]

        out_np = np.transpose(arr_moved, inv_perm)
        return (_from_numpy(np.ascontiguousarray(out_np), saved_a.dtype, saved_a.device),)


def _hardswish_backward(grad, _a, saved_a, keyset):
    """Backward for hardswish: d/dx[x * hardsigmoid(x)]."""
    with _grad_context(keyset):
        three = _scalar_tensor_like(saved_a, 3.0)
        sixth = _scalar_tensor_like(saved_a, 1.0 / 6.0)
        two = _scalar_tensor_like(saved_a, 2.0)
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        # Masks using relu+sign trick (no gt/lt ops needed)
        # gt_neg3: 1 where x > -3, else 0
        gt_neg3 = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, saved_a, three)))
        # lt_3: 1 where x < 3, else 0
        lt_3 = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, three, redispatch("neg", keyset, saved_a))))
        # inner_mask: 1 where -3 < x < 3
        inner_mask = redispatch("mul", keyset, gt_neg3, lt_3)
        # ge_3_mask: 1 where x >= 3
        ge_3_mask = redispatch("where", keyset,
            redispatch("ge", keyset, saved_a, three),
            ones, redispatch("mul", keyset, ones, zero))
        # Hardswish grad:
        # x < -3:        0
        # -3 <= x < 3:   (2x + 3) / 6
        # x >= 3:        1
        two_x_plus_3 = redispatch("add", keyset, redispatch("mul", keyset, two, saved_a), three)
        inner_grad = redispatch("mul", keyset, two_x_plus_3, sixth)
        dout = redispatch("add", keyset,
            redispatch("mul", keyset, inner_grad, inner_mask),
            ge_3_mask)
        return (redispatch("mul", keyset, grad, dout),)


def _hardsigmoid_backward(grad, _a, saved_a, keyset):
    """Backward for hardsigmoid: max(0, min(1, (x+3)/6))."""
    with _grad_context(keyset):
        three = _scalar_tensor_like(saved_a, 3.0)
        sixth = _scalar_tensor_like(saved_a, 1.0 / 6.0)
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        # 1/6 where -3 < x < 3, else 0
        gt_neg3 = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, saved_a, three)))
        lt_3 = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, three, redispatch("neg", keyset, saved_a))))
        inner_mask = redispatch("mul", keyset, gt_neg3, lt_3)
        mask_f = redispatch("where", keyset,
            redispatch("gt", keyset, inner_mask, _scalar_tensor_like(saved_a, 0.0)),
            ones, redispatch("mul", keyset, ones, zero))
        return (redispatch("mul", keyset, grad, redispatch("mul", keyset, mask_f, sixth)),)


def _softsign_backward(grad, _a, saved_a, keyset):
    """Backward for softsign: out = x / (1 + |x|), grad = 1 / (1 + |x|)^2."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        denom = redispatch("add", keyset, one, redispatch("abs", keyset, saved_a))
        denom_sq = redispatch("mul", keyset, denom, denom)
        return (redispatch("div", keyset, grad, denom_sq),)


def _selu_backward(grad, _a, saved_a, keyset):
    """Backward for selu: scale * (x if x > 0 else alpha*(exp(x)-1))."""
    SCALE = 1.0507009873554804934193349852946
    ALPHA = 1.6732631921893986195596513061800
    with _grad_context(keyset):
        scale_t = _scalar_tensor_like(saved_a, SCALE)
        alpha_scale_t = _scalar_tensor_like(saved_a, SCALE * ALPHA)
        small_eps = _scalar_tensor_like(saved_a, 1e-7)
        ones = saved_a._ones_like()
        # pos_mask: 1 where x > 0
        pos_mask = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, saved_a, small_eps)))
        # neg_mask: 1 where x <= 0
        neg_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
        exp_x = redispatch("exp", keyset, saved_a)
        neg_deriv = redispatch("mul", keyset, alpha_scale_t, exp_x)
        deriv = redispatch("add", keyset,
            redispatch("mul", keyset, pos_mask, scale_t),
            redispatch("mul", keyset, neg_mask, neg_deriv))
        return (redispatch("mul", keyset, grad, deriv),)


def _celu_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for celu: max(0,x) + min(0, alpha*(exp(x/alpha)-1))."""
    alpha = args[0] if args else kwargs.get("alpha", 1.0)
    with _grad_context(keyset):
        alpha_t = _scalar_tensor_like(saved_a, float(alpha))
        small_eps = _scalar_tensor_like(saved_a, 1e-7)
        ones = saved_a._ones_like()
        # pos_mask: 1 where x > 0
        pos_mask = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, saved_a, small_eps)))
        # neg_mask: 1 where x <= 0
        neg_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
        exp_x_alpha = redispatch("exp", keyset, redispatch("div", keyset, saved_a, alpha_t))
        deriv = redispatch("add", keyset,
            pos_mask,
            redispatch("mul", keyset, neg_mask, exp_x_alpha))
        return (redispatch("mul", keyset, grad, deriv),)


def _threshold_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for threshold: out = x if x > threshold else value."""
    threshold = args[0] if args else kwargs.get("threshold", 0.0)
    with _grad_context(keyset):
        threshold_t = _scalar_tensor_like(saved_a, float(threshold))
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        # mask: 1 where x > threshold, else 0
        gt_mask = redispatch("sign", keyset, redispatch("relu", keyset,
            redispatch("add", keyset, saved_a,
                redispatch("neg", keyset, threshold_t))))
        # Strictly greater: use relu(x - threshold) sign trick
        # Note: sign(relu(x - threshold)) = 1 when x > threshold (for float)
        mask = redispatch("where", keyset,
            redispatch("gt", keyset, saved_a, threshold_t),
            ones, redispatch("mul", keyset, ones, zero))
        return (redispatch("mul", keyset, grad, mask),)


def _instance_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for instance_norm: normalize over spatial dims per (N, C)."""
    weight = args[0] if args else kwargs.get("weight", None)
    bias = args[1] if len(args) > 1 else kwargs.get("bias", None)
    eps = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)

    with _grad_context(keyset):
        shape = saved_a.shape
        ndim = len(shape)
        # Spatial axes: everything except batch (0) and channel (1)
        spatial_axes = tuple(range(2, ndim))

        # Compute per-(N,C) mean and variance over spatial dims
        mean = redispatch("mean", keyset, saved_a, dim=spatial_axes, keepdim=True)
        diff = redispatch("add", keyset, saved_a, redispatch("neg", keyset, mean))
        var = redispatch("mean", keyset, redispatch("mul", keyset, diff, diff),
                         dim=spatial_axes, keepdim=True)
        eps_t = _scalar_tensor_like(saved_a, float(eps))
        inv_std = redispatch("rsqrt", keyset, redispatch("add", keyset, var, eps_t))
        x_hat = redispatch("mul", keyset, diff, inv_std)

        if weight is not None:
            # weight shape: (C,) → reshape to (1, C, 1, ...) for broadcasting
            w_shape = [1, shape[1]] + [1] * (ndim - 2)
            dl_dxhat = redispatch("mul", keyset, grad,
                redispatch("reshape", keyset, weight, tuple(w_shape)))
        else:
            dl_dxhat = grad

        # n = number of spatial elements
        n = 1
        for d in spatial_axes:
            n *= shape[d]
        n_t = _scalar_tensor_like(saved_a, float(n))

        mean_dl_dxhat = redispatch("div", keyset,
            redispatch("sum", keyset, dl_dxhat, dim=spatial_axes, keepdim=True), n_t)
        mean_dl_dxhat_xhat = redispatch("div", keyset,
            redispatch("sum", keyset,
                redispatch("mul", keyset, dl_dxhat, x_hat),
                dim=spatial_axes, keepdim=True), n_t)

        grad_input = redispatch("mul", keyset, inv_std,
            redispatch("add", keyset,
                redispatch("add", keyset, dl_dxhat,
                    redispatch("neg", keyset, mean_dl_dxhat)),
                redispatch("neg", keyset,
                    redispatch("mul", keyset, x_hat, mean_dl_dxhat_xhat))))
        return (grad_input,)


def _normalize_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for F.normalize: out = x / max(||x||_p, eps)."""
    p = args[0] if args else kwargs.get("p", 2)
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 1)
    eps = args[2] if len(args) > 2 else kwargs.get("eps", 1e-12)
    with _grad_context(keyset):
        norm = redispatch("norm", keyset, saved_a, p, dim=dim, keepdim=True)
        eps_t = _scalar_tensor_like(saved_a, float(eps))
        safe_norm = redispatch("add", keyset, norm, eps_t)
        # n = x / safe_norm  (the normalized vector)
        n = redispatch("div", keyset, saved_a, safe_norm)
        # grad_input = (grad - (grad·n).sum(dim, keepdim=True) * n) / safe_norm
        dot = redispatch("sum", keyset,
            redispatch("mul", keyset, grad, n), dim=dim, keepdim=True)
        grad_proj = redispatch("mul", keyset, dot, n)
        return (redispatch("div", keyset,
            redispatch("add", keyset, grad, redispatch("neg", keyset, grad_proj)),
            safe_norm),)


# ---------------------------------------------------------------------------
# Round 2: Remaining backward ops
# ---------------------------------------------------------------------------

# --- Tier 1: Training-Critical ---

def _clamp_min_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for clamp_min: grad * (x >= min)."""
    min_val = args[0] if args else kwargs.get("min_val", kwargs.get("min", 0))
    with _grad_context(keyset):
        min_t = _scalar_tensor_like(saved_a, float(min_val))
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        mask = redispatch("where", keyset,
            redispatch("ge", keyset, saved_a, min_t),
            ones, redispatch("mul", keyset, ones, zero))
        return (redispatch("mul", keyset, grad, mask),)


def _clamp_max_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for clamp_max: grad * (x <= max)."""
    max_val = args[0] if args else kwargs.get("max_val", kwargs.get("max", 0))
    with _grad_context(keyset):
        max_t = _scalar_tensor_like(saved_a, float(max_val))
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        mask = redispatch("where", keyset,
            redispatch("le", keyset, saved_a, max_t),
            ones, redispatch("mul", keyset, ones, zero))
        return (redispatch("mul", keyset, grad, mask),)


def _min_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for elementwise min(a, b) — same logic as minimum."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        le_mask = redispatch("le", keyset, saved_a, saved_b)
        mask_a = redispatch("where", keyset, le_mask, ones, redispatch("mul", keyset, ones, zero))
        mask_b = redispatch("add", keyset, ones, redispatch("neg", keyset, mask_a))
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, mask_a)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, mask_b)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _max_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for elementwise max(a, b) — same logic as maximum."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        ge_mask = redispatch("ge", keyset, saved_a, saved_b)
        mask_a = redispatch("where", keyset, ge_mask, ones, redispatch("mul", keyset, ones, zero))
        mask_b = redispatch("add", keyset, ones, redispatch("neg", keyset, mask_a))
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, mask_a)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, mask_b)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _cumprod_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for cumprod: uses reverse cumsum of grad * cumprod / x."""
    dim = args[0] if args else kwargs.get("dim", 0)
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        x_np = _to_numpy(saved_a).astype(np.float64)
        grad_np = _to_numpy(grad).astype(np.float64)
        y_np = np.cumprod(x_np, axis=dim)
        gy = grad_np * y_np
        rev_cumsum = np.flip(np.cumsum(np.flip(gy, axis=dim), axis=dim), axis=dim)
        safe_x = np.where(x_np != 0, x_np, 1.0)
        result = np.where(x_np != 0, rev_cumsum / safe_x, 0.0)
        return (_from_numpy(result.astype(_to_numpy(saved_a).dtype), saved_a.dtype, saved_a.device),)


def _repeat_interleave_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for repeat_interleave: sum over interleaved groups."""
    repeats = args[0] if args else kwargs.get("repeats")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        if dim is None:
            # Input was flattened
            n = saved_a.numel()
            if isinstance(repeats, int):
                result = grad_np.reshape(n, repeats).sum(axis=1)
            else:
                reps = _to_numpy(repeats) if hasattr(repeats, 'shape') else np.array(repeats)
                result = np.zeros(n, dtype=grad_np.dtype)
                idx = 0
                for i in range(n):
                    r = int(reps[i])
                    result[i] = grad_np[idx:idx + r].sum()
                    idx += r
            return (_from_numpy(result.reshape(saved_a.shape), saved_a.dtype, saved_a.device),)
        else:
            d = dim if dim >= 0 else dim + len(saved_a.shape)
            n = saved_a.shape[d]
            if isinstance(repeats, int):
                new_shape = list(grad_np.shape)
                new_shape[d] = n
                new_shape.insert(d + 1, repeats)
                result = grad_np.reshape(new_shape).sum(axis=d + 1)
            else:
                reps = _to_numpy(repeats) if hasattr(repeats, 'shape') else np.array(repeats)
                perm = [d] + [i for i in range(len(saved_a.shape)) if i != d]
                inv_perm = [0] * len(perm)
                for i, p in enumerate(perm):
                    inv_perm[p] = i
                grad_moved = np.transpose(grad_np, perm)
                other_shape = grad_moved.shape[1:]
                result_moved = np.zeros((n,) + other_shape, dtype=grad_np.dtype)
                idx = 0
                for i in range(n):
                    r = int(reps[i])
                    result_moved[i] = grad_moved[idx:idx + r].sum(axis=0)
                    idx += r
                result = np.transpose(result_moved, inv_perm)
            return (_from_numpy(result, saved_a.dtype, saved_a.device),)


def _autograd_scatter(name):
    """Custom autograd wrapper for scatter(a, dim, index, src)."""
    def wrapper(a, dim, index, src):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim, index, src)
        a_rg = getattr(a, "requires_grad", False)
        src_rg = getattr(src, "requires_grad", False)
        if GradMode.enabled and (a_rg or src_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _scatter_backward(grad, a, src, dim, index, backward_keyset)

            inputs = tuple(t for t in (a, src) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _scatter_backward(grad, a, src, dim, index, keyset):
    """Backward for scatter: grad_a is grad with scattered positions zeroed, grad_src is gathered."""
    with _grad_context(keyset):
        grad_a = None
        grad_src = None
        if getattr(a, "requires_grad", False):
            # Zero out the scattered positions
            import numpy as np
            from .cpu.ops import _to_numpy, _from_numpy
            grad_np = _to_numpy(grad).copy()
            idx_np = _to_numpy(index).astype(np.int64)
            d = dim if dim >= 0 else dim + grad_np.ndim
            it = np.nditer(idx_np, flags=['multi_index'])
            while not it.finished:
                mi = list(it.multi_index)
                mi[d] = int(it[0])
                grad_np[tuple(mi)] = 0.0
                it.iternext()
            grad_a = _from_numpy(grad_np, a.dtype, a.device)
        if getattr(src, "requires_grad", False):
            grad_src = redispatch("gather", keyset, grad, dim, index)
    return grad_a, grad_src


def _floor_divide_backward(grad, a, b, _saved_a, _saved_b, keyset):
    """Backward for floor_divide: not differentiable, grad = 0."""
    grad_a = None
    grad_b = None
    if getattr(a, "requires_grad", False):
        grad_a = redispatch("mul", keyset, grad, _scalar_tensor_like(a, 0.0))
        grad_a = reduce_grad(grad_a, a.shape)
    if getattr(b, "requires_grad", False):
        grad_b = redispatch("mul", keyset, grad, _scalar_tensor_like(b, 0.0))
        grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


# --- Tier 2: Trig / Math ---

def _tan_backward(grad, _a, saved_a, keyset):
    """Backward for tan: grad / cos(x)^2."""
    with _grad_context(keyset):
        cos_x = redispatch("cos", keyset, saved_a)
        cos_sq = redispatch("mul", keyset, cos_x, cos_x)
        return (redispatch("div", keyset, grad, cos_sq),)


def _asin_backward(grad, _a, saved_a, keyset):
    """Backward for asin: grad / sqrt(1 - x^2)."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        denom = redispatch("sqrt", keyset, redispatch("add", keyset, one, redispatch("neg", keyset, x_sq)))
        return (redispatch("div", keyset, grad, denom),)


def _acos_backward(grad, _a, saved_a, keyset):
    """Backward for acos: -grad / sqrt(1 - x^2)."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        denom = redispatch("sqrt", keyset, redispatch("add", keyset, one, redispatch("neg", keyset, x_sq)))
        neg_grad = redispatch("neg", keyset, grad)
        return (redispatch("div", keyset, neg_grad, denom),)


def _atan_backward(grad, _a, saved_a, keyset):
    """Backward for atan: grad / (1 + x^2)."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        denom = redispatch("add", keyset, one, x_sq)
        return (redispatch("div", keyset, grad, denom),)


def _atan2_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for atan2(a, b): grad_a = grad * b / (a^2 + b^2), grad_b = -grad * a / (a^2 + b^2)."""
    with _grad_context(keyset):
        a_sq = redispatch("mul", keyset, saved_a, saved_a)
        b_sq = redispatch("mul", keyset, saved_b, saved_b)
        denom = redispatch("add", keyset, a_sq, b_sq)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("div", keyset, redispatch("mul", keyset, grad, saved_b), denom)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            neg_a = redispatch("neg", keyset, saved_a)
            grad_b = redispatch("div", keyset, redispatch("mul", keyset, grad, neg_a), denom)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _sinh_backward(grad, _a, saved_a, keyset):
    """Backward for sinh: grad * cosh(x)."""
    with _grad_context(keyset):
        cosh_x = redispatch("cosh", keyset, saved_a)
        return (redispatch("mul", keyset, grad, cosh_x),)


def _cosh_backward(grad, _a, saved_a, keyset):
    """Backward for cosh: grad * sinh(x)."""
    with _grad_context(keyset):
        sinh_x = redispatch("sinh", keyset, saved_a)
        return (redispatch("mul", keyset, grad, sinh_x),)


def _asinh_backward(grad, _a, saved_a, keyset):
    """Backward for asinh: grad / sqrt(x^2 + 1)."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        denom = redispatch("sqrt", keyset, redispatch("add", keyset, x_sq, one))
        return (redispatch("div", keyset, grad, denom),)


def _acosh_backward(grad, _a, saved_a, keyset):
    """Backward for acosh: grad / sqrt(x^2 - 1)."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        denom = redispatch("sqrt", keyset, redispatch("add", keyset, x_sq, redispatch("neg", keyset, one)))
        return (redispatch("div", keyset, grad, denom),)


def _atanh_backward(grad, _a, saved_a, keyset):
    """Backward for atanh: grad / (1 - x^2)."""
    with _grad_context(keyset):
        one = _scalar_tensor_like(saved_a, 1.0)
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        denom = redispatch("add", keyset, one, redispatch("neg", keyset, x_sq))
        return (redispatch("div", keyset, grad, denom),)


def _log10_backward(grad, _a, saved_a, keyset):
    """Backward for log10: grad / (x * ln(10))."""
    import math
    with _grad_context(keyset):
        ln10 = _scalar_tensor_like(saved_a, math.log(10.0))
        denom = redispatch("mul", keyset, saved_a, ln10)
        return (redispatch("div", keyset, grad, denom),)


def _erfc_backward(grad, _a, saved_a, keyset):
    """Backward for erfc: -grad * 2/sqrt(pi) * exp(-x^2)."""
    import math
    with _grad_context(keyset):
        coeff = _scalar_tensor_like(saved_a, -2.0 / math.sqrt(math.pi))
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        neg_x_sq = redispatch("neg", keyset, x_sq)
        exp_val = redispatch("exp", keyset, neg_x_sq)
        factor = redispatch("mul", keyset, coeff, exp_val)
        return (redispatch("mul", keyset, grad, factor),)


def _logaddexp_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for logaddexp: grad_a = grad * exp(a) / (exp(a) + exp(b)), grad_b = grad * exp(b) / (exp(a) + exp(b))."""
    with _grad_context(keyset):
        # Use softmax-like trick for numerical stability
        # max_ab = max(a, b)
        # grad_a = grad * exp(a - max_ab) / (exp(a - max_ab) + exp(b - max_ab))
        max_ab = redispatch("maximum", keyset, saved_a, saved_b)
        a_shifted = redispatch("add", keyset, saved_a, redispatch("neg", keyset, max_ab))
        b_shifted = redispatch("add", keyset, saved_b, redispatch("neg", keyset, max_ab))
        exp_a = redispatch("exp", keyset, a_shifted)
        exp_b = redispatch("exp", keyset, b_shifted)
        denom = redispatch("add", keyset, exp_a, exp_b)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, redispatch("div", keyset, exp_a, denom))
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, redispatch("div", keyset, exp_b, denom))
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _logaddexp2_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for logaddexp2: same as logaddexp but base-2."""
    import math
    with _grad_context(keyset):
        ln2 = _scalar_tensor_like(saved_a, math.log(2.0))
        # Convert to natural log scale: logaddexp2(a, b) = log2(2^a + 2^b)
        # d/da = 2^a / (2^a + 2^b) = 1 / (1 + 2^(b-a))
        max_ab = redispatch("maximum", keyset, saved_a, saved_b)
        a_shifted = redispatch("add", keyset, saved_a, redispatch("neg", keyset, max_ab))
        b_shifted = redispatch("add", keyset, saved_b, redispatch("neg", keyset, max_ab))
        # 2^x = exp(x * ln2)
        exp_a = redispatch("exp", keyset, redispatch("mul", keyset, a_shifted, ln2))
        exp_b = redispatch("exp", keyset, redispatch("mul", keyset, b_shifted, ln2))
        denom = redispatch("add", keyset, exp_a, exp_b)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, redispatch("div", keyset, exp_a, denom))
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, redispatch("div", keyset, exp_b, denom))
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


# --- Tier 3: Less Common ---

def _fmin_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for fmin: like min but NaN-aware (NaN inputs get 0 grad)."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        # fmin: returns non-NaN if either is non-NaN
        a_nan = redispatch("isnan", keyset, saved_a)
        b_nan = redispatch("isnan", keyset, saved_b)
        le_mask = redispatch("le", keyset, saved_a, saved_b)
        # a gets grad when a <= b and a is not NaN, or b is NaN
        mask_a = redispatch("where", keyset,
            redispatch("logical_or", keyset, redispatch("logical_and", keyset, le_mask, redispatch("logical_not", keyset, a_nan)), b_nan),
            ones, redispatch("mul", keyset, ones, zero))
        mask_b = redispatch("add", keyset, ones, redispatch("neg", keyset, mask_a))
        # Zero grad for NaN inputs
        mask_a = redispatch("where", keyset, a_nan, redispatch("mul", keyset, ones, zero), mask_a)
        mask_b = redispatch("where", keyset, b_nan, redispatch("mul", keyset, ones, zero), mask_b)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, mask_a)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, mask_b)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _fmax_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for fmax: like max but NaN-aware (NaN inputs get 0 grad)."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        a_nan = redispatch("isnan", keyset, saved_a)
        b_nan = redispatch("isnan", keyset, saved_b)
        ge_mask = redispatch("ge", keyset, saved_a, saved_b)
        mask_a = redispatch("where", keyset,
            redispatch("logical_or", keyset, redispatch("logical_and", keyset, ge_mask, redispatch("logical_not", keyset, a_nan)), b_nan),
            ones, redispatch("mul", keyset, ones, zero))
        mask_b = redispatch("add", keyset, ones, redispatch("neg", keyset, mask_a))
        mask_a = redispatch("where", keyset, a_nan, redispatch("mul", keyset, ones, zero), mask_a)
        mask_b = redispatch("where", keyset, b_nan, redispatch("mul", keyset, ones, zero), mask_b)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, mask_a)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, mask_b)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _fmod_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for fmod: grad_a = grad, grad_b = -grad * trunc(a/b)."""
    with _grad_context(keyset):
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = reduce_grad(grad, a.shape)
        if getattr(b, "requires_grad", False):
            ratio = redispatch("trunc", keyset, redispatch("div", keyset, saved_a, saved_b))
            grad_b = redispatch("neg", keyset, redispatch("mul", keyset, grad, ratio))
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _hypot_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for hypot: grad_a = grad * a / hypot(a,b), grad_b = grad * b / hypot(a,b)."""
    with _grad_context(keyset):
        h = redispatch("sqrt", keyset,
            redispatch("add", keyset,
                redispatch("mul", keyset, saved_a, saved_a),
                redispatch("mul", keyset, saved_b, saved_b)))
        eps = _scalar_tensor_like(saved_a, 1e-12)
        safe_h = redispatch("add", keyset, h, eps)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, redispatch("div", keyset, saved_a, safe_h))
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, redispatch("div", keyset, saved_b, safe_h))
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _remainder_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for remainder: grad_a = grad, grad_b = -grad * floor(a/b)."""
    with _grad_context(keyset):
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = reduce_grad(grad, a.shape)
        if getattr(b, "requires_grad", False):
            ratio = redispatch("floor", keyset, redispatch("div", keyset, saved_a, saved_b))
            grad_b = redispatch("neg", keyset, redispatch("mul", keyset, grad, ratio))
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _rot90_backward(grad, _a, _saved_a, keyset, args, kwargs):
    """Backward for rot90: apply inverse rotation rot90(grad, -k, dims)."""
    k = args[0] if args else kwargs.get("k", 1)
    dims = args[1] if len(args) > 1 else kwargs.get("dims", (0, 1))
    with _grad_context(keyset):
        return (redispatch("rot90", keyset, grad, -k, dims),)


def _take_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for take: scatter gradient back to original positions."""
    index = args[0] if args else kwargs.get("index")
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        idx_np = _to_numpy(index).astype(np.int64).flatten()
        result = np.zeros(saved_a.numel(), dtype=grad_np.dtype)
        np.add.at(result, idx_np, grad_np.flatten())
        return (_from_numpy(result.reshape(saved_a.shape), saved_a.dtype, saved_a.device),)


def _take_along_dim_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for take_along_dim: scatter-add gradient back along dim."""
    indices = args[0] if args else kwargs.get("indices")
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        idx_np = _to_numpy(indices).astype(np.int64)
        result = np.zeros(saved_a.shape, dtype=grad_np.dtype)
        d = dim if dim >= 0 else dim + len(saved_a.shape)
        # Use scatter-add to handle repeated indices correctly
        it = np.nditer(idx_np, flags=['multi_index'])
        while not it.finished:
            mi = list(it.multi_index)
            src_idx = tuple(it.multi_index)
            mi[d] = int(it[0])
            result[tuple(mi)] += grad_np[src_idx]
            it.iternext()
        return (_from_numpy(result, saved_a.dtype, saved_a.device),)


def _autograd_cummax(name):
    """Custom autograd wrapper for cummax which returns (values, indices) namedtuple."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        result = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and a.requires_grad:
            values = result.values if hasattr(result, 'values') else result[0]
            indices = result.indices if hasattr(result, 'indices') else result[1]
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _cummax_backward(grad, a, indices, backward_keyset, args, kwargs)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            values.grad_fn = node
            values.requires_grad = True
            # Return same structure with updated values
            if hasattr(result, '_replace'):
                result = result._replace(values=values)
            else:
                result = (values, indices)
        return result

    return wrapper


def _cummax_backward(grad, a, indices, keyset, args, kwargs):
    """Backward for cummax: scatter-add grad to positions indicated by cummax indices."""
    dim = args[0] if args else kwargs.get("dim", 0)
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        idx_np = _to_numpy(indices).astype(np.int64)
        result = np.zeros(a.shape, dtype=grad_np.dtype)
        d = dim if dim >= 0 else dim + len(a.shape)
        # Must use scatter-add (not put) to accumulate repeated indices
        it = np.nditer(idx_np, flags=['multi_index'])
        while not it.finished:
            mi = list(it.multi_index)
            src_idx = tuple(it.multi_index)
            mi[d] = int(it[0])
            result[tuple(mi)] += grad_np[src_idx]
            it.iternext()
        return (_from_numpy(result, a.dtype, a.device),)


# ---------------------------------------------------------------------------
# P0 Gap Fix — Round 3: addmm, upsample, pool 1d/3d, conv3d, indexing,
#                        math, grid_sample, affine_grid, masked_fill_
# ---------------------------------------------------------------------------

# ---- Task 1: addmm backward ----

def _addmm_backward(grad, input, mat1, mat2, saved_input, saved_mat1, saved_mat2, keyset, args, kwargs):
    beta = kwargs.get("beta", args[0] if args else 1)
    alpha = kwargs.get("alpha", args[1] if len(args) > 1 else 1)
    with _grad_context(keyset):
        grad_input = None
        grad_mat1 = None
        grad_mat2 = None
        if getattr(input, "requires_grad", False):
            grad_input = redispatch("mul", keyset, grad, _scalar_tensor_like(grad, float(beta)))
            grad_input = reduce_grad(grad_input, input.shape)
        if getattr(mat1, "requires_grad", False):
            grad_mat1 = redispatch("mul", keyset,
                redispatch("matmul", keyset, grad, redispatch("transpose", keyset, saved_mat2, -1, -2)),
                _scalar_tensor_like(grad, float(alpha)))
        if getattr(mat2, "requires_grad", False):
            grad_mat2 = redispatch("mul", keyset,
                redispatch("matmul", keyset, redispatch("transpose", keyset, saved_mat1, -1, -2), grad),
                _scalar_tensor_like(grad, float(alpha)))
    return grad_input, grad_mat1, grad_mat2


def _autograd_addmm(name="addmm"):
    """Autograd wrapper for addmm(input, mat1, mat2, *, beta=1, alpha=1)."""
    def wrapper(input, mat1, mat2, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, input, mat1, mat2, *args, **kwargs)
        any_rg = (getattr(input, "requires_grad", False) or
                  getattr(mat1, "requires_grad", False) or
                  getattr(mat2, "requires_grad", False))
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                saved_input, saved_mat1, saved_mat2 = saved[0], saved[1], saved[2]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _addmm_backward(grad, input, mat1, mat2,
                                       saved_input, saved_mat1, saved_mat2,
                                       backward_keyset, args, kwargs)

            inputs = (input, mat1, mat2)
            node = Node(_backward, inputs)
            node_holder["node"] = node
            node.save_for_backward(*inputs)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


# ---- Task 2: Upsample backward (5 ops) ----

def _upsample_nearest2d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        N, C, H_in, W_in = saved_input.shape
        H_out, W_out = args[0]
        grad_input_np = np.zeros((N, C, H_in, W_in), dtype=grad_np.dtype)
        h_idx = (np.arange(H_out, dtype=np.float64) * H_in / H_out).astype(np.intp)
        w_idx = (np.arange(W_out, dtype=np.float64) * W_in / W_out).astype(np.intp)
        np.clip(h_idx, 0, H_in - 1, out=h_idx)
        np.clip(w_idx, 0, W_in - 1, out=w_idx)
        np.add.at(grad_input_np, (slice(None), slice(None), h_idx[:, None], w_idx[None, :]), grad_np)
        return (_from_numpy(np.ascontiguousarray(grad_input_np), input.dtype, input.device),)


def _upsample_nearest1d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        N, C, L_in = saved_input.shape
        (L_out,) = args[0]
        grad_input_np = np.zeros((N, C, L_in), dtype=grad_np.dtype)
        l_idx = (np.arange(L_out, dtype=np.float64) * L_in / L_out).astype(np.intp)
        np.clip(l_idx, 0, L_in - 1, out=l_idx)
        np.add.at(grad_input_np, (slice(None), slice(None), l_idx), grad_np)
        return (_from_numpy(np.ascontiguousarray(grad_input_np), input.dtype, input.device),)


def _upsample_bilinear2d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        N, C, H_in, W_in = saved_input.shape
        output_size = args[0]
        align_corners = args[1] if len(args) > 1 else kwargs.get("align_corners", False)
        H_out, W_out = output_size
        grad_input_np = np.zeros((N, C, H_in, W_in), dtype=grad_np.dtype)

        if align_corners and H_out > 1:
            h_scale = (H_in - 1) / (H_out - 1)
        else:
            h_scale = H_in / H_out
        if align_corners and W_out > 1:
            w_scale = (W_in - 1) / (W_out - 1)
        else:
            w_scale = W_in / W_out

        for oh in range(H_out):
            if align_corners:
                h = oh * h_scale
            else:
                h = (oh + 0.5) * h_scale - 0.5
            h = max(0.0, min(h, H_in - 1))
            h0 = int(np.floor(h))
            h1 = min(h0 + 1, H_in - 1)
            hf = h - h0
            for ow in range(W_out):
                if align_corners:
                    w = ow * w_scale
                else:
                    w = (ow + 0.5) * w_scale - 0.5
                w = max(0.0, min(w, W_in - 1))
                w0 = int(np.floor(w))
                w1 = min(w0 + 1, W_in - 1)
                wf = w - w0
                g = grad_np[:, :, oh, ow]
                grad_input_np[:, :, h0, w0] += g * (1 - hf) * (1 - wf)
                grad_input_np[:, :, h0, w1] += g * (1 - hf) * wf
                grad_input_np[:, :, h1, w0] += g * hf * (1 - wf)
                grad_input_np[:, :, h1, w1] += g * hf * wf
        return (_from_numpy(np.ascontiguousarray(grad_input_np), input.dtype, input.device),)


def _upsample_linear1d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        N, C, L_in = saved_input.shape
        output_size = args[0]
        align_corners = args[1] if len(args) > 1 else kwargs.get("align_corners", False)
        (L_out,) = output_size
        grad_input_np = np.zeros((N, C, L_in), dtype=grad_np.dtype)

        if align_corners and L_out > 1:
            scale = (L_in - 1) / (L_out - 1)
        else:
            scale = L_in / L_out

        for ol in range(L_out):
            if align_corners:
                l = ol * scale
            else:
                l = (ol + 0.5) * scale - 0.5
            l = max(0.0, min(l, L_in - 1))
            l0 = int(np.floor(l))
            l1 = min(l0 + 1, L_in - 1)
            lf = l - l0
            g = grad_np[:, :, ol]
            grad_input_np[:, :, l0] += g * (1 - lf)
            grad_input_np[:, :, l1] += g * lf
        return (_from_numpy(np.ascontiguousarray(grad_input_np), input.dtype, input.device),)


def _upsample_bicubic2d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        N, C, H_in, W_in = saved_input.shape
        output_size = args[0]
        align_corners = args[1] if len(args) > 1 else kwargs.get("align_corners", False)
        H_out, W_out = output_size
        grad_input_np = np.zeros((N, C, H_in, W_in), dtype=grad_np.dtype)
        A = -0.75

        def _cubic_weight(x):
            ax = abs(x)
            if ax <= 1:
                return (A + 2) * ax * ax * ax - (A + 3) * ax * ax + 1
            elif ax < 2:
                return A * ax * ax * ax - 5 * A * ax * ax + 8 * A * ax - 4 * A
            return 0.0

        if align_corners and H_out > 1:
            h_scale = (H_in - 1) / (H_out - 1)
        else:
            h_scale = H_in / H_out
        if align_corners and W_out > 1:
            w_scale = (W_in - 1) / (W_out - 1)
        else:
            w_scale = W_in / W_out

        for oh in range(H_out):
            if align_corners:
                h = oh * h_scale
            else:
                h = (oh + 0.5) * h_scale - 0.5
            for ow in range(W_out):
                if align_corners:
                    w = ow * w_scale
                else:
                    w = (ow + 0.5) * w_scale - 0.5
                g = grad_np[:, :, oh, ow]
                h_floor = int(np.floor(h))
                w_floor = int(np.floor(w))
                for dh in range(-1, 3):
                    ih = h_floor + dh
                    if ih < 0 or ih >= H_in:
                        continue
                    wh = _cubic_weight(h - ih)
                    for dw in range(-1, 3):
                        iw = w_floor + dw
                        if iw < 0 or iw >= W_in:
                            continue
                        ww = _cubic_weight(w - iw)
                        grad_input_np[:, :, ih, iw] += g * wh * ww
        return (_from_numpy(np.ascontiguousarray(grad_input_np), input.dtype, input.device),)


# ---- Task 3: Pool 1d/3d backward (8 ops) ----

def _max_pool1d_backward(grad, input, saved_input, out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        # Handle (values, indices) tuple from max_pool with return_indices
        out_np = _to_numpy(out[0] if isinstance(out, tuple) else out)

        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        dilation = args[3] if len(args) > 3 else kwargs.get("dilation", 1)

        kW = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        sW = stride if isinstance(stride, int) else stride[0]
        pW = padding if isinstance(padding, int) else padding[0]
        dW = dilation if isinstance(dilation, int) else dilation[0]

        N, C, W = input_np.shape
        _, _, W_out = grad_np.shape

        if pW > 0:
            input_padded = np.pad(input_np, ((0,0),(0,0),(pW,pW)),
                                  mode='constant', constant_values=-np.inf)
        else:
            input_padded = input_np

        grad_input_padded = np.zeros_like(input_padded)
        for ow in range(W_out):
            for kw in range(kW):
                iw = ow * sW + kw * dW
                if iw < input_padded.shape[2]:
                    mask = (input_padded[:, :, iw] == out_np[:, :, ow])
                    grad_input_padded[:, :, iw] += grad_np[:, :, ow] * mask

        if pW > 0:
            grad_input_np = grad_input_padded[:, :, pW:pW+W]
        else:
            grad_input_np = grad_input_padded

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


def _max_pool3d_backward(grad, input, saved_input, out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        out_np = _to_numpy(out[0] if isinstance(out, tuple) else out)

        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        dilation = args[3] if len(args) > 3 else kwargs.get("dilation", 1)

        kD, kH, kW = (kernel_size,)*3 if isinstance(kernel_size, int) else tuple(kernel_size)
        sD, sH, sW = (stride,)*3 if isinstance(stride, int) else tuple(stride)
        pD, pH, pW = (padding,)*3 if isinstance(padding, int) else tuple(padding)
        dD, dH, dW = (dilation,)*3 if isinstance(dilation, int) else tuple(dilation)

        N, C, D, H, W = input_np.shape
        _, _, D_out, H_out, W_out = grad_np.shape

        if pD > 0 or pH > 0 or pW > 0:
            input_padded = np.pad(input_np, ((0,0),(0,0),(pD,pD),(pH,pH),(pW,pW)),
                                  mode='constant', constant_values=-np.inf)
        else:
            input_padded = input_np

        grad_input_padded = np.zeros_like(input_padded)
        for od in range(D_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    for kd in range(kD):
                        for kh in range(kH):
                            for kw in range(kW):
                                id_ = od * sD + kd * dD
                                ih = oh * sH + kh * dH
                                iw = ow * sW + kw * dW
                                if id_ < input_padded.shape[2] and ih < input_padded.shape[3] and iw < input_padded.shape[4]:
                                    mask = (input_padded[:, :, id_, ih, iw] == out_np[:, :, od, oh, ow])
                                    grad_input_padded[:, :, id_, ih, iw] += grad_np[:, :, od, oh, ow] * mask

        if pD > 0 or pH > 0 or pW > 0:
            grad_input_np = grad_input_padded[:, :, pD:pD+D, pH:pH+H, pW:pW+W]
        else:
            grad_input_np = grad_input_padded

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


def _avg_pool1d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)

        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        count_include_pad = args[4] if len(args) > 4 else kwargs.get("count_include_pad", True)

        kW = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        sW = stride if isinstance(stride, int) else stride[0]
        pW = padding if isinstance(padding, int) else padding[0]

        N, C, W = input_np.shape
        _, _, W_out = grad_np.shape

        if pW > 0:
            grad_input_padded = np.zeros((N, C, W + 2*pW), dtype=input_np.dtype)
        else:
            grad_input_padded = np.zeros_like(input_np)

        for ow in range(W_out):
            w_start = ow * sW
            w_end = min(w_start + kW, grad_input_padded.shape[2])
            if count_include_pad:
                count = kW
            else:
                actual_w = min(w_end, W + pW) - max(w_start, pW)
                count = max(actual_w, 1)
            grad_input_padded[:, :, w_start:w_end] += grad_np[:, :, ow:ow+1] / count

        if pW > 0:
            grad_input_np = grad_input_padded[:, :, pW:pW+W]
        else:
            grad_input_np = grad_input_padded

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


def _avg_pool3d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)

        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        count_include_pad = args[4] if len(args) > 4 else kwargs.get("count_include_pad", True)

        kD, kH, kW = (kernel_size,)*3 if isinstance(kernel_size, int) else tuple(kernel_size)
        sD, sH, sW = (stride,)*3 if isinstance(stride, int) else tuple(stride)
        pD, pH, pW = (padding,)*3 if isinstance(padding, int) else tuple(padding)

        N, C, D, H, W = input_np.shape
        _, _, D_out, H_out, W_out = grad_np.shape

        if pD > 0 or pH > 0 or pW > 0:
            grad_input_padded = np.zeros((N, C, D+2*pD, H+2*pH, W+2*pW), dtype=input_np.dtype)
        else:
            grad_input_padded = np.zeros_like(input_np)

        for od in range(D_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    d_start = od * sD
                    h_start = oh * sH
                    w_start = ow * sW
                    d_end = min(d_start + kD, grad_input_padded.shape[2])
                    h_end = min(h_start + kH, grad_input_padded.shape[3])
                    w_end = min(w_start + kW, grad_input_padded.shape[4])
                    if count_include_pad:
                        count = kD * kH * kW
                    else:
                        actual_d = min(d_end, D+pD) - max(d_start, pD)
                        actual_h = min(h_end, H+pH) - max(h_start, pH)
                        actual_w = min(w_end, W+pW) - max(w_start, pW)
                        count = max(actual_d * actual_h * actual_w, 1)
                    grad_input_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
                        grad_np[:, :, od:od+1, oh:oh+1, ow:ow+1] / count

        if pD > 0 or pH > 0 or pW > 0:
            grad_input_np = grad_input_padded[:, :, pD:pD+D, pH:pH+H, pW:pW+W]
        else:
            grad_input_np = grad_input_padded

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


def _adaptive_avg_pool1d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        N, C, L = input_np.shape
        output_size = args[0]
        oL = output_size if isinstance(output_size, int) else output_size[0]

        grad_input_np = np.zeros_like(input_np)
        for ol in range(oL):
            l_start = ol * L // oL
            l_end = (ol + 1) * L // oL
            count = l_end - l_start
            grad_input_np[:, :, l_start:l_end] += grad_np[:, :, ol:ol+1] / count

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


def _adaptive_avg_pool3d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        N, C, D, H, W = input_np.shape
        output_size = args[0]
        if isinstance(output_size, int):
            oD = oH = oW = output_size
        else:
            oD, oH, oW = output_size

        grad_input_np = np.zeros_like(input_np)
        for od in range(oD):
            d_start = od * D // oD
            d_end = (od + 1) * D // oD
            for oh in range(oH):
                h_start = oh * H // oH
                h_end = (oh + 1) * H // oH
                for ow in range(oW):
                    w_start = ow * W // oW
                    w_end = (ow + 1) * W // oW
                    count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start)
                    grad_input_np[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
                        grad_np[:, :, od:od+1, oh:oh+1, ow:ow+1] / count

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


def _adaptive_max_pool1d_backward(grad, input, saved_input, out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        out_np = _to_numpy(out[0] if isinstance(out, tuple) else out)
        N, C, L = input_np.shape
        output_size = args[0]
        oL = output_size if isinstance(output_size, int) else output_size[0]

        grad_input_np = np.zeros_like(input_np)
        for ol in range(oL):
            l_start = ol * L // oL
            l_end = (ol + 1) * L // oL
            for il in range(l_start, l_end):
                mask = (input_np[:, :, il] == out_np[:, :, ol])
                grad_input_np[:, :, il] += grad_np[:, :, ol] * mask

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


def _adaptive_max_pool2d_backward(grad, input, saved_input, out, keyset, args, kwargs):
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy
        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        out_np = _to_numpy(out[0] if isinstance(out, tuple) else out)
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
                for ih in range(h_start, h_end):
                    for iw in range(w_start, w_end):
                        mask = (input_np[:, :, ih, iw] == out_np[:, :, oh, ow])
                        grad_input_np[:, :, ih, iw] += grad_np[:, :, oh, ow] * mask

        return (_from_numpy(np.ascontiguousarray(grad_input_np.astype(input_np.dtype)),
                            input.dtype, input.device),)


# ---- Task 5: Indexing backward (4 ops) ----

def _autograd_index_put(name="index_put"):
    """Autograd wrapper for index_put(input, indices, values, accumulate=False)."""
    def wrapper(a, indices, values, accumulate=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, indices, values, accumulate)
        a_rg = getattr(a, "requires_grad", False)
        v_rg = getattr(values, "requires_grad", False)
        if GradMode.enabled and (a_rg or v_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _index_put_backward(grad, a, values, indices, accumulate, backward_keyset)

            inputs = tuple(t for t in (a, values) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _index_put_backward(grad, a, values, indices, accumulate, keyset):
    with _grad_context(keyset):
        grad_self = None
        if getattr(a, "requires_grad", False):
            grad_self = grad.clone()
            if not accumulate:
                zeros = redispatch("zeros", keyset, (1,), dtype=grad.dtype, device=grad.device)
                grad_self = redispatch("index_put", keyset, grad_self, indices, zeros, False)
        grad_values = None
        if getattr(values, "requires_grad", False):
            grad_values = grad[tuple(i for i in indices)]
    return grad_self, grad_values


def _autograd_index_put_inplace(name="index_put_"):
    """Autograd wrapper for index_put_ (in-place)."""
    def wrapper(a, indices, values, accumulate=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, indices, values, accumulate)
        a_rg = getattr(a, "requires_grad", False)
        v_rg = getattr(values, "requires_grad", False)
        if GradMode.enabled and (a_rg or v_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _index_put_backward(grad, a, values, indices, accumulate, backward_keyset)

            inputs = tuple(t for t in (a, values) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_scatter_add_inplace(name="scatter_add_"):
    """Autograd wrapper for scatter_add_(self, dim, index, src)."""
    def wrapper(a, dim, index, src):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim, index, src)
        a_rg = getattr(a, "requires_grad", False)
        src_rg = getattr(src, "requires_grad", False)
        if GradMode.enabled and (a_rg or src_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _scatter_add_inplace_backward(grad, a, src, dim, index, backward_keyset)

            inputs = tuple(t for t in (a, src) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _scatter_add_inplace_backward(grad, a, src, dim, index, keyset):
    with _grad_context(keyset):
        grad_self = grad.clone() if getattr(a, "requires_grad", False) else None
        grad_src = None
        if getattr(src, "requires_grad", False):
            grad_src = redispatch("gather", keyset, grad, dim, index)
    return grad_self, grad_src


def _autograd_index_add_inplace(name="index_add_"):
    """Autograd wrapper for index_add_(self, dim, index, source, alpha=1)."""
    def wrapper(a, dim, index, source, alpha=1):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim, index, source, alpha)
        a_rg = getattr(a, "requires_grad", False)
        src_rg = getattr(source, "requires_grad", False)
        if GradMode.enabled and (a_rg or src_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _index_add_inplace_backward(grad, a, source, dim, index, alpha, backward_keyset)

            inputs = tuple(t for t in (a, source) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _index_add_inplace_backward(grad, a, source, dim, index, alpha, keyset):
    with _grad_context(keyset):
        grad_self = grad.clone() if getattr(a, "requires_grad", False) else None
        grad_src = None
        if getattr(source, "requires_grad", False):
            grad_src = redispatch("index_select", keyset, grad, dim, index)
            if alpha != 1:
                grad_src = redispatch("mul", keyset, grad_src, _scalar_tensor_like(grad, float(alpha)))
    return grad_self, grad_src


# ---- Task 6: Math backward (5 ops) ----

def _logsumexp_backward(grad, _a, saved_a, keyset, args, kwargs):
    dim = args[0] if args else kwargs.get("dim", None)
    keepdim = args[1] if len(args) > 1 else kwargs.get("keepdim", False)
    with _grad_context(keyset):
        sm = redispatch("softmax", keyset, saved_a, dim=dim)
        if not keepdim and dim is not None:
            grad = redispatch("unsqueeze", keyset, grad, dim)
        return (redispatch("mul", keyset, grad, sm),)


def _autograd_einsum(name="einsum"):
    """Autograd wrapper for einsum(equation, operands_list)."""
    def wrapper(equation, operands):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, equation, operands)
        any_rg = any(getattr(op, "requires_grad", False) for op in operands)
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _einsum_backward(grad, equation, operands, backward_keyset)

            inputs = tuple(op for op in operands if hasattr(op, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            for op in operands:
                if hasattr(op, "requires_grad"):
                    node.save_for_backward(op)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _einsum_backward(grad, equation, operands, keyset):
    """Backward for einsum using numpy."""
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        parts = equation.replace(' ', '').split('->')
        if len(parts) != 2:
            # Implicit output — fallback to numerical
            return tuple(None for _ in operands)
        input_part, output_subs = parts
        input_subs_list = input_part.split(',')

        grads = []
        grad_np = _to_numpy(grad)
        operand_nps = [_to_numpy(op) for op in operands]

        for i, op in enumerate(operands):
            if not getattr(op, "requires_grad", False):
                grads.append(None)
                continue
            # Build einsum for gradient: contract grad with all other operands
            other_subs = []
            other_arrays = []
            for j, (subs, arr) in enumerate(zip(input_subs_list, operand_nps)):
                if j != i:
                    other_subs.append(subs)
                    other_arrays.append(arr)
            grad_eq = ','.join([output_subs] + other_subs) + '->' + input_subs_list[i]
            grad_np_i = np.einsum(grad_eq, grad_np, *other_arrays)
            grads.append(_from_numpy(np.ascontiguousarray(grad_np_i), op.dtype, op.device))
        return tuple(grads)


def _baddbmm_backward(grad, input, batch1, batch2, saved_input, saved_batch1, saved_batch2, keyset, args, kwargs):
    beta = kwargs.get("beta", args[0] if args else 1)
    alpha = kwargs.get("alpha", args[1] if len(args) > 1 else 1)
    with _grad_context(keyset):
        grad_input = None
        grad_batch1 = None
        grad_batch2 = None
        if getattr(input, "requires_grad", False):
            grad_input = redispatch("mul", keyset, grad, _scalar_tensor_like(grad, float(beta)))
            grad_input = reduce_grad(grad_input, input.shape)
        if getattr(batch1, "requires_grad", False):
            grad_batch1 = redispatch("mul", keyset,
                redispatch("matmul", keyset, grad, redispatch("transpose", keyset, saved_batch2, -1, -2)),
                _scalar_tensor_like(grad, float(alpha)))
        if getattr(batch2, "requires_grad", False):
            grad_batch2 = redispatch("mul", keyset,
                redispatch("matmul", keyset, redispatch("transpose", keyset, saved_batch1, -1, -2), grad),
                _scalar_tensor_like(grad, float(alpha)))
    return grad_input, grad_batch1, grad_batch2


def _autograd_baddbmm(name="baddbmm"):
    """Autograd wrapper for baddbmm(input, batch1, batch2, *, beta=1, alpha=1)."""
    def wrapper(input, batch1, batch2, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, input, batch1, batch2, *args, **kwargs)
        any_rg = (getattr(input, "requires_grad", False) or
                  getattr(batch1, "requires_grad", False) or
                  getattr(batch2, "requires_grad", False))
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                saved_input, saved_batch1, saved_batch2 = saved[0], saved[1], saved[2]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _baddbmm_backward(grad, input, batch1, batch2,
                                         saved_input, saved_batch1, saved_batch2,
                                         backward_keyset, args, kwargs)

            inputs = (input, batch1, batch2)
            node = Node(_backward, inputs)
            node_holder["node"] = node
            node.save_for_backward(*inputs)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _cross_backward(grad, a, b, saved_a, saved_b, dim, keyset):
    """Backward for cross: d/da cross(a,b) = cross(b, grad), d/db cross(a,b) = cross(grad, a)."""
    with _grad_context(keyset):
        grad_a = redispatch("cross", keyset, saved_b, grad, dim) if getattr(a, "requires_grad", False) else None
        grad_b = redispatch("cross", keyset, grad, saved_a, dim) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


def _autograd_cross(name="cross"):
    """Autograd wrapper for cross(a, b, dim)."""
    def wrapper(a, b, dim=None):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, dim)
        a_rg = getattr(a, "requires_grad", False)
        b_rg = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_rg or b_rg):
            node_holder = {}

            def _backward(grad):
                saved_a, saved_b = node_holder["node"].saved_tensors()
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _cross_backward(grad, a, b, saved_a, saved_b, dim, backward_keyset)

            node = Node(_backward, (a, b))
            node_holder["node"] = node
            node.save_for_backward(a, b)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_cummin(name):
    """Custom autograd wrapper for cummin which returns (values, indices) namedtuple."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        result = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and a.requires_grad:
            values = result.values if hasattr(result, 'values') else result[0]
            indices = result.indices if hasattr(result, 'indices') else result[1]
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _cummin_backward(grad, a, indices, backward_keyset, args, kwargs)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            values.grad_fn = node
            values.requires_grad = True
            if hasattr(result, '_replace'):
                result = result._replace(values=values)
            else:
                result = (values, indices)
        return result

    return wrapper


def _cummin_backward(grad, a, indices, keyset, args, kwargs):
    """Backward for cummin: same pattern as cummax — scatter-add to index positions."""
    dim = args[0] if args else kwargs.get("dim", 0)
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        idx_np = _to_numpy(indices).astype(np.int64)
        result = np.zeros(a.shape, dtype=grad_np.dtype)
        d = dim if dim >= 0 else dim + len(a.shape)
        it = np.nditer(idx_np, flags=['multi_index'])
        while not it.finished:
            mi = list(it.multi_index)
            src_idx = tuple(it.multi_index)
            mi[d] = int(it[0])
            result[tuple(mi)] += grad_np[src_idx]
            it.iternext()
        return (_from_numpy(result, a.dtype, a.device),)


# ---- Task 7: Other backward (3 ops) ----

def _autograd_grid_sample(name="grid_sample"):
    """Autograd wrapper for grid_sample(input, grid, mode, padding_mode, align_corners)."""
    def wrapper(input, grid, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, input, grid, *args, **kwargs)
        i_rg = getattr(input, "requires_grad", False)
        g_rg = getattr(grid, "requires_grad", False)
        if GradMode.enabled and (i_rg or g_rg):
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                saved_input, saved_grid = saved[0], saved[1]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _grid_sample_backward(grad, input, grid, saved_input, saved_grid,
                                             backward_keyset, args, kwargs)

            inputs = (input, grid)
            node = Node(_backward, inputs)
            node_holder["node"] = node
            node.save_for_backward(*inputs)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _grid_sample_backward(grad, input, grid, saved_input, saved_grid, keyset, args, kwargs):
    """Backward for grid_sample using numpy — bilinear mode only."""
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        grid_np = _to_numpy(saved_grid)

        mode = args[0] if args else kwargs.get("mode", "bilinear")
        padding_mode = args[1] if len(args) > 1 else kwargs.get("padding_mode", "zeros")
        align_corners = args[2] if len(args) > 2 else kwargs.get("align_corners", False)

        N, C, H, W = input_np.shape
        _, H_out, W_out, _ = grid_np.shape

        grad_input_np = np.zeros_like(input_np) if getattr(input, "requires_grad", False) else None
        grad_grid_np = np.zeros_like(grid_np) if getattr(grid, "requires_grad", False) else None

        def _unnormalize(coord, size, align):
            if align:
                return (coord + 1) * (size - 1) / 2
            else:
                return ((coord + 1) * size - 1) / 2

        for n in range(N):
            for oh in range(H_out):
                for ow in range(W_out):
                    gx = grid_np[n, oh, ow, 0]
                    gy = grid_np[n, oh, ow, 1]
                    ix = _unnormalize(gx, W, align_corners)
                    iy = _unnormalize(gy, H, align_corners)

                    if mode == "nearest":
                        ix_round = int(np.round(ix))
                        iy_round = int(np.round(iy))
                        if 0 <= ix_round < W and 0 <= iy_round < H:
                            if grad_input_np is not None:
                                grad_input_np[n, :, iy_round, ix_round] += grad_np[n, :, oh, ow]
                    else:  # bilinear
                        ix0 = int(np.floor(ix))
                        iy0 = int(np.floor(iy))
                        ix1 = ix0 + 1
                        iy1 = iy0 + 1
                        fx = ix - ix0
                        fy = iy - iy0

                        def _safe_get(arr, n, c, h, w):
                            if 0 <= h < H and 0 <= w < W:
                                return arr[n, c, h, w]
                            return 0.0

                        if grad_input_np is not None:
                            for c in range(C):
                                g = grad_np[n, c, oh, ow]
                                if 0 <= iy0 < H and 0 <= ix0 < W:
                                    grad_input_np[n, c, iy0, ix0] += g * (1-fy) * (1-fx)
                                if 0 <= iy0 < H and 0 <= ix1 < W:
                                    grad_input_np[n, c, iy0, ix1] += g * (1-fy) * fx
                                if 0 <= iy1 < H and 0 <= ix0 < W:
                                    grad_input_np[n, c, iy1, ix0] += g * fy * (1-fx)
                                if 0 <= iy1 < H and 0 <= ix1 < W:
                                    grad_input_np[n, c, iy1, ix1] += g * fy * fx

                        if grad_grid_np is not None:
                            for c in range(C):
                                g = grad_np[n, c, oh, ow]
                                v00 = _safe_get(input_np, n, c, iy0, ix0)
                                v01 = _safe_get(input_np, n, c, iy0, ix1)
                                v10 = _safe_get(input_np, n, c, iy1, ix0)
                                v11 = _safe_get(input_np, n, c, iy1, ix1)
                                dx = (v01 - v00) * (1-fy) + (v11 - v10) * fy
                                dy = (v10 - v00) * (1-fx) + (v11 - v01) * fx
                                if align_corners:
                                    dx *= (W - 1) / 2
                                    dy *= (H - 1) / 2
                                else:
                                    dx *= W / 2
                                    dy *= H / 2
                                grad_grid_np[n, oh, ow, 0] += g * dx
                                grad_grid_np[n, oh, ow, 1] += g * dy

        grad_input_t = None
        if grad_input_np is not None:
            grad_input_t = _from_numpy(np.ascontiguousarray(grad_input_np), input.dtype, input.device)
        grad_grid_t = None
        if grad_grid_np is not None:
            grad_grid_t = _from_numpy(np.ascontiguousarray(grad_grid_np), grid.dtype, grid.device)
        return grad_input_t, grad_grid_t


def _autograd_affine_grid(name="affine_grid"):
    """Autograd wrapper for affine_grid(theta, size, align_corners)."""
    def wrapper(theta, size, align_corners=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, theta, size, align_corners)
        if GradMode.enabled and getattr(theta, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _affine_grid_backward(grad, theta, size, align_corners, backward_keyset)

            node = Node(_backward, (theta,))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _affine_grid_backward(grad, theta, size, align_corners, keyset):
    """Backward for affine_grid: grad_theta = grad^T @ base_grid."""
    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        N = size[0]
        if len(size) == 4:
            _, _, H, W = size
            if align_corners:
                h_points = np.linspace(-1, 1, H)
                w_points = np.linspace(-1, 1, W)
            else:
                h_points = np.linspace(-1, 1, 2*H+1)[1::2]
                w_points = np.linspace(-1, 1, 2*W+1)[1::2]
            grid_h, grid_w = np.meshgrid(h_points, w_points, indexing='ij')
            ones = np.ones_like(grid_h)
            base = np.stack([grid_w.ravel(), grid_h.ravel(), ones.ravel()], axis=1)  # (H*W, 3)
            # grad is (N, H, W, 2), reshape to (N, H*W, 2)
            grad_flat = grad_np.reshape(N, H*W, 2)
            # grad_theta = grad_flat^T @ base = (N, 2, H*W) @ (H*W, 3) = (N, 2, 3)
            grad_theta_np = np.matmul(grad_flat.transpose(0, 2, 1), base[np.newaxis])
        else:
            _, _, D, H, W = size
            if align_corners:
                d_points = np.linspace(-1, 1, D)
                h_points = np.linspace(-1, 1, H)
                w_points = np.linspace(-1, 1, W)
            else:
                d_points = np.linspace(-1, 1, 2*D+1)[1::2]
                h_points = np.linspace(-1, 1, 2*H+1)[1::2]
                w_points = np.linspace(-1, 1, 2*W+1)[1::2]
            grid_d, grid_h, grid_w = np.meshgrid(d_points, h_points, w_points, indexing='ij')
            ones = np.ones_like(grid_d)
            base = np.stack([grid_w.ravel(), grid_h.ravel(), grid_d.ravel(), ones.ravel()], axis=1)
            grad_flat = grad_np.reshape(N, D*H*W, 3)
            grad_theta_np = np.matmul(grad_flat.transpose(0, 2, 1), base[np.newaxis])

        return (_from_numpy(np.ascontiguousarray(grad_theta_np.astype(np.float32)),
                            theta.dtype, theta.device),)


def _masked_fill_inplace_backward(grad, _a, saved_a, args, keyset):
    """Backward for masked_fill_ (in-place)."""
    mask = args[0]
    with _grad_context(keyset):
        zeros = redispatch("zeros", keyset, grad.shape, dtype=grad.dtype, device=grad.device)
        grad_input = redispatch("where", keyset, mask, zeros, grad)
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
    # Phase 8: Missing backward ops
    # Part A: Missing backward for existing registered ops
    ("sub", lambda: _autograd_binary("sub", _sub_backward, save_inputs=False)),
    ("mm", lambda: _autograd_binary("mm", _mm_backward)),
    ("bmm", lambda: _autograd_binary("bmm", _bmm_backward)),
    ("lerp", lambda: _autograd_lerp("lerp")),
    ("addcmul", lambda: _autograd_addcmul("addcmul")),
    ("addcdiv", lambda: _autograd_addcdiv("addcdiv")),
    ("amax", lambda: _autograd_unary_args("amax", _amax_backward)),
    ("amin", lambda: _autograd_unary_args("amin", _amin_backward)),
    # Part B: Backward for new ops
    ("log1p", lambda: _autograd_unary("log1p", _log1p_backward)),
    ("expm1", lambda: _autograd_unary("expm1", _expm1_backward)),
    ("reciprocal", lambda: _autograd_unary("reciprocal", _reciprocal_backward)),
    ("maximum", lambda: _autograd_binary("maximum", _maximum_backward)),
    ("minimum", lambda: _autograd_binary("minimum", _minimum_backward)),
    ("dot", lambda: _autograd_binary("dot", _dot_backward)),
    ("outer", lambda: _autograd_binary("outer", _outer_backward)),
    ("mv", lambda: _autograd_binary("mv", _mv_backward)),
    ("flatten", lambda: _autograd_unary_args("flatten", _flatten_backward, save_input=False)),
    ("unflatten", lambda: _autograd_unary_args("unflatten", _unflatten_backward, save_input=False)),
    ("movedim", lambda: _autograd_unary_args("movedim", _movedim_backward, save_input=False)),
    ("moveaxis", lambda: _autograd_unary_args("moveaxis", _movedim_backward, save_input=False)),
    ("diagonal", lambda: _autograd_unary_args("diagonal", _diagonal_backward, save_input=True)),
    ("hardswish", lambda: _autograd_unary("hardswish", _hardswish_backward)),
    ("hardsigmoid", lambda: _autograd_unary("hardsigmoid", _hardsigmoid_backward)),
    ("softsign", lambda: _autograd_unary("softsign", _softsign_backward)),
    ("selu", lambda: _autograd_unary("selu", _selu_backward)),
    ("celu", lambda: _autograd_unary_args("celu", _celu_backward)),
    ("threshold", lambda: _autograd_unary_args("threshold", _threshold_backward, save_input=True)),
    ("instance_norm", lambda: _autograd_unary_args("instance_norm", _instance_norm_backward)),
    ("normalize", lambda: _autograd_unary_args("normalize", _normalize_backward)),
    # Round 2 — Tier 1: Training-critical
    ("clamp_min", lambda: _autograd_unary_args("clamp_min", _clamp_min_backward)),
    ("clamp_max", lambda: _autograd_unary_args("clamp_max", _clamp_max_backward)),
    ("min", lambda: _autograd_binary("min", _min_backward)),
    ("max", lambda: _autograd_binary("max", _max_backward)),
    ("cumprod", lambda: _autograd_unary_args("cumprod", _cumprod_backward)),
    ("repeat_interleave", lambda: _autograd_unary_args("repeat_interleave", _repeat_interleave_backward)),
    ("scatter", lambda: _autograd_scatter("scatter")),
    ("floor_divide", lambda: _autograd_binary("floor_divide", _floor_divide_backward, save_inputs=False)),
    # Round 2 — Tier 2: Trig/math
    ("tan", lambda: _autograd_unary("tan", _tan_backward)),
    ("asin", lambda: _autograd_unary("asin", _asin_backward)),
    ("acos", lambda: _autograd_unary("acos", _acos_backward)),
    ("atan", lambda: _autograd_unary("atan", _atan_backward)),
    ("atan2", lambda: _autograd_binary("atan2", _atan2_backward)),
    ("sinh", lambda: _autograd_unary("sinh", _sinh_backward)),
    ("cosh", lambda: _autograd_unary("cosh", _cosh_backward)),
    ("asinh", lambda: _autograd_unary("asinh", _asinh_backward)),
    ("acosh", lambda: _autograd_unary("acosh", _acosh_backward)),
    ("atanh", lambda: _autograd_unary("atanh", _atanh_backward)),
    ("log10", lambda: _autograd_unary("log10", _log10_backward)),
    ("erfc", lambda: _autograd_unary("erfc", _erfc_backward)),
    ("logaddexp", lambda: _autograd_binary("logaddexp", _logaddexp_backward)),
    ("logaddexp2", lambda: _autograd_binary("logaddexp2", _logaddexp2_backward)),
    # Round 2 — Tier 3: Less common
    ("fmin", lambda: _autograd_binary("fmin", _fmin_backward)),
    ("fmax", lambda: _autograd_binary("fmax", _fmax_backward)),
    ("fmod", lambda: _autograd_binary("fmod", _fmod_backward)),
    ("hypot", lambda: _autograd_binary("hypot", _hypot_backward)),
    ("remainder", lambda: _autograd_binary("remainder", _remainder_backward)),
    ("rot90", lambda: _autograd_unary_args("rot90", _rot90_backward, save_input=False)),
    ("take", lambda: _autograd_unary_args("take", _take_backward, save_input=True)),
    ("take_along_dim", lambda: _autograd_unary_args("take_along_dim", _take_along_dim_backward, save_input=True)),
    ("cummax", lambda: _autograd_cummax("cummax")),
    # P0 Gap Fix — Round 3
    # Task 1: addmm
    ("addmm", lambda: _autograd_addmm("addmm")),
    # Task 2: Upsample backward
    ("upsample_nearest2d", lambda: _autograd_pool("upsample_nearest2d", _upsample_nearest2d_backward)),
    ("upsample_nearest1d", lambda: _autograd_pool("upsample_nearest1d", _upsample_nearest1d_backward)),
    ("upsample_bilinear2d", lambda: _autograd_pool("upsample_bilinear2d", _upsample_bilinear2d_backward)),
    ("upsample_linear1d", lambda: _autograd_pool("upsample_linear1d", _upsample_linear1d_backward)),
    ("upsample_bicubic2d", lambda: _autograd_pool("upsample_bicubic2d", _upsample_bicubic2d_backward)),
    # Task 3: Pool 1d/3d backward
    ("max_pool1d", lambda: _autograd_pool("max_pool1d", _max_pool1d_backward)),
    ("max_pool3d", lambda: _autograd_pool("max_pool3d", _max_pool3d_backward)),
    ("avg_pool1d", lambda: _autograd_pool("avg_pool1d", _avg_pool1d_backward)),
    ("avg_pool3d", lambda: _autograd_pool("avg_pool3d", _avg_pool3d_backward)),
    ("adaptive_avg_pool1d", lambda: _autograd_pool("adaptive_avg_pool1d", _adaptive_avg_pool1d_backward)),
    ("adaptive_avg_pool3d", lambda: _autograd_pool("adaptive_avg_pool3d", _adaptive_avg_pool3d_backward)),
    ("adaptive_max_pool1d", lambda: _autograd_pool("adaptive_max_pool1d", _adaptive_max_pool1d_backward)),
    ("adaptive_max_pool2d", lambda: _autograd_pool("adaptive_max_pool2d", _adaptive_max_pool2d_backward)),
    # Task 4: Conv 3d
    ("conv3d", lambda: _autograd_conv("conv3d")),
    ("conv_transpose3d", lambda: _autograd_conv("conv_transpose3d")),
    # Task 5: Indexing backward
    ("index_put", lambda: _autograd_index_put("index_put")),
    ("index_put_", lambda: _autograd_index_put_inplace("index_put_")),
    ("scatter_add_", lambda: _autograd_scatter_add_inplace("scatter_add_")),
    ("index_add_", lambda: _autograd_index_add_inplace("index_add_")),
    # Task 6: Math backward
    ("logsumexp", lambda: _autograd_unary_args("logsumexp", _logsumexp_backward)),
    ("einsum", lambda: _autograd_einsum("einsum")),
    ("baddbmm", lambda: _autograd_baddbmm("baddbmm")),
    ("cross", lambda: _autograd_cross("cross")),
    ("cummin", lambda: _autograd_cummin("cummin")),
    # Task 7: Other backward
    ("grid_sample", lambda: _autograd_grid_sample("grid_sample")),
    ("affine_grid", lambda: _autograd_affine_grid("affine_grid")),
    ("masked_fill_", lambda: _autograd_inplace("masked_fill_", _masked_fill_inplace_backward, save_input=False)),
):
    if len(_entry) == 2:
        _name, _factory = _entry
        _register_autograd_op(_name, _factory)
    else:
        _name, _factory, _include_meta = _entry
        _register_autograd_op(_name, _factory, include_meta=_include_meta)
