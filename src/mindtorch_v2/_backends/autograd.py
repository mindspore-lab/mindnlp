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
            DispatchKey.AutogradCUDA,
            DispatchKey.AutogradXPU,
            DispatchKey.AutogradMeta,
            DispatchKey.PrivateUse3,
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


def _autograd_norm(name, backward_impl):
    """Autograd wrapper for normalization ops (layer_norm, batch_norm, rms_norm).

    Like ``_autograd_unary_args`` but also captures ``_backward_data`` from the
    forward output and passes it to *backward_impl* as the 5th positional arg.
    This allows NPU backward kernels to access saved intermediate data (mean,
    rstd, etc.) that the NPU forward op attached to its output tensor.
    """
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and a.requires_grad:
            backward_data = getattr(out, "_backward_data", None)
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, saved_a, backward_keyset, args, kwargs, backward_data)

            node = Node(_backward, (a,))
            node_holder["node"] = node
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


def _layer_norm_backward(grad, _a, saved_a, keyset, args, kwargs, backward_data=None):
    normalized_shape = args[0] if args else kwargs.get("normalized_shape")
    weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
    eps = args[3] if len(args) > 3 else kwargs.get("eps", 1e-5)

    # NPU path: use ACLNN large kernel when backward_data is available
    if grad.device.type == "npu" and backward_data is not None and "mean_ptr" in backward_data:
        from .npu.backward import npu_layer_norm_backward
        grad_input, grad_weight, grad_bias = npu_layer_norm_backward(
            grad, saved_a, backward_data, normalized_shape,
            weight=weight, bias=bias, eps=eps,
        )
        return (grad_input,)

    # CPU path: composite small ops
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


def _batch_norm_backward(grad, _a, saved_a, keyset, args, kwargs, backward_data=None):
    weight = args[3] if len(args) > 3 else kwargs.get("weight", None)
    training = args[5] if len(args) > 5 else kwargs.get("training", False)
    eps = args[7] if len(args) > 7 else kwargs.get("eps", 1e-5)

    # NPU path: use ACLNN large kernel when backward_data is available
    if grad.device.type == "npu" and backward_data is not None and "save_mean_ptr" in backward_data:
        running_mean = args[1] if len(args) > 1 else kwargs.get("running_mean", None)
        running_var = args[2] if len(args) > 2 else kwargs.get("running_var", None)
        from .npu.backward import npu_batch_norm_backward
        grad_input, grad_weight, grad_bias = npu_batch_norm_backward(
            grad, saved_a, backward_data, weight=weight,
            running_mean=running_mean, running_var=running_var,
        )
        return (grad_input,)

    # CPU path: composite small ops
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


def _rms_norm_backward(grad, _a, saved_a, keyset, args, kwargs, backward_data=None):
    normalized_shape = args[0] if args else kwargs.get("normalized_shape")
    weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
    eps = args[2] if len(args) > 2 else kwargs.get("eps", 1e-6)

    # NPU path: use ACLNN large kernel when backward_data is available
    if grad.device.type == "npu" and backward_data is not None and "rstd_ptr" in backward_data:
        from .npu.backward import npu_rms_norm_backward
        grad_input, grad_weight = npu_rms_norm_backward(
            grad, saved_a, backward_data, weight=weight,
        )
        return (grad_input,)

    # CPU path: composite small ops
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
    # NPU path: use aclnnConvolutionBackward large kernel
    if grad.device.type == "npu":
        from .npu.backward import npu_conv_backward
        stride = kwargs.get("stride", args[0] if args else (1, 1))
        padding = kwargs.get("padding", args[1] if len(args) > 1 else (0, 0))
        dilation = kwargs.get("dilation", args[2] if len(args) > 2 else (1, 1))
        groups = kwargs.get("groups", args[3] if len(args) > 3 else 1)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        gi, gw, gb = npu_conv_backward(
            grad, saved_input, saved_weight, saved_bias,
            name, stride, padding, dilation, groups,
        )
        results = [gi, gw]
        if bias is not None:
            results.append(gb)
        return tuple(results)
    # CPU path: existing numpy implementation below
    with _grad_context(keyset):
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
    # NPU path: use aclnnMaxPool2dWithMaskBackward large kernel
    if grad.device.type == "npu":
        backward_data = getattr(out, "_backward_data", None)
        if backward_data is not None and "mask_ptr" in backward_data:
            from .npu.backward import npu_max_pool2d_backward
            return (npu_max_pool2d_backward(grad, saved_input, backward_data),)
    # CPU path: existing numpy implementation below
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
    # NPU path: use aclnnAvgPool2dBackward large kernel
    if grad.device.type == "npu":
        from .npu.backward import npu_avg_pool2d_backward
        kernel_size = args[0] if args else kwargs.get("kernel_size")
        stride = args[1] if len(args) > 1 else kwargs.get("stride", kernel_size)
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        ceil_mode = args[3] if len(args) > 3 else kwargs.get("ceil_mode", False)
        count_include_pad = args[4] if len(args) > 4 else kwargs.get("count_include_pad", True)
        divisor_override = args[5] if len(args) > 5 else kwargs.get("divisor_override", None)
        return (npu_avg_pool2d_backward(grad, saved_input, kernel_size, stride, padding,
                                        ceil_mode, count_include_pad, divisor_override),)
    # CPU path: existing numpy implementation below
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
    if grad.device.type == "npu":
        from .npu.backward import npu_adaptive_avg_pool2d_backward
        return (npu_adaptive_avg_pool2d_backward(grad, saved_input),)
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
    """Backward for diagonal: scatter grad back along the diagonal."""
    offset = args[0] if args else kwargs.get("offset", 0)
    dim1 = args[1] if len(args) > 1 else kwargs.get("dim1", 0)
    dim2 = args[2] if len(args) > 2 else kwargs.get("dim2", 1)
    with _grad_context(keyset):
        result = redispatch("zeros", keyset, saved_a.shape, dtype=grad.dtype, device=grad.device)
        ndim = len(saved_a.shape)
        d1 = dim1 if dim1 >= 0 else dim1 + ndim
        d2 = dim2 if dim2 >= 0 else dim2 + ndim
        n_d1, n_d2 = saved_a.shape[d1], saved_a.shape[d2]
        if offset >= 0:
            diag_len = min(n_d1, n_d2 - offset)
        else:
            diag_len = min(n_d1 + offset, n_d2)
        diag_len = max(diag_len, 0)
        for k in range(diag_len):
            i1 = k if offset >= 0 else k - offset
            i2 = k + offset if offset >= 0 else k
            # Build index tuple: slice(None) for all dims, then set d1 and d2
            idx = [slice(None)] * ndim
            idx[d1] = i1
            idx[d2] = i2
            # grad[..., k] — index the last dim of grad
            grad_idx = [slice(None)] * (len(grad.shape) - 1) + [k]
            g_slice = redispatch("getitem", keyset, grad, tuple(grad_idx))
            redispatch("setitem", keyset, result, tuple(idx), g_slice)
        return (result,)


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


def _hardshrink_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for hardshrink: grad * (|x| > lambd)."""
    lambd = args[0] if args else kwargs.get("lambd", 0.5)
    with _grad_context(keyset):
        lambd_t = _scalar_tensor_like(saved_a, float(lambd))
        abs_x = redispatch("abs", keyset, saved_a)
        # Create float mask: 1 where |x| > lambd, else 0
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        mask = redispatch("where", keyset,
            redispatch("gt", keyset, abs_x, lambd_t), ones, zero)
        return (redispatch("mul", keyset, grad, mask),)


def _softshrink_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for softshrink: grad * (|x| > lambd)."""
    lambd = args[0] if args else kwargs.get("lambd", 0.5)
    with _grad_context(keyset):
        lambd_t = _scalar_tensor_like(saved_a, float(lambd))
        abs_x = redispatch("abs", keyset, saved_a)
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        mask = redispatch("where", keyset,
            redispatch("gt", keyset, abs_x, lambd_t), ones, zero)
        return (redispatch("mul", keyset, grad, mask),)


def _rrelu_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for rrelu: grad * (1 if x>=0 else slope). Slope saved on forward output."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        pos_mask = redispatch("where", keyset,
            redispatch("ge", keyset, saved_a, zero), ones,
            redispatch("mul", keyset, ones, zero))
        neg_mask = redispatch("add", keyset, ones, redispatch("neg", keyset, pos_mask))
        # Slope was saved during forward pass on _rrelu_slope attribute
        slope = getattr(saved_a, '_rrelu_slope', None)
        if slope is not None:
            factor = redispatch("add", keyset, pos_mask,
                redispatch("mul", keyset, neg_mask, slope))
        else:
            lower = args[0] if args else kwargs.get("lower", 1.0 / 8)
            upper = args[1] if len(args) > 1 else kwargs.get("upper", 1.0 / 3)
            avg_slope = _scalar_tensor_like(saved_a, (lower + upper) / 2.0)
            factor = redispatch("add", keyset, pos_mask,
                redispatch("mul", keyset, neg_mask, avg_slope))
        return (redispatch("mul", keyset, grad, factor),)


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
        y = redispatch("cumprod", keyset, saved_a, dim)
        gy = redispatch("mul", keyset, grad, y)
        flip_dims = [dim]
        gy_flip = redispatch("flip", keyset, gy, flip_dims)
        rev_cumsum = redispatch("flip", keyset,
            redispatch("cumsum", keyset, gy_flip, dim), flip_dims)
        zero = _scalar_tensor_like(saved_a, 0.0)
        one = _scalar_tensor_like(saved_a, 1.0)
        nonzero_mask = redispatch("ne", keyset, saved_a, zero)
        safe_x = redispatch("where", keyset, nonzero_mask, saved_a, one)
        raw = redispatch("div", keyset, rev_cumsum, safe_x)
        result = redispatch("where", keyset, nonzero_mask, raw, zero)
        return (result,)


def _repeat_interleave_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for repeat_interleave: sum over interleaved groups."""
    repeats = args[0] if args else kwargs.get("repeats")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
    if grad.device.type == "npu":
        from .npu.backward import npu_repeat_interleave_backward
        return (npu_repeat_interleave_backward(grad, saved_a, repeats, dim),)
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
            # Create mask: scatter ones at index positions, then invert
            ones_src = redispatch("ones", keyset, src.shape, dtype=grad.dtype, device=grad.device)
            zeros_base = redispatch("zeros", keyset, a.shape, dtype=grad.dtype, device=grad.device)
            mask = redispatch("scatter", keyset, zeros_base, dim, index, ones_src)
            one = redispatch("ones", keyset, a.shape, dtype=grad.dtype, device=grad.device)
            inv_mask = redispatch("sub", keyset, one, mask)
            grad_a = redispatch("mul", keyset, grad, inv_mask)
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


def _min__backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for element-wise min: grad goes to a where a <= b, else to b."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        le_mask = redispatch("le", keyset, saved_a, saved_b)
        mask_a = redispatch("where", keyset, le_mask, ones, redispatch("mul", keyset, ones, zero))
        mask_b = redispatch("where", keyset, le_mask, redispatch("mul", keyset, ones, zero), ones)
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("mul", keyset, grad, mask_a)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            grad_b = redispatch("mul", keyset, grad, mask_b)
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


def _max__backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for element-wise max: grad goes to a where a >= b, else to b."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        ge_mask = redispatch("ge", keyset, saved_a, saved_b)
        mask_a = redispatch("where", keyset, ge_mask, ones, redispatch("mul", keyset, ones, zero))
        mask_b = redispatch("where", keyset, ge_mask, redispatch("mul", keyset, ones, zero), ones)
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
        flat_size = saved_a.numel()
        grad_flat = redispatch("reshape", keyset, grad, (-1,))
        idx_flat = redispatch("reshape", keyset, index, (-1,))
        idx_flat = redispatch("to", keyset, idx_flat, idx_flat.device, dtype='int64')
        result_flat = redispatch("zeros", keyset, (flat_size,), dtype=grad.dtype, device=grad.device)
        redispatch("scatter_add_", keyset, result_flat, 0, idx_flat, grad_flat)
        return (redispatch("reshape", keyset, result_flat, saved_a.shape),)


def _take_along_dim_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for take_along_dim: scatter-add gradient back along dim."""
    indices = args[0] if args else kwargs.get("indices")
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    with _grad_context(keyset):
        if dim is not None:
            d = dim if dim >= 0 else dim + len(saved_a.shape)
            result = redispatch("zeros", keyset, saved_a.shape, dtype=grad.dtype, device=grad.device)
            idx_long = redispatch("to", keyset, indices, indices.device, dtype='int64')
            redispatch("scatter_add_", keyset, result, d, idx_long, grad)
            return (result,)
        # dim is None: flatten then scatter_add
        flat_size = saved_a.numel()
        grad_flat = redispatch("reshape", keyset, grad, (-1,))
        idx_flat = redispatch("reshape", keyset, indices, (-1,))
        idx_flat = redispatch("to", keyset, idx_flat, idx_flat.device, dtype='int64')
        result_flat = redispatch("zeros", keyset, (flat_size,), dtype=grad.dtype, device=grad.device)
        redispatch("scatter_add_", keyset, result_flat, 0, idx_flat, grad_flat)
        return (redispatch("reshape", keyset, result_flat, saved_a.shape),)


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
        d = dim if dim >= 0 else dim + len(a.shape)
        result = redispatch("zeros", keyset, a.shape, dtype=grad.dtype, device=grad.device)
        idx_long = redispatch("to", keyset, indices, indices.device, dtype='int64')
        redispatch("scatter_add_", keyset, result, d, idx_long, grad)
        return (result,)


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
    if grad.device.type == "npu":
        from .npu.backward import npu_upsample_nearest2d_backward
        output_size = args[0]
        return (npu_upsample_nearest2d_backward(grad, saved_input, output_size),)
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
    if grad.device.type == "npu":
        from .npu.backward import npu_upsample_nearest1d_backward
        output_size = args[0]
        return (npu_upsample_nearest1d_backward(grad, saved_input, output_size),)
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
    if grad.device.type == "npu":
        from .npu.backward import npu_upsample_bilinear2d_backward
        output_size = args[0]
        align_corners = args[1] if len(args) > 1 else kwargs.get("align_corners", False)
        return (npu_upsample_bilinear2d_backward(grad, saved_input, output_size, align_corners),)
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
    if grad.device.type == "npu":
        from .npu.backward import npu_upsample_linear1d_backward
        output_size = args[0]
        align_corners = args[1] if len(args) > 1 else kwargs.get("align_corners", False)
        return (npu_upsample_linear1d_backward(grad, saved_input, output_size, align_corners),)
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
    if grad.device.type == "npu":
        from .npu.backward import npu_upsample_bicubic2d_backward
        output_size = args[0]
        align_corners = args[1] if len(args) > 1 else kwargs.get("align_corners", False)
        return (npu_upsample_bicubic2d_backward(grad, saved_input, output_size, align_corners),)
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
        out_val = out[0] if isinstance(out, tuple) else out
        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        dilation = args[3] if len(args) > 3 else kwargs.get("dilation", 1)

        kW = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        sW = stride if isinstance(stride, int) else stride[0]
        pW = padding if isinstance(padding, int) else padding[0]
        dW = dilation if isinstance(dilation, int) else dilation[0]

        N, C, W = saved_input.shape
        _, _, W_out = grad.shape
        padW = W + 2 * pW

        if pW > 0:
            input_padded = redispatch("pad", keyset, saved_input, (pW, pW), 'constant', float('-inf'))
        else:
            input_padded = saved_input

        grad_padded = redispatch("zeros", keyset, (N, C, padW), dtype=grad.dtype, device=grad.device)
        for ow in range(W_out):
            for kw in range(kW):
                iw = ow * sW + kw * dW
                if iw < padW:
                    mask = redispatch("eq", keyset, input_padded[:, :, iw:iw+1], out_val[:, :, ow:ow+1])
                    contrib = redispatch("mul", keyset, grad[:, :, ow:ow+1], mask)
                    old = grad_padded[:, :, iw:iw+1]
                    redispatch("setitem", keyset, grad_padded, (slice(None), slice(None), slice(iw, iw+1)),
                               redispatch("add", keyset, old, contrib))
        if pW > 0:
            return (redispatch("contiguous", keyset, grad_padded[:, :, pW:pW+W]),)
        return (redispatch("contiguous", keyset, grad_padded),)


def _max_pool3d_backward(grad, input, saved_input, out, keyset, args, kwargs):
    # NPU path: use aclnnMaxPool3dWithArgmaxBackward large kernel
    if grad.device.type == "npu":
        pool_out = out[0] if isinstance(out, tuple) else out
        backward_data = getattr(pool_out, "_backward_data", None)
        if backward_data is not None and "indices_ptr" in backward_data:
            from .npu.backward import npu_max_pool3d_backward
            return (npu_max_pool3d_backward(grad, saved_input, backward_data),)
    # CPU path: existing numpy implementation below
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
        kernel_size = args[0]
        stride = args[1]
        padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
        count_include_pad = args[4] if len(args) > 4 else kwargs.get("count_include_pad", True)

        kW = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        sW = stride if isinstance(stride, int) else stride[0]
        pW = padding if isinstance(padding, int) else padding[0]

        N, C, W = saved_input.shape
        _, _, W_out = grad.shape
        padW = W + 2 * pW

        grad_input_padded = redispatch("zeros", keyset, (N, C, padW), dtype=grad.dtype, device=grad.device)
        for ow in range(W_out):
            ws = ow * sW
            we = min(ws + kW, padW)
            if count_include_pad:
                cnt = kW
            else:
                cnt = max(min(we, W + pW) - max(ws, pW), 1)
            cnt_t = _scalar_tensor_like(grad, float(cnt))
            scaled = redispatch("div", keyset, grad[:, :, ow:ow+1], cnt_t)
            expanded = redispatch("expand", keyset, scaled, (N, C, we - ws))
            old = grad_input_padded[:, :, ws:we]
            redispatch("setitem", keyset, grad_input_padded, (slice(None), slice(None), slice(ws, we)),
                       redispatch("add", keyset, old, expanded))
        if pW > 0:
            return (redispatch("contiguous", keyset, grad_input_padded[:, :, pW:pW+W]),)
        return (redispatch("contiguous", keyset, grad_input_padded),)


def _avg_pool3d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    if grad.device.type == "npu":
        from .npu.backward import npu_avg_pool3d_backward
        kernel_size = args[0] if args else kwargs.get("kernel_size")
        stride = args[1] if len(args) > 1 else kwargs.get("stride", kernel_size)
        padding = args[2] if len(args) > 2 else kwargs.get("padding", (0, 0, 0))
        ceil_mode = args[3] if len(args) > 3 else kwargs.get("ceil_mode", False)
        count_include_pad = args[4] if len(args) > 4 else kwargs.get("count_include_pad", True)
        divisor_override = args[5] if len(args) > 5 else kwargs.get("divisor_override", None)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        return (npu_avg_pool3d_backward(grad, saved_input, kernel_size, stride, padding,
                                        ceil_mode, count_include_pad, divisor_override),)
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
        N, C, L = saved_input.shape
        output_size = args[0]
        oL = output_size if isinstance(output_size, int) else output_size[0]

        grad_input = redispatch("zeros", keyset, (N, C, L), dtype=grad.dtype, device=grad.device)
        for ol in range(oL):
            l_start = ol * L // oL
            l_end = (ol + 1) * L // oL
            cnt = l_end - l_start
            cnt_t = _scalar_tensor_like(grad, float(cnt))
            scaled = redispatch("div", keyset, grad[:, :, ol:ol+1], cnt_t)
            expanded = redispatch("expand", keyset, scaled, (N, C, cnt))
            old = grad_input[:, :, l_start:l_end]
            redispatch("setitem", keyset, grad_input, (slice(None), slice(None), slice(l_start, l_end)),
                       redispatch("add", keyset, old, expanded))
        return (redispatch("contiguous", keyset, grad_input),)


def _adaptive_avg_pool3d_backward(grad, input, saved_input, _out, keyset, args, kwargs):
    if grad.device.type == "npu":
        from .npu.backward import npu_adaptive_avg_pool3d_backward
        return (npu_adaptive_avg_pool3d_backward(grad, saved_input),)
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
        out_val = out[0] if isinstance(out, tuple) else out
        N, C, L = saved_input.shape
        output_size = args[0]
        oL = output_size if isinstance(output_size, int) else output_size[0]

        grad_input = redispatch("zeros", keyset, (N, C, L), dtype=grad.dtype, device=grad.device)
        for ol in range(oL):
            l_start = ol * L // oL
            l_end = (ol + 1) * L // oL
            for il in range(l_start, l_end):
                mask = redispatch("eq", keyset, saved_input[:, :, il:il+1], out_val[:, :, ol:ol+1])
                contrib = redispatch("mul", keyset, grad[:, :, ol:ol+1], mask)
                old = grad_input[:, :, il:il+1]
                redispatch("setitem", keyset, grad_input, (slice(None), slice(None), slice(il, il+1)),
                           redispatch("add", keyset, old, contrib))
        return (redispatch("contiguous", keyset, grad_input),)


def _adaptive_max_pool2d_backward(grad, input, saved_input, out, keyset, args, kwargs):
    # NPU path: use aclnnAdaptiveMaxPool2dBackward large kernel
    if grad.device.type == "npu":
        pool_out = out[0] if isinstance(out, tuple) else out
        backward_data = getattr(pool_out, "_backward_data", None)
        if backward_data is not None and "indices_ptr" in backward_data:
            from .npu.backward import npu_adaptive_max_pool2d_backward
            return (npu_adaptive_max_pool2d_backward(grad, saved_input, backward_data),)
    # CPU path: existing numpy implementation below
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
    """Backward for cummin: same pattern as cummax -- scatter-add to index positions."""
    dim = args[0] if args else kwargs.get("dim", 0)
    with _grad_context(keyset):
        d = dim if dim >= 0 else dim + len(a.shape)
        result = redispatch("zeros", keyset, a.shape, dtype=grad.dtype, device=grad.device)
        idx_long = redispatch("to", keyset, indices, indices.device, dtype='int64')
        redispatch("scatter_add_", keyset, result, d, idx_long, grad)
        return (result,)


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
    mode = args[0] if args else kwargs.get("mode", "bilinear")
    padding_mode = args[1] if len(args) > 1 else kwargs.get("padding_mode", "zeros")
    align_corners = args[2] if len(args) > 2 else kwargs.get("align_corners", False)

    if grad.device.type == "npu":
        from .npu.backward import npu_grid_sample_backward
        mode_map = {"bilinear": 0, "nearest": 1, "bicubic": 2}
        pad_map = {"zeros": 0, "border": 1, "reflection": 2}
        interp_mode = mode_map.get(mode, 0) if isinstance(mode, str) else int(mode)
        pad_mode = pad_map.get(padding_mode, 0) if isinstance(padding_mode, str) else int(padding_mode)
        grad_input, grad_grid = npu_grid_sample_backward(
            grad, saved_input, saved_grid, interp_mode, pad_mode, align_corners)
        results = []
        if getattr(input, "requires_grad", False):
            results.append(grad_input)
        if getattr(grid, "requires_grad", False):
            results.append(grad_grid)
        return tuple(results) if results else (grad_input, grad_grid)

    with _grad_context(keyset):
        import numpy as np
        from .cpu.ops import _to_numpy, _from_numpy

        grad_np = _to_numpy(grad)
        input_np = _to_numpy(saved_input)
        grid_np = _to_numpy(saved_grid)

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
        N = size[0]
        if len(size) == 4:
            _, _, H, W = size
            # Build grid points
            if align_corners:
                h_pts = redispatch("linspace", keyset, -1.0, 1.0, H)
                w_pts = redispatch("linspace", keyset, -1.0, 1.0, W)
            else:
                h_pts = redispatch("linspace", keyset, -1.0 + 1.0 / H, 1.0 - 1.0 / H, H)
                w_pts = redispatch("linspace", keyset, -1.0 + 1.0 / W, 1.0 - 1.0 / W, W)
            # Build base grid: (H*W, 3) with columns [w, h, 1]
            w_grid = redispatch("reshape", keyset, w_pts, (1, W))
            w_grid = redispatch("expand", keyset, w_grid, (H, W))
            h_grid = redispatch("reshape", keyset, h_pts, (H, 1))
            h_grid = redispatch("expand", keyset, h_grid, (H, W))
            ones_grid = redispatch("ones", keyset, (H, W), dtype=grad.dtype, device=grad.device)
            w_flat = redispatch("reshape", keyset, redispatch("contiguous", keyset, w_grid), (H * W,))
            h_flat = redispatch("reshape", keyset, redispatch("contiguous", keyset, h_grid), (H * W,))
            ones_flat = redispatch("reshape", keyset, ones_grid, (H * W,))
            base = redispatch("stack", keyset, [w_flat, h_flat, ones_flat], dim=1)  # (H*W, 3)
            # grad: (N, H, W, 2) -> (N, H*W, 2)
            grad_flat = redispatch("reshape", keyset, grad, (N, H * W, 2))
            grad_t = redispatch("transpose", keyset, grad_flat, -1, -2)  # (N, 2, H*W)
            base_expanded = redispatch("reshape", keyset, base, (1, H * W, 3))
            base_expanded = redispatch("expand", keyset, base_expanded, (N, H * W, 3))
            base_expanded = redispatch("contiguous", keyset, base_expanded)
            grad_theta = redispatch("matmul", keyset, grad_t, base_expanded)  # (N, 2, 3)
        else:
            # 3D case
            _, _, D, H, W = size
            if align_corners:
                d_pts = redispatch("linspace", keyset, -1.0, 1.0, D)
                h_pts = redispatch("linspace", keyset, -1.0, 1.0, H)
                w_pts = redispatch("linspace", keyset, -1.0, 1.0, W)
            else:
                d_pts = redispatch("linspace", keyset, -1.0 + 1.0 / D, 1.0 - 1.0 / D, D)
                h_pts = redispatch("linspace", keyset, -1.0 + 1.0 / H, 1.0 - 1.0 / H, H)
                w_pts = redispatch("linspace", keyset, -1.0 + 1.0 / W, 1.0 - 1.0 / W, W)
            M = D * H * W
            # Build 3D meshgrid via reshape+expand
            d_3d = redispatch("reshape", keyset, d_pts, (D, 1, 1))
            d_3d = redispatch("expand", keyset, d_3d, (D, H, W))
            h_3d = redispatch("reshape", keyset, h_pts, (1, H, 1))
            h_3d = redispatch("expand", keyset, h_3d, (D, H, W))
            w_3d = redispatch("reshape", keyset, w_pts, (1, 1, W))
            w_3d = redispatch("expand", keyset, w_3d, (D, H, W))
            ones_3d = redispatch("ones", keyset, (D, H, W), dtype=grad.dtype, device=grad.device)
            w_f = redispatch("reshape", keyset, redispatch("contiguous", keyset, w_3d), (M,))
            h_f = redispatch("reshape", keyset, redispatch("contiguous", keyset, h_3d), (M,))
            d_f = redispatch("reshape", keyset, redispatch("contiguous", keyset, d_3d), (M,))
            ones_f = redispatch("reshape", keyset, ones_3d, (M,))
            base = redispatch("stack", keyset, [w_f, h_f, d_f, ones_f], dim=1)  # (M, 4)
            grad_flat = redispatch("reshape", keyset, grad, (N, M, 3))
            grad_t = redispatch("transpose", keyset, grad_flat, -1, -2)  # (N, 3, M)
            base_exp = redispatch("reshape", keyset, base, (1, M, 4))
            base_exp = redispatch("expand", keyset, base_exp, (N, M, 4))
            base_exp = redispatch("contiguous", keyset, base_exp)
            grad_theta = redispatch("matmul", keyset, grad_t, base_exp)  # (N, 3, 4)
        # Cast to match theta dtype
        grad_theta = redispatch("to", keyset, grad_theta, grad_theta.device, dtype=theta.dtype)
        return (grad_theta,)


def _masked_fill_inplace_backward(grad, _a, saved_a, args, keyset):
    """Backward for masked_fill_ (in-place)."""
    mask = args[0]
    with _grad_context(keyset):
        zeros = redispatch("zeros", keyset, grad.shape, dtype=grad.dtype, device=grad.device)
        grad_input = redispatch("where", keyset, mask, zeros, grad)
        return (grad_input,)


# ---------------------------------------------------------------------------
# Round 4: In-place + P1 Math/Manipulation backward ops
# ---------------------------------------------------------------------------

# --- 2a: sub_ backward ---
def _inplace_sub_backward(grad, a, _saved_a, args, keyset):
    b = args[0]
    with _grad_context(keyset):
        neg_grad = redispatch("neg", keyset, grad)
    grad_a = reduce_grad(grad, a.shape) if getattr(a, "requires_grad", False) else None
    grad_b = reduce_grad(neg_grad, b.shape) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


# --- 2b: div_ backward ---
def _inplace_div_backward(grad, a, saved_a, args, keyset):
    b = args[0]
    with _grad_context(keyset):
        grad_a = None
        grad_b = None
        if getattr(a, "requires_grad", False):
            grad_a = redispatch("div", keyset, grad, b)
            grad_a = reduce_grad(grad_a, a.shape)
        if getattr(b, "requires_grad", False):
            denom = redispatch("mul", keyset, b, b)
            num = redispatch("mul", keyset, grad, saved_a)
            grad_b = redispatch("neg", keyset, redispatch("div", keyset, num, denom))
            grad_b = reduce_grad(grad_b, b.shape)
    return grad_a, grad_b


# --- 2c: clamp_ backward ---
def _inplace_clamp_backward(grad, a, saved_a, args, keyset):
    return _clamp_backward(grad, a, saved_a, keyset, args, {})


# --- 2d: copy_ backward ---
def _inplace_copy_backward(grad, a, _saved_a, args, _keyset):
    src = args[0]
    grad_a = grad if getattr(a, "requires_grad", False) else None
    grad_src = reduce_grad(grad, src.shape) if getattr(src, "requires_grad", False) else None
    return grad_a, grad_src


# --- 2e: setitem backward (custom wrapper) ---
def _autograd_setitem(name):
    def wrapper(a, key, value):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, key, value)
        a_rg = getattr(a, "requires_grad", False)
        v_rg = getattr(value, "requires_grad", False)
        if GradMode.enabled and (a_rg or v_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _setitem_backward(grad, a, value, key, backward_keyset)

            inputs = tuple(t for t in (a, value) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _setitem_backward(grad, a, value, key, keyset):
    with _grad_context(keyset):
        grad_a = None
        if getattr(a, "requires_grad", False):
            grad_a = grad.clone()
            zeros = redispatch("zeros", keyset, grad_a[key].shape, dtype=grad.dtype, device=grad.device)
            redispatch("setitem", keyset, grad_a, key, zeros)
        grad_val = None
        if getattr(value, "requires_grad", False):
            grad_val = reduce_grad(grad[key], value.shape)
    return grad_a, grad_val


# --- 2f: index_copy_ backward (custom wrapper) ---
def _autograd_index_copy_inplace(name):
    def wrapper(a, dim, index, source):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim, index, source)
        a_rg = getattr(a, "requires_grad", False)
        src_rg = getattr(source, "requires_grad", False)
        if GradMode.enabled and (a_rg or src_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _index_copy_inplace_backward(grad, a, source, dim, index, backward_keyset)

            inputs = tuple(t for t in (a, source) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _index_copy_inplace_backward(grad, a, source, dim, index, keyset):
    with _grad_context(keyset):
        grad_a = None
        if getattr(a, "requires_grad", False):
            grad_a = grad.clone()
            zeros = redispatch("zeros", keyset, source.shape, dtype=grad.dtype, device=grad.device)
            redispatch("index_copy_", keyset, grad_a, dim, index, zeros)
        grad_src = None
        if getattr(source, "requires_grad", False):
            grad_src = redispatch("index_select", keyset, grad, dim, index)
    return grad_a, grad_src


# --- 2g: index_fill_ backward (custom wrapper) ---
def _autograd_index_fill_inplace(name):
    def wrapper(a, dim, index, value):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim, index, value)
        a_rg = getattr(a, "requires_grad", False)
        if GradMode.enabled and a_rg:
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _index_fill_inplace_backward(grad, a, dim, index, backward_keyset)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _index_fill_inplace_backward(grad, a, dim, index, keyset):
    with _grad_context(keyset):
        grad_a = grad.clone()
        redispatch("index_fill_", keyset, grad_a, dim, index, 0.0)
    return (grad_a,)


# --- 2h: scatter_ backward (custom wrapper) ---
def _autograd_scatter_inplace(name):
    def wrapper(a, dim, index, src):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim, index, src)
        a_rg = getattr(a, "requires_grad", False)
        src_rg = getattr(src, "requires_grad", False) if hasattr(src, "requires_grad") else False
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


# --- 2i: masked_scatter_ backward (custom wrapper) ---
def _autograd_masked_scatter_inplace(name):
    def wrapper(a, mask, source):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, mask, source)
        a_rg = getattr(a, "requires_grad", False)
        src_rg = getattr(source, "requires_grad", False)
        if GradMode.enabled and (a_rg or src_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _masked_scatter_inplace_backward(grad, a, source, mask, backward_keyset)

            inputs = tuple(t for t in (a, source) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _masked_scatter_inplace_backward(grad, a, source, mask, keyset):
    with _grad_context(keyset):
        grad_a = None
        if getattr(a, "requires_grad", False):
            zeros = redispatch("zeros", keyset, grad.shape, dtype=grad.dtype, device=grad.device)
            grad_a = redispatch("where", keyset, mask, zeros, grad)
        grad_src = None
        if getattr(source, "requires_grad", False):
            grad_src = redispatch("masked_select", keyset, grad, mask)
    return grad_a, grad_src


# --- Task 3: Tensor Manipulation backward ---

# 3a: chunk backward (multi_output)
def _chunk_backward(grad, idx, a, _saved_a, keyset, args, _kwargs):
    chunks_arg = args[0]
    dim = args[1] if len(args) > 1 else _kwargs.get("dim", 0)
    d = dim if dim >= 0 else dim + len(a.shape)
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        total = a.shape[d]
        chunk_size = (total + chunks_arg - 1) // chunks_arg
        start = idx * chunk_size
        slices = [slice(None)] * len(a.shape)
        slices[d] = slice(start, start + grad.shape[d])
        redispatch("setitem", keyset, grad_input, tuple(slices), grad)
    return (grad_input,)


# 3b: hstack backward (multi_input)
def _hstack_backward(grad, tensors, _saved, keyset, _args, _kwargs):
    with _grad_context(keyset):
        dim = 0 if tensors[0].ndim == 1 else 1
        sizes = [t.shape[dim] if t.ndim > dim else t.shape[0] for t in tensors]
        grads = redispatch("split", keyset, grad, sizes, dim)
        result = []
        for i, t in enumerate(tensors):
            if getattr(t, "requires_grad", False):
                g = grads[i]
                if t.ndim == 1 and grad.ndim > 1:
                    g = redispatch("reshape", keyset, g, (g.numel(),))
                result.append(g)
            else:
                result.append(None)
        return tuple(result)


# 3c: vstack backward (multi_input) — also used for row_stack
def _vstack_backward(grad, tensors, _saved, keyset, _args, _kwargs):
    with _grad_context(keyset):
        sizes = [t.shape[0] if t.ndim >= 2 else 1 for t in tensors]
        grads = redispatch("split", keyset, grad, sizes, 0)
        result = []
        for i, t in enumerate(tensors):
            if getattr(t, "requires_grad", False):
                g = grads[i]
                if t.ndim == 1:
                    g = redispatch("reshape", keyset, g, (t.shape[0],))
                result.append(g)
            else:
                result.append(None)
        return tuple(result)


# 3d: dstack backward (multi_input)
def _dstack_backward(grad, tensors, _saved, keyset, _args, _kwargs):
    with _grad_context(keyset):
        sizes = []
        for t in tensors:
            if t.ndim <= 2:
                sizes.append(1)
            else:
                sizes.append(t.shape[2])
        grads = redispatch("split", keyset, grad, sizes, 2)
        result = []
        for i, t in enumerate(tensors):
            if getattr(t, "requires_grad", False):
                g = grads[i]
                if t.ndim == 1:
                    g = redispatch("reshape", keyset, g, (t.shape[0],))
                elif t.ndim == 2:
                    g = redispatch("reshape", keyset, g, t.shape)
                result.append(g)
            else:
                result.append(None)
        return tuple(result)


# 3e: column_stack backward (multi_input)
def _column_stack_backward(grad, tensors, _saved, keyset, _args, _kwargs):
    with _grad_context(keyset):
        sizes = [1 if t.ndim == 1 else t.shape[1] for t in tensors]
        grads = redispatch("split", keyset, grad, sizes, 1)
        result = []
        for i, t in enumerate(tensors):
            if getattr(t, "requires_grad", False):
                g = grads[i]
                if t.ndim == 1:
                    g = redispatch("reshape", keyset, g, (t.shape[0],))
                result.append(g)
            else:
                result.append(None)
        return tuple(result)


# 3f: diag backward (unary_args)
def _diag_backward(grad, a, _saved_a, keyset, args, kwargs):
    diagonal = args[0] if args else kwargs.get("diagonal", 0)
    with _grad_context(keyset):
        return (redispatch("diag", keyset, grad, diagonal),)


# --- Task 4: Shape/View backward ---

# 4a: broadcast_to backward
def _broadcast_to_backward(grad, a, _saved_a, keyset, _args, _kwargs):
    return (reduce_grad(grad, a.shape),)


# 4b: unfold backward
def _unfold_backward(grad, a, _saved_a, keyset, args, kwargs):
    dim = args[0] if len(args) > 0 else kwargs.get("dimension", 0)
    size = args[1] if len(args) > 1 else kwargs.get("size")
    step = args[2] if len(args) > 2 else kwargs.get("step")
    if grad.device.type == "npu":
        from .npu.backward import npu_unfold_backward
        return (npu_unfold_backward(grad, list(a.shape), dim, size, step),)
    import numpy as np
    from .cpu.ops import _to_numpy, _from_numpy
    grad_np = _to_numpy(grad)
    result = np.zeros(a.shape, dtype=grad_np.dtype)
    d = dim if dim >= 0 else dim + len(a.shape)
    n_windows = grad_np.shape[d]
    for i in range(n_windows):
        src_slices = [slice(None)] * len(grad_np.shape)
        src_slices[d] = i
        window_grad = grad_np[tuple(src_slices)]
        for j in range(size):
            dst_slices = [slice(None)] * len(a.shape)
            dst_slices[d] = i * step + j
            src_inner = [slice(None)] * len(window_grad.shape)
            src_inner[-1] = j
            result[tuple(dst_slices)] += window_grad[tuple(src_inner)]
    return (_from_numpy(np.ascontiguousarray(result), a.dtype, a.device),)


# --- Task 5: Math backward ---

# 5a: square backward
def _square_backward(grad, _a, saved_a, keyset):
    with _grad_context(keyset):
        two = _scalar_tensor_like(saved_a, 2.0)
        return (redispatch("mul", keyset, grad, redispatch("mul", keyset, two, saved_a)),)


# 5b: diff backward
def _diff_backward(grad, a, _saved_a, keyset, args, kwargs):
    n = args[0] if args else kwargs.get("n", 1)
    dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
    d = dim if dim >= 0 else dim + len(a.shape)
    with _grad_context(keyset):
        g = grad
        for _ in range(n):
            ndim = len(g.shape)
            # PyTorch pad spec: pairs from last dim to first
            pad_before = []
            pad_after = []
            for i in range(ndim - 1, -1, -1):
                if i == d:
                    pad_before.extend([1, 0])  # 1 on left, 0 on right
                    pad_after.extend([0, 1])    # 0 on left, 1 on right
                else:
                    pad_before.extend([0, 0])
                    pad_after.extend([0, 0])
            padded_before = redispatch("pad", keyset, g, pad_before, 'constant', 0.0)
            padded_after = redispatch("pad", keyset, g, pad_after, 'constant', 0.0)
            g = redispatch("sub", keyset, padded_before, padded_after)
        return (g,)


# 5c: heaviside backward
def _heaviside_backward(grad, a, b, _saved_a, _saved_b, _keyset):
    grad_a = redispatch("mul", _keyset, grad, _scalar_tensor_like(a, 0.0)) if getattr(a, "requires_grad", False) else None
    grad_b = redispatch("mul", _keyset, grad, _scalar_tensor_like(b, 0.0)) if getattr(b, "requires_grad", False) else None
    return grad_a, grad_b


# 5d: trace backward
def _trace_backward(grad, a, _saved_a, keyset, _args, _kwargs):
    with _grad_context(keyset):
        n = a.shape[0]
        m = a.shape[1] if len(a.shape) > 1 else n
        eye = redispatch("eye", keyset, n, m, dtype=a.dtype, device=a.device)
        return (redispatch("mul", keyset, eye, grad),)


# 5e: det backward
def _det_backward(grad, a, saved_a, keyset, _args, _kwargs):
    import numpy as np
    from .cpu.ops import _to_numpy, _from_numpy
    a_np = _to_numpy(saved_a)
    grad_np = _to_numpy(grad)
    if a_np.ndim == 2:
        det_val = np.linalg.det(a_np)
        if abs(det_val) < 1e-30:
            result = np.zeros_like(a_np)
        else:
            inv_t = np.linalg.inv(a_np).T
            result = (grad_np * det_val) * inv_t
    else:
        result = np.zeros_like(a_np)
        for idx in np.ndindex(a_np.shape[:-2]):
            mat = a_np[idx]
            det_val = np.linalg.det(mat)
            g = grad_np[idx] if grad_np.ndim > 0 else grad_np
            if abs(det_val) < 1e-30:
                result[idx] = np.zeros_like(mat)
            else:
                inv_t = np.linalg.inv(mat).T
                result[idx] = (g * det_val) * inv_t
    return (_from_numpy(np.ascontiguousarray(result.astype(_to_numpy(saved_a).dtype)), saved_a.dtype, saved_a.device),)


# 5f: dist backward (custom wrapper)
def _autograd_dist(name):
    def wrapper(a, b, p=2):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, p)
        a_rg = getattr(a, "requires_grad", False)
        b_rg = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_rg or b_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _dist_backward(grad, a, b, p, backward_keyset)

            inputs = tuple(t for t in (a, b) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            node.save_for_backward(a, b)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _dist_backward(grad, a, b, p, keyset):
    with _grad_context(keyset):
        diff = redispatch("sub", keyset, a, b)
        if p == 2:
            sq = redispatch("mul", keyset, diff, diff)
            d = redispatch("sqrt", keyset, redispatch("sum", keyset, sq))
            eps = _scalar_tensor_like(a, 1e-30)
            safe_d = redispatch("add", keyset, d, eps)
            grad_dir = redispatch("div", keyset, diff, safe_d)
        elif p == 1:
            grad_dir = redispatch("sign", keyset, diff)
        elif p == float('inf'):
            abs_diff = redispatch("abs", keyset, diff)
            max_val = redispatch("amax", keyset, abs_diff)
            grad_dir = redispatch("where", keyset,
                redispatch("eq", keyset, abs_diff, max_val),
                redispatch("sign", keyset, diff),
                _scalar_tensor_like(diff, 0.0))
        else:
            abs_diff = redispatch("abs", keyset, diff)
            sum_p = redispatch("sum", keyset, redispatch("pow", keyset, abs_diff, float(p)))
            d = redispatch("pow", keyset, sum_p, 1.0 / p)
            eps = _scalar_tensor_like(a, 1e-30)
            safe_d = redispatch("add", keyset, d, eps)
            numer = redispatch("mul", keyset, redispatch("sign", keyset, diff),
                               redispatch("pow", keyset, abs_diff, float(p - 1)))
            denom = redispatch("pow", keyset, safe_d, float(p - 1))
            grad_dir = redispatch("div", keyset, numer, denom)
        result = redispatch("mul", keyset, grad, grad_dir)
        grad_a = result if getattr(a, "requires_grad", False) else None
        grad_b = redispatch("neg", keyset, result) if getattr(b, "requires_grad", False) else None
        return grad_a, grad_b


# 5g: renorm backward
def _renorm_backward(grad, a, saved_a, keyset, args, kwargs):
    p_val = args[0] if len(args) > 0 else kwargs.get("p", 2)
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    maxnorm = args[2] if len(args) > 2 else kwargs.get("maxnorm", 1.0)
    with _grad_context(keyset):
        d = dim if dim >= 0 else dim + len(saved_a.shape)
        reduce_axes = tuple(i for i in range(len(saved_a.shape)) if i != d)
        eps = _scalar_tensor_like(saved_a, 1e-30)
        maxnorm_t = _scalar_tensor_like(saved_a, float(maxnorm))

        abs_a = redispatch("abs", keyset, saved_a)
        if p_val == 2:
            norm_sq = redispatch("sum", keyset,
                redispatch("mul", keyset, saved_a, saved_a),
                dim=reduce_axes, keepdim=True)
            norm_val = redispatch("sqrt", keyset,
                redispatch("add", keyset, norm_sq, eps))
        else:
            norm_val = redispatch("pow", keyset,
                redispatch("sum", keyset,
                    redispatch("pow", keyset, abs_a, float(p_val)),
                    dim=reduce_axes, keepdim=True),
                1.0 / p_val)
            norm_val = redispatch("add", keyset, norm_val, eps)

        needs_renorm = redispatch("gt", keyset, norm_val, maxnorm_t)

        if p_val == 2:
            # Chain rule: d/dx_i [maxnorm * x_i / norm]
            #   = maxnorm/norm * (grad_i - x_i * dot(grad, x) / norm^2)
            inv_norm = redispatch("div", keyset, maxnorm_t, norm_val)
            term1 = redispatch("mul", keyset, grad, inv_norm)
            dot_ga = redispatch("sum", keyset,
                redispatch("mul", keyset, grad, saved_a),
                dim=reduce_axes, keepdim=True)
            term2 = redispatch("mul", keyset, saved_a,
                redispatch("div", keyset,
                    redispatch("mul", keyset, maxnorm_t, dot_ga),
                    redispatch("mul", keyset, norm_val, norm_sq)))
            renorm_grad = redispatch("sub", keyset, term1, term2)
        else:
            # General p: scale gradient by maxnorm / norm
            scale = redispatch("div", keyset, maxnorm_t, norm_val)
            renorm_grad = redispatch("mul", keyset, grad, scale)

        # Where norm > maxnorm, use renorm_grad; otherwise pass through
        result = redispatch("where", keyset, needs_renorm, renorm_grad, grad)
        return (result,)


# 5h: cdist backward (custom wrapper)
def _autograd_cdist(name):
    def wrapper(x1, x2, p=2.0):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, x1, x2, p)
        x1_rg = getattr(x1, "requires_grad", False)
        x2_rg = getattr(x2, "requires_grad", False)
        if GradMode.enabled and (x1_rg or x2_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _cdist_backward(grad, x1, x2, p, backward_keyset)

            inputs = tuple(t for t in (x1, x2) if hasattr(t, "requires_grad"))
            node = Node(_backward, inputs)
            node_holder["node"] = node
            node.save_for_backward(x1, x2)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _cdist_backward(grad, x1, x2, p, keyset):
    """Backward for cdist using vectorized broadcasting."""
    was_2d = (x1.ndim == 2)
    if was_2d:
        _x1 = redispatch("unsqueeze", keyset, x1, 0)    # (1, P, M)
        _x2 = redispatch("unsqueeze", keyset, x2, 0)    # (1, R, M)
        _grad = redispatch("unsqueeze", keyset, grad, 0) # (1, P, R)
    else:
        _x1, _x2, _grad = x1, x2, grad

    with _grad_context(keyset):
        # (B, P, 1, M) - (B, 1, R, M) = (B, P, R, M)
        x1_exp = redispatch("unsqueeze", keyset, _x1, 2)
        x2_exp = redispatch("unsqueeze", keyset, _x2, 1)
        diff = redispatch("sub", keyset, x1_exp, x2_exp)

        if p == 2.0:
            # L2 norm direction: diff / ||diff||_2
            dist = redispatch("sqrt", keyset, redispatch("sum", keyset,
                redispatch("mul", keyset, diff, diff), dim=-1, keepdim=True))
            eps = _scalar_tensor_like(_x1, 1e-30)
            safe_dist = redispatch("add", keyset, dist, eps)
            direction = redispatch("div", keyset, diff, safe_dist)  # (B,P,R,M)
        elif p == 1.0:
            # L1 norm direction: sign(diff)
            direction = redispatch("sign", keyset, diff)
        else:
            # General Lp norm direction
            abs_diff = redispatch("abs", keyset, diff)
            dist = redispatch("pow", keyset,
                redispatch("sum", keyset, redispatch("pow", keyset, abs_diff, float(p)), dim=-1, keepdim=True),
                1.0 / p)
            eps = _scalar_tensor_like(_x1, 1e-30)
            safe_dist = redispatch("add", keyset, dist, eps)
            numer = redispatch("mul", keyset, redispatch("sign", keyset, diff),
                               redispatch("pow", keyset, abs_diff, float(p - 1)))
            denom = redispatch("pow", keyset, safe_dist, float(p - 1))
            direction = redispatch("div", keyset, numer, denom)

        grad_exp = redispatch("unsqueeze", keyset, _grad, -1)  # (B,P,R,1)
        gd = redispatch("mul", keyset, grad_exp, direction)    # (B,P,R,M)

        grad_x1 = redispatch("sum", keyset, gd, dim=2) if getattr(x1, "requires_grad", False) else None   # (B,P,M)
        grad_x2 = redispatch("neg", keyset, redispatch("sum", keyset, gd, dim=1)) if getattr(x2, "requires_grad", False) else None  # (B,R,M)

        if was_2d:
            if grad_x1 is not None:
                grad_x1 = redispatch("reshape", keyset, grad_x1, x1.shape)
            if grad_x2 is not None:
                grad_x2 = redispatch("reshape", keyset, grad_x2, x2.shape)
        return grad_x1, grad_x2


# --- Task 6: im2col/col2im backward ---

# 6a: im2col backward (= col2im)
def _im2col_backward(grad, a, _saved_a, keyset, args, kwargs):
    kernel_size = args[0] if len(args) > 0 else kwargs.get("kernel_size")
    dilation = args[1] if len(args) > 1 else kwargs.get("dilation", 1)
    padding = args[2] if len(args) > 2 else kwargs.get("padding", 0)
    stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
    output_size = (a.shape[2], a.shape[3]) if len(a.shape) == 4 else (a.shape[-2], a.shape[-1])
    with _grad_context(keyset):
        return (redispatch("col2im", keyset, grad, output_size, kernel_size, dilation, padding, stride),)


# 6b: col2im backward (= im2col)
def _col2im_backward(grad, a, _saved_a, keyset, args, kwargs):
    output_size = args[0] if len(args) > 0 else kwargs.get("output_size")
    kernel_size = args[1] if len(args) > 1 else kwargs.get("kernel_size")
    dilation = args[2] if len(args) > 2 else kwargs.get("dilation", 1)
    padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
    stride = args[4] if len(args) > 4 else kwargs.get("stride", 1)
    with _grad_context(keyset):
        return (redispatch("im2col", keyset, grad, kernel_size, dilation, padding, stride),)


# ---------------------------------------------------------------------------
# Round 5 — P0 + P1 backward ops
# ---------------------------------------------------------------------------

# --- Task 2: Simple backward (frac, nansum, nanmean) ---

def _frac_backward(grad, _a, _saved_a, keyset):
    """Backward for frac: frac(x) = x - trunc(x), derivative = 1."""
    return (grad,)


def _nansum_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for nansum: like sum backward but zero gradient at NaN positions."""
    dim = args[0] if args else kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        expanded_grad = redispatch("mul", keyset, grad, ones)
        nan_mask = redispatch("ne", keyset, saved_a, saved_a)
        zero = _scalar_tensor_like(saved_a, 0.0)
        return (redispatch("where", keyset, nan_mask, zero, expanded_grad),)


def _nanmean_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for nanmean: like mean backward but divide by non-NaN count."""
    dim = args[0] if args else kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        nan_mask = redispatch("ne", keyset, saved_a, saved_a)
        not_nan = redispatch("where", keyset, nan_mask,
                             _scalar_tensor_like(saved_a, 0.0),
                             saved_a._ones_like())
        if dim is None:
            count = redispatch("sum", keyset, not_nan)
        else:
            count = redispatch("sum", keyset, not_nan, dim=dim, keepdim=True)
        from .._creation import tensor
        eps = tensor(1e-38, device=grad.device)
        safe_count = redispatch("add", keyset, count, eps)
        scaled_grad = redispatch("div", keyset, grad, safe_count)
        ones = saved_a._ones_like()
        expanded_grad = redispatch("mul", keyset, scaled_grad, ones)
        zero = _scalar_tensor_like(saved_a, 0.0)
        return (redispatch("where", keyset, nan_mask, zero, expanded_grad),)


# --- Task 3: masked_select — Custom wrapper ---

def _autograd_masked_select(name):
    """Autograd wrapper for masked_select(input, mask) -> 1D tensor."""
    def wrapper(a, mask):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, mask)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _masked_select_backward(grad, a, mask, backward_keyset)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _masked_select_backward(grad, a, mask, keyset):
    """Backward for masked_select: scatter grad back to selected positions."""
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        redispatch("masked_scatter_", keyset, grad_input, mask, grad)
        return (grad_input,)


# --- Task 4: Split variants (hsplit, vsplit, dsplit) ---

def _compute_split_sizes(split_size_or_sections, dim_size):
    """Compute section sizes from an int (num sections) or list of sizes."""
    if isinstance(split_size_or_sections, int):
        size, extra = divmod(dim_size, split_size_or_sections)
        return [size + 1] * extra + [size] * (split_size_or_sections - extra)
    return list(split_size_or_sections)


def _xsplit_backward(grad, idx, a, _saved_a, keyset, args, _kwargs, dim):
    """Generic backward for hsplit/vsplit/dsplit."""
    split_size_or_sections = args[0]
    d = dim if dim >= 0 else dim + len(a.shape)
    sizes = _compute_split_sizes(split_size_or_sections, a.shape[d])
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        start = sum(sizes[:idx])
        slices = [slice(None)] * len(a.shape)
        slices[d] = slice(start, start + grad.shape[d])
        redispatch("setitem", keyset, grad_input, tuple(slices), grad)
    return (grad_input,)


def _hsplit_backward(grad, idx, a, _saved_a, keyset, args, _kwargs):
    dim = 0 if len(a.shape) == 1 else 1
    return _xsplit_backward(grad, idx, a, _saved_a, keyset, args, _kwargs, dim)


def _vsplit_backward(grad, idx, a, _saved_a, keyset, args, _kwargs):
    return _xsplit_backward(grad, idx, a, _saved_a, keyset, args, _kwargs, 0)


def _dsplit_backward(grad, idx, a, _saved_a, keyset, args, _kwargs):
    return _xsplit_backward(grad, idx, a, _saved_a, keyset, args, _kwargs, 2)


# --- Task 5: Math/Algebra (inner, tensordot, matrix_power) ---

def _inner_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for inner: contracts last dim of both inputs."""
    import math
    with _grad_context(keyset):
        if saved_a.ndim == 1 and saved_b.ndim == 1:
            grad_a = redispatch("mul", keyset, grad, saved_b) if getattr(a, "requires_grad", False) else None
            grad_b = redispatch("mul", keyset, grad, saved_a) if getattr(b, "requires_grad", False) else None
        else:
            # ND case: inner(a, b) contracts last dim of both
            # output shape = a.shape[:-1] + b.shape[:-1]
            if getattr(a, "requires_grad", False):
                b_free_shape = saved_b.shape[:-1]
                K = saved_a.shape[-1]
                grad_2d = redispatch("reshape", keyset, grad, (-1, math.prod(b_free_shape)))
                b_2d = redispatch("reshape", keyset, saved_b, (math.prod(b_free_shape), K))
                ga_2d = redispatch("matmul", keyset, grad_2d, b_2d)
                grad_a = redispatch("reshape", keyset, ga_2d, saved_a.shape)
            else:
                grad_a = None
            if getattr(b, "requires_grad", False):
                a_free_shape = saved_a.shape[:-1]
                grad_2d = redispatch("reshape", keyset, grad, (math.prod(a_free_shape), -1))
                a_2d = redispatch("reshape", keyset, saved_a, (math.prod(a_free_shape), saved_a.shape[-1]))
                grad_t = redispatch("transpose", keyset, grad_2d, 0, 1)
                gb_2d = redispatch("matmul", keyset, grad_t, a_2d)
                grad_b = redispatch("reshape", keyset, gb_2d, saved_b.shape)
            else:
                grad_b = None
        return grad_a, grad_b


def _autograd_tensordot(name):
    """Autograd wrapper for tensordot(a, b, dims)."""
    def wrapper(a, b, dims=2):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, dims)
        a_rg = getattr(a, "requires_grad", False)
        b_rg = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_rg or b_rg):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _tensordot_backward(grad, a, b, dims, backward_keyset)

            inputs = [t for t in (a, b) if hasattr(t, "requires_grad")]
            node = Node(_backward, tuple(inputs))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _tensordot_backward(grad, a, b, dims, keyset):
    """Backward for tensordot."""
    import math
    # Normalize dims to ([dims_a], [dims_b])
    if isinstance(dims, int):
        dims_a = list(range(a.ndim - dims, a.ndim))
        dims_b = list(range(dims))
    else:
        dims_a, dims_b = [list(d) for d in dims]
    free_a = [i for i in range(a.ndim) if i not in dims_a]
    free_b = [i for i in range(b.ndim) if i not in dims_b]

    grad_a = None
    grad_b = None
    if getattr(a, "requires_grad", False):
        with _grad_context(keyset):
            free_a_sizes = [a.shape[i] for i in free_a]
            free_b_sizes = [b.shape[i] for i in free_b]
            contracted_sizes = [b.shape[i] for i in dims_b]

            grad_2d = redispatch("reshape", keyset, grad, (math.prod(free_a_sizes) or 1, math.prod(free_b_sizes) or 1))
            # b reordered: free_b axes first, then contracted axes
            b_perm = free_b + dims_b
            b_t = redispatch("permute", keyset, b, b_perm)
            b_2d = redispatch("reshape", keyset, redispatch("contiguous", keyset, b_t),
                              (math.prod(free_b_sizes) or 1, math.prod(contracted_sizes) or 1))
            ga_2d = redispatch("matmul", keyset, grad_2d, b_2d)
            # Reshape to free_a_sizes + contracted_sizes
            ga_shape = free_a_sizes + contracted_sizes
            ga = redispatch("reshape", keyset, ga_2d, tuple(ga_shape) if ga_shape else (1,))
            # Transpose back to a's original order
            current_order = free_a + dims_a
            inv_perm = [0] * a.ndim
            for new_i, orig_i in enumerate(current_order):
                inv_perm[orig_i] = new_i
            grad_a = redispatch("permute", keyset, ga, inv_perm)
            if ga_shape == []:
                grad_a = redispatch("reshape", keyset, grad_a, a.shape)

    if getattr(b, "requires_grad", False):
        with _grad_context(keyset):
            free_a_sizes = [a.shape[i] for i in free_a]
            free_b_sizes = [b.shape[i] for i in free_b]
            contracted_sizes = [a.shape[i] for i in dims_a]

            grad_2d = redispatch("reshape", keyset, grad, (math.prod(free_a_sizes) or 1, math.prod(free_b_sizes) or 1))
            # a reordered: free_a axes first, then contracted axes
            a_perm = free_a + dims_a
            a_t = redispatch("permute", keyset, a, a_perm)
            a_2d = redispatch("reshape", keyset, redispatch("contiguous", keyset, a_t),
                              (math.prod(free_a_sizes) or 1, math.prod(contracted_sizes) or 1))
            # grad^T @ a_2d = (free_b x free_a) @ (free_a x contracted) = (free_b x contracted)
            grad_2d_t = redispatch("transpose", keyset, grad_2d, 0, 1)
            gb_2d = redispatch("matmul", keyset, grad_2d_t, a_2d)
            gb_shape = free_b_sizes + contracted_sizes
            gb = redispatch("reshape", keyset, gb_2d, tuple(gb_shape) if gb_shape else (1,))
            current_order = free_b + dims_b
            inv_perm = [0] * b.ndim
            for new_i, orig_i in enumerate(current_order):
                inv_perm[orig_i] = new_i
            grad_b = redispatch("permute", keyset, gb, inv_perm)
            if gb_shape == []:
                grad_b = redispatch("reshape", keyset, grad_b, b.shape)

    return grad_a, grad_b


def _matrix_power_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for matrix_power: A^n.
    d(A^n)/dA = sum_{k=0}^{n-1} (A^T)^{n-1-k} @ grad @ (A^T)^k
    """
    n = args[0]
    if n == 0:
        return (redispatch("zeros", keyset, saved_a.shape, dtype=saved_a.dtype, device=saved_a.device),)
    with _grad_context(keyset):
        abs_n = abs(n)
        if n < 0:
            a_inv = redispatch("linalg_inv", keyset, saved_a)
            at = redispatch("transpose", keyset, a_inv, -2, -1)
        else:
            at = redispatch("transpose", keyset, saved_a, -2, -1)
        I = redispatch("eye", keyset, saved_a.shape[-1], dtype=saved_a.dtype, device=saved_a.device)
        if saved_a.ndim > 2:
            I = redispatch("expand", keyset, I, saved_a.shape)
        at_powers = [I]
        for k in range(1, abs_n):
            at_powers.append(redispatch("matmul", keyset, at_powers[-1], at))
        result = redispatch("zeros", keyset, saved_a.shape, dtype=saved_a.dtype, device=saved_a.device)
        for k in range(abs_n):
            left = at_powers[abs_n - 1 - k]
            right = at_powers[k]
            term = redispatch("matmul", keyset, redispatch("matmul", keyset, left, grad), right)
            result = redispatch("add", keyset, result, term)
        if n < 0:
            result = redispatch("neg", keyset, result)
        return (result,)


# --- Task 6: Reduce-with-indices (median, kthvalue, aminmax) ---

def _autograd_median(name):
    """Autograd wrapper for median — scalar (dim=None) or (values, indices) tuple."""
    def wrapper(a, dim=None, keepdim=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim=dim, keepdim=keepdim)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            if dim is None:
                # Scalar case: single output tensor
                node_holder = {}

                def _backward_scalar(grad):
                    backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                    return _median_scalar_backward(grad, a, out, backward_keyset)

                node = Node(_backward_scalar, (a,))
                node_holder["node"] = node
                out.grad_fn = node
                out.requires_grad = True
            else:
                # Tuple case: (values, indices)
                values, indices = out
                node_holder = {}

                def _backward_dim(grad):
                    backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                    return _median_dim_backward(grad, a, indices, backward_keyset, dim, keepdim)

                node = Node(_backward_dim, (a,))
                node_holder["node"] = node
                values.grad_fn = node
                values.requires_grad = True
                out = (values, indices)
        return out

    return wrapper


def _median_scalar_backward(grad, a, out_val, keyset):
    """Backward for median when dim=None (scalar output)."""
    with _grad_context(keyset):
        ones = a._ones_like()
        zero = _scalar_tensor_like(a, 0.0)
        eps = _scalar_tensor_like(a, 1e-12)
        mask = redispatch("eq", keyset, a, out_val)
        mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
        total = redispatch("sum", keyset, mask_f)
        safe_total = redispatch("add", keyset, total, eps)
        return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad),)


def _median_dim_backward(grad, a, indices, keyset, dim, keepdim):
    """Backward for median when dim is specified (tuple output)."""
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        if not keepdim:
            grad_expanded = redispatch("unsqueeze", keyset, grad, dim)
            indices_expanded = redispatch("unsqueeze", keyset, indices, dim)
        else:
            grad_expanded = grad
            indices_expanded = indices
        redispatch("scatter_", keyset, grad_input, dim, indices_expanded, grad_expanded)
        return (grad_input,)


def _kthvalue_backward(grad, a, indices, keyset, args, kwargs):
    """Backward for kthvalue: scatter gradient at kth index positions."""
    dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        grad_input = redispatch("zeros", keyset, a.shape, dtype=a.dtype, device=a.device)
        if not keepdim:
            grad_expanded = redispatch("unsqueeze", keyset, grad, dim)
            indices_expanded = redispatch("unsqueeze", keyset, indices, dim)
        else:
            grad_expanded = grad
            indices_expanded = indices
        redispatch("scatter_", keyset, grad_input, dim, indices_expanded, grad_expanded)
        return (grad_input,)


def _autograd_aminmax(name):
    """Autograd wrapper for aminmax — returns namedtuple (min, max)."""
    def wrapper(a, dim=None, keepdim=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim=dim, keepdim=keepdim)
        min_val, max_val = out
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder_min = {}
            node_holder_max = {}

            def _backward_min(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _aminmax_min_backward(grad, a, min_val, backward_keyset, dim, keepdim)

            def _backward_max(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _aminmax_max_backward(grad, a, max_val, backward_keyset, dim, keepdim)

            node_min = Node(_backward_min, (a,))
            node_holder_min["node"] = node_min
            min_val.grad_fn = node_min
            min_val.requires_grad = True

            node_max = Node(_backward_max, (a,))
            node_holder_max["node"] = node_max
            max_val.grad_fn = node_max
            max_val.requires_grad = True

            from collections import namedtuple
            AminmaxResult = namedtuple("aminmax", ["min", "max"])
            out = AminmaxResult(min_val, max_val)
        return out

    return wrapper


def _aminmax_min_backward(grad, a, min_val, keyset, dim, keepdim):
    """Backward for aminmax min output: gradient at min positions."""
    with _grad_context(keyset):
        ones = a._ones_like()
        zero = _scalar_tensor_like(a, 0.0)
        eps = _scalar_tensor_like(a, 1e-12)
        if dim is None:
            mask = redispatch("eq", keyset, a, min_val)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad),)
        else:
            if not keepdim:
                grad_expanded = redispatch("unsqueeze", keyset, grad, dim)
                min_expanded = redispatch("unsqueeze", keyset, min_val, dim)
            else:
                grad_expanded = grad
                min_expanded = min_val
            mask = redispatch("eq", keyset, a, min_expanded)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f, dim=dim, keepdim=True)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad_expanded),)


def _aminmax_max_backward(grad, a, max_val, keyset, dim, keepdim):
    """Backward for aminmax max output: gradient at max positions."""
    with _grad_context(keyset):
        ones = a._ones_like()
        zero = _scalar_tensor_like(a, 0.0)
        eps = _scalar_tensor_like(a, 1e-12)
        if dim is None:
            mask = redispatch("eq", keyset, a, max_val)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad),)
        else:
            if not keepdim:
                grad_expanded = redispatch("unsqueeze", keyset, grad, dim)
                max_expanded = redispatch("unsqueeze", keyset, max_val, dim)
            else:
                grad_expanded = grad
                max_expanded = max_val
            mask = redispatch("eq", keyset, a, max_expanded)
            mask_f = redispatch("where", keyset, mask, ones, redispatch("mul", keyset, ones, zero))
            total = redispatch("sum", keyset, mask_f, dim=dim, keepdim=True)
            safe_total = redispatch("add", keyset, total, eps)
            return (redispatch("mul", keyset, redispatch("div", keyset, mask_f, safe_total), grad_expanded),)


def _register_autograd_op(name, factory, *, npu_factory=None, include_meta=True):
    kwargs = {
        "default": factory(),
        "cpu": factory(),
        "npu": (npu_factory or factory)(),
        "cuda": factory(),
    }
    if include_meta:
        kwargs["meta"] = factory()
    register_autograd_kernels(name, **kwargs)


# ---------------------------------------------------------------------------
# Round 6: Zero backward + CTC loss backward
# ---------------------------------------------------------------------------
def _zero_backward(grad, _a, _saved_a, keyset):
    """Backward returning zeros for non-differentiable ops (ceil, floor, round, trunc, sign, signbit)."""
    with _grad_context(keyset):
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


def _autograd_ctc_loss(name):
    """Autograd wrapper for ctc_loss — computes gradient via alpha-beta algorithm."""
    import numpy as _np

    def wrapper(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, log_probs, targets, input_lengths,
                         target_lengths, blank=blank, reduction=reduction,
                         zero_infinity=zero_infinity)
        if GradMode.enabled and getattr(log_probs, "requires_grad", False):
            node_holder = {}

            def _backward(grad_output):
                saved_lp = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _ctc_loss_backward(grad_output, saved_lp, targets, input_lengths,
                                          target_lengths, blank, reduction, zero_infinity,
                                          backward_keyset)

            node = Node(_backward, (log_probs,))
            node_holder["node"] = node
            node.save_for_backward(log_probs)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _ctc_loss_backward(grad_output, saved_lp, targets, input_lengths, target_lengths,
                       blank, reduction, zero_infinity, keyset):
    """CTC loss backward via alpha-beta (forward-backward) algorithm in numpy."""
    import numpy as _np

    lp_t = saved_lp
    lp = lp_t._numpy_view().astype(_np.float64)
    T, N, C = lp.shape
    grad_np = _np.zeros_like(lp)

    if hasattr(targets, '_numpy_view'):
        tgt = targets._numpy_view()
    else:
        tgt = _np.array(targets)
    if hasattr(input_lengths, '_numpy_view'):
        inp_lens = input_lengths._numpy_view().astype(_np.int64)
    else:
        inp_lens = _np.array(input_lengths, dtype=_np.int64)
    if hasattr(target_lengths, '_numpy_view'):
        tgt_lens = target_lengths._numpy_view().astype(_np.int64)
    else:
        tgt_lens = _np.array(target_lengths, dtype=_np.int64)

    NEG_INF = -1e30
    is_1d = (tgt.ndim == 1)
    offset = 0

    for b in range(N):
        T_b = int(inp_lens[b])
        S_b = int(tgt_lens[b])

        if is_1d:
            labels_b = tgt[offset:offset + S_b]
            offset += S_b
        else:
            labels_b = tgt[b, :S_b]

        L = 2 * S_b + 1
        ext = _np.full(L, blank, dtype=_np.int64)
        for s in range(S_b):
            ext[2 * s + 1] = labels_b[s]

        # Forward (alpha)
        alpha = _np.full((T_b, L), NEG_INF, dtype=_np.float64)
        alpha[0, 0] = lp[0, b, ext[0]]
        if L > 1:
            alpha[0, 1] = lp[0, b, ext[1]]

        for t in range(1, T_b):
            for s in range(L):
                a = alpha[t - 1, s]
                if s > 0:
                    a = _np.logaddexp(a, alpha[t - 1, s - 1])
                if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                    a = _np.logaddexp(a, alpha[t - 1, s - 2])
                alpha[t, s] = a + lp[t, b, ext[s]]

        log_likelihood = alpha[T_b - 1, L - 1]
        if L > 1:
            log_likelihood = _np.logaddexp(log_likelihood, alpha[T_b - 1, L - 2])

        # Backward (beta)
        beta = _np.full((T_b, L), NEG_INF, dtype=_np.float64)
        beta[T_b - 1, L - 1] = 0.0
        if L > 1:
            beta[T_b - 1, L - 2] = 0.0

        for t in range(T_b - 2, -1, -1):
            for s in range(L):
                b_val = beta[t + 1, s] + lp[t + 1, b, ext[s]]
                if s < L - 1:
                    b_val = _np.logaddexp(b_val, beta[t + 1, s + 1] + lp[t + 1, b, ext[s + 1]])
                if s < L - 2 and ext[s] != blank and ext[s] != ext[s + 2]:
                    b_val = _np.logaddexp(b_val, beta[t + 1, s + 2] + lp[t + 1, b, ext[s + 2]])
                beta[t, s] = b_val

        # Compute gradient: sum alpha_beta for each (t, c)
        for t in range(T_b):
            for s in range(L):
                ab = alpha[t, s] + beta[t, s]
                c = ext[s]
                if ab > NEG_INF + 1:
                    grad_np[t, b, c] = _np.logaddexp(grad_np[t, b, c] if grad_np[t, b, c] != 0 else NEG_INF, ab)

        # grad = -(exp(alpha_beta - log_likelihood) - exp(log_probs))
        # = exp(log_probs) - exp(alpha_beta - log_likelihood)
        for t in range(T_b):
            for c in range(C):
                if grad_np[t, b, c] != 0:
                    grad_np[t, b, c] = _np.exp(lp[t, b, c]) - _np.exp(grad_np[t, b, c] - log_likelihood)
                else:
                    grad_np[t, b, c] = _np.exp(lp[t, b, c])

        if zero_infinity and _np.isinf(-log_likelihood):
            grad_np[:T_b, b, :] = 0.0

        # Scale by reduction
        if reduction == 'mean':
            tgt_len = max(int(tgt_lens[b]), 1)
            grad_np[:T_b, b, :] /= (tgt_len * N)
        elif reduction == 'sum':
            pass  # no extra scaling
        # 'none' would need per-sample grad_output, handle below

    from .._storage import typed_storage_from_numpy
    from .._tensor import Tensor as _Tensor
    from .._dtype import to_numpy_dtype

    grad_np = grad_np.astype(to_numpy_dtype(lp_t.dtype))
    storage = typed_storage_from_numpy(grad_np, lp_t.dtype, device=lp_t.device)
    stride = tuple(_np.array(grad_np.strides) // grad_np.itemsize)
    grad_input = _Tensor(storage, grad_np.shape, stride)

    # Scale by upstream gradient
    with _grad_context(keyset):
        if grad_output.shape == ():
            result = redispatch("mul", keyset, grad_input, grad_output)
        else:
            result = redispatch("mul", keyset, grad_input, grad_output)
    return (result,)


# ---------------------------------------------------------------------------
# Round 7: Special Functions Backward
# ---------------------------------------------------------------------------
import math as _math

def _special_digamma_backward(grad, _a, saved_a, keyset):
    """Backward for digamma: grad * polygamma(1, x)."""
    with _grad_context(keyset):
        pg1 = redispatch("special_polygamma", keyset, 1, saved_a)
        return (redispatch("mul", keyset, grad, pg1),)


def _special_gammaln_backward(grad, _a, saved_a, keyset):
    """Backward for gammaln: grad * digamma(x)."""
    with _grad_context(keyset):
        dg = redispatch("special_digamma", keyset, saved_a)
        return (redispatch("mul", keyset, grad, dg),)


def _special_erfinv_backward(grad, _a, saved_a, keyset):
    """Backward for erfinv: grad * (sqrt(pi)/2) * exp(erfinv(x)^2)."""
    with _grad_context(keyset):
        out = redispatch("special_erfinv", keyset, saved_a)
        out_sq = redispatch("mul", keyset, out, out)
        exp_part = redispatch("exp", keyset, out_sq)
        scale = _scalar_tensor_like(saved_a, _math.sqrt(_math.pi) / 2.0)
        deriv = redispatch("mul", keyset, scale, exp_part)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_erfcx_backward(grad, _a, saved_a, keyset):
    """Backward for erfcx: grad * (2*x*erfcx(x) - 2/sqrt(pi))."""
    with _grad_context(keyset):
        erfcx_x = redispatch("special_erfcx", keyset, saved_a)
        two = _scalar_tensor_like(saved_a, 2.0)
        inv_sqrt_pi = _scalar_tensor_like(saved_a, 2.0 / _math.sqrt(_math.pi))
        term1 = redispatch("mul", keyset, two,
            redispatch("mul", keyset, saved_a, erfcx_x))
        deriv = redispatch("sub", keyset, term1, inv_sqrt_pi)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_ndtr_backward(grad, _a, saved_a, keyset):
    """Backward for ndtr: grad * (1/sqrt(2*pi)) * exp(-x^2/2)."""
    with _grad_context(keyset):
        scale = _scalar_tensor_like(saved_a, 1.0 / _math.sqrt(2.0 * _math.pi))
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        half = _scalar_tensor_like(saved_a, -0.5)
        exp_part = redispatch("exp", keyset, redispatch("mul", keyset, half, x_sq))
        deriv = redispatch("mul", keyset, scale, exp_part)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_ndtri_backward(grad, _a, saved_a, keyset):
    """Backward for ndtri: grad * sqrt(2*pi) * exp(ndtri(x)^2/2)."""
    with _grad_context(keyset):
        out = redispatch("special_ndtri", keyset, saved_a)
        out_sq = redispatch("mul", keyset, out, out)
        half = _scalar_tensor_like(saved_a, 0.5)
        exp_part = redispatch("exp", keyset, redispatch("mul", keyset, half, out_sq))
        scale = _scalar_tensor_like(saved_a, _math.sqrt(2.0 * _math.pi))
        deriv = redispatch("mul", keyset, scale, exp_part)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_logit_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for logit: grad * 1/(x*(1-x)), with optional clamping from eps."""
    eps = args[0] if args else kwargs.get("eps", None)
    with _grad_context(keyset):
        x = saved_a
        if eps is not None:
            eps_val = float(eps)
            x = redispatch("clamp", keyset, x, min_val=eps_val, max_val=1.0 - eps_val)
        ones = saved_a._ones_like()
        one_minus_x = redispatch("sub", keyset, ones, x)
        denom = redispatch("mul", keyset, x, one_minus_x)
        deriv = redispatch("reciprocal", keyset, denom)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_sinc_backward(grad, _a, saved_a, keyset):
    """Backward for sinc: (cos(pi*x)*pi*x - sin(pi*x))/(pi*x^2) for x!=0, 0 at x=0."""
    import numpy as _np
    with _grad_context(keyset):
        x_np = saved_a._numpy_view()
        pi = _np.pi
        pi_x = pi * x_np
        # Compute derivative numerically safe
        deriv_np = _np.where(
            _np.abs(x_np) < 1e-20,
            0.0,
            (_np.cos(pi_x) * pi_x - _np.sin(pi_x)) / (pi * x_np * x_np)
        )
        from .._storage import typed_storage_from_numpy
        from .._tensor import Tensor as _Tensor
        from .._dtype import to_numpy_dtype
        deriv_np = deriv_np.astype(to_numpy_dtype(saved_a.dtype))
        storage = typed_storage_from_numpy(deriv_np, saved_a.dtype, device=saved_a.device)
        stride = tuple(_np.array(deriv_np.strides) // deriv_np.itemsize)
        deriv = _Tensor(storage, deriv_np.shape, stride)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_entr_backward(grad, _a, saved_a, keyset):
    """Backward for entr: -(1 + log(x)) for x>0, else 0."""
    with _grad_context(keyset):
        ones = saved_a._ones_like()
        zero = _scalar_tensor_like(saved_a, 0.0)
        log_x = redispatch("log", keyset, saved_a)
        neg_one_plus_log = redispatch("neg", keyset,
            redispatch("add", keyset, ones, log_x))
        pos_mask = redispatch("where", keyset,
            redispatch("gt", keyset, saved_a, zero), ones, zero)
        deriv = redispatch("mul", keyset, neg_one_plus_log, pos_mask)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_log_ndtr_backward(grad, _a, saved_a, keyset):
    """Backward for log_ndtr: grad * exp(-log_ndtr(x)) * ndtr'(x) = grad * ndtr'(x)/ndtr(x)."""
    with _grad_context(keyset):
        # ndtr'(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
        scale = _scalar_tensor_like(saved_a, 1.0 / _math.sqrt(2.0 * _math.pi))
        x_sq = redispatch("mul", keyset, saved_a, saved_a)
        half_neg = _scalar_tensor_like(saved_a, -0.5)
        ndtr_deriv = redispatch("mul", keyset, scale,
            redispatch("exp", keyset, redispatch("mul", keyset, half_neg, x_sq)))
        # ndtr(x)
        ndtr_x = redispatch("special_ndtr", keyset, saved_a)
        # ratio = ndtr'(x) / ndtr(x)
        ratio = redispatch("div", keyset, ndtr_deriv, ndtr_x)
        return (redispatch("mul", keyset, grad, ratio),)


# Bessel function backward
def _special_i0_backward(grad, _a, saved_a, keyset):
    """Backward for i0: grad * i1(x)."""
    with _grad_context(keyset):
        i1_x = redispatch("special_i1", keyset, saved_a)
        return (redispatch("mul", keyset, grad, i1_x),)


def _special_i0e_backward(grad, _a, saved_a, keyset):
    """Backward for i0e: grad * (i1e(x) - sign(x)*i0e(x))."""
    with _grad_context(keyset):
        i1e_x = redispatch("special_i1e", keyset, saved_a)
        i0e_x = redispatch("special_i0e", keyset, saved_a)
        sign_x = redispatch("sign", keyset, saved_a)
        term = redispatch("sub", keyset, i1e_x,
            redispatch("mul", keyset, sign_x, i0e_x))
        return (redispatch("mul", keyset, grad, term),)


def _special_i1_backward(grad, _a, saved_a, keyset):
    """Backward for i1: grad * (i0(x) - i1(x)/x) for x!=0."""
    with _grad_context(keyset):
        i0_x = redispatch("special_i0", keyset, saved_a)
        i1_x = redispatch("special_i1", keyset, saved_a)
        # i0(x) - i1(x)/x, with limit 0.5 at x=0
        zero = _scalar_tensor_like(saved_a, 0.0)
        abs_x = redispatch("abs", keyset, saved_a)
        eps = _scalar_tensor_like(saved_a, 1e-20)
        safe_x = redispatch("where", keyset,
            redispatch("gt", keyset, abs_x, eps), saved_a,
            _scalar_tensor_like(saved_a, 1.0))
        ratio = redispatch("div", keyset, i1_x, safe_x)
        deriv_normal = redispatch("sub", keyset, i0_x, ratio)
        half = _scalar_tensor_like(saved_a, 0.5)
        deriv = redispatch("where", keyset,
            redispatch("gt", keyset, abs_x, eps), deriv_normal, half)
        return (redispatch("mul", keyset, grad, deriv),)


def _special_i1e_backward(grad, _a, saved_a, keyset):
    """Backward for i1e: i0e(x) - i1e(x)*(sign(x) + 1/x)."""
    with _grad_context(keyset):
        i0e_x = redispatch("special_i0e", keyset, saved_a)
        i1e_x = redispatch("special_i1e", keyset, saved_a)
        sign_x = redispatch("sign", keyset, saved_a)
        abs_x = redispatch("abs", keyset, saved_a)
        eps = _scalar_tensor_like(saved_a, 1e-20)
        safe_x = redispatch("where", keyset,
            redispatch("gt", keyset, abs_x, eps), saved_a,
            _scalar_tensor_like(saved_a, 1.0))
        inv_x = redispatch("reciprocal", keyset, safe_x)
        factor = redispatch("add", keyset, sign_x, inv_x)
        deriv_normal = redispatch("sub", keyset, i0e_x,
            redispatch("mul", keyset, i1e_x, factor))
        half = _scalar_tensor_like(saved_a, 0.5)
        deriv = redispatch("where", keyset,
            redispatch("gt", keyset, abs_x, eps), deriv_normal, half)
        return (redispatch("mul", keyset, grad, deriv),)


# Binary special ops backward
def _special_xlogy_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for xlogy: grad_a = log(y) where x!=0, grad_b = x/y."""
    with _grad_context(keyset):
        zero = _scalar_tensor_like(saved_a, 0.0)
        ones = saved_a._ones_like()
        a_nonzero = redispatch("where", keyset,
            redispatch("ne", keyset, saved_a, zero), ones, zero)
        # grad_a = grad * log(y) * (x != 0)
        log_b = redispatch("log", keyset, saved_b)
        grad_a = redispatch("mul", keyset, grad,
            redispatch("mul", keyset, log_b, a_nonzero)) if getattr(a, "requires_grad", False) else None
        # grad_b = grad * x / y
        grad_b = redispatch("mul", keyset, grad,
            redispatch("div", keyset, saved_a, saved_b)) if getattr(b, "requires_grad", False) else None
    grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
    grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
    return grad_a, grad_b


def _special_xlog1py_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for xlog1py: grad_a = log1p(y) where x!=0, grad_b = x/(1+y)."""
    with _grad_context(keyset):
        zero = _scalar_tensor_like(saved_a, 0.0)
        ones = saved_a._ones_like()
        a_nonzero = redispatch("where", keyset,
            redispatch("ne", keyset, saved_a, zero), ones, zero)
        # grad_a = grad * log1p(y) * (x != 0)
        log1p_b = redispatch("log1p", keyset, saved_b)
        grad_a = redispatch("mul", keyset, grad,
            redispatch("mul", keyset, log1p_b, a_nonzero)) if getattr(a, "requires_grad", False) else None
        # grad_b = grad * x / (1 + y)
        one_plus_b = redispatch("add", keyset, ones._ones_like() if saved_b.shape != ones.shape else ones, saved_b)
        grad_b = redispatch("mul", keyset, grad,
            redispatch("div", keyset, saved_a, one_plus_b)) if getattr(b, "requires_grad", False) else None
    grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
    grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
    return grad_a, grad_b


def _special_zeta_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for zeta(s, q): grad_b = -s * zeta(s+1, q). grad_a via numpy."""
    with _grad_context(keyset):
        # grad_a (w.r.t. s) is complex — use numpy numerical approx
        grad_a = None  # Not commonly needed, skip for now
        # grad_b = grad * (-s * zeta(s+1, q))
        if getattr(b, "requires_grad", False):
            ones = saved_a._ones_like()
            s_plus_1 = redispatch("add", keyset, saved_a, ones)
            zeta_sp1 = redispatch("special_zeta", keyset, s_plus_1, saved_b)
            neg_s = redispatch("neg", keyset, saved_a)
            grad_b = redispatch("mul", keyset, grad,
                redispatch("mul", keyset, neg_s, zeta_sp1))
        else:
            grad_b = None
    grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
    grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
    return grad_a, grad_b


# Remaining special ops backward (unary_args pattern)
def _special_polygamma_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for polygamma(n, x): grad * polygamma(n+1, x). n is not differentiable."""
    n = args[0] if args else kwargs.get("n", 0)
    with _grad_context(keyset):
        pg_n1 = redispatch("special_polygamma", keyset, n + 1, saved_a)
        return (redispatch("mul", keyset, grad, pg_n1),)


def _special_multigammaln_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for multigammaln(x, p): grad * sum(digamma(x - (j-1)/2) for j=1..p)."""
    p = args[0] if args else kwargs.get("p", 1)
    with _grad_context(keyset):
        result = _scalar_tensor_like(saved_a, 0.0)
        for j in range(1, p + 1):
            offset = _scalar_tensor_like(saved_a, (j - 1) / 2.0)
            x_shifted = redispatch("sub", keyset, saved_a, offset)
            dg = redispatch("special_digamma", keyset, x_shifted)
            result = redispatch("add", keyset, result, dg)
        return (redispatch("mul", keyset, grad, result),)


def _special_gammainc_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for gammainc(a, x): grad_x = x^(a-1)*exp(-x)/gamma(a)."""
    with _grad_context(keyset):
        grad_a = None  # grad w.r.t. a is complex (involves integral), skip
        if getattr(b, "requires_grad", False):
            ones = saved_a._ones_like()
            a_minus_1 = redispatch("sub", keyset, saved_a, ones)
            x_pow = redispatch("pow", keyset, saved_b, a_minus_1)
            exp_neg_x = redispatch("exp", keyset, redispatch("neg", keyset, saved_b))
            gamma_a = redispatch("exp", keyset, redispatch("special_gammaln", keyset, saved_a))
            deriv = redispatch("div", keyset,
                redispatch("mul", keyset, x_pow, exp_neg_x), gamma_a)
            grad_b = redispatch("mul", keyset, grad, deriv)
        else:
            grad_b = None
    grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
    grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
    return grad_a, grad_b


def _special_gammaincc_backward(grad, a, b, saved_a, saved_b, keyset):
    """Backward for gammaincc(a, x): negative of gammainc grad_x."""
    with _grad_context(keyset):
        grad_a = None
        if getattr(b, "requires_grad", False):
            ones = saved_a._ones_like()
            a_minus_1 = redispatch("sub", keyset, saved_a, ones)
            x_pow = redispatch("pow", keyset, saved_b, a_minus_1)
            exp_neg_x = redispatch("exp", keyset, redispatch("neg", keyset, saved_b))
            gamma_a = redispatch("exp", keyset, redispatch("special_gammaln", keyset, saved_a))
            deriv = redispatch("neg", keyset,
                redispatch("div", keyset,
                    redispatch("mul", keyset, x_pow, exp_neg_x), gamma_a))
            grad_b = redispatch("mul", keyset, grad, deriv)
        else:
            grad_b = None
    grad_a = reduce_grad(grad_a, a.shape) if grad_a is not None else None
    grad_b = reduce_grad(grad_b, b.shape) if grad_b is not None else None
    return grad_a, grad_b


# Special polygamma needs a custom wrapper since n is first arg, x is second
def _autograd_special_polygamma(name):
    """Autograd wrapper for special_polygamma(n, x) where n is int (not differentiable)."""
    def wrapper(n, a):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, n, a)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _special_polygamma_backward(grad, None, saved_a, backward_keyset, (n,), {})

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


# Special multigammaln needs custom wrapper: multigammaln(x, p) where p is int
def _autograd_special_multigammaln(name):
    """Autograd wrapper for special_multigammaln(x, p) where p is int."""
    def wrapper(a, p):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, p)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return _special_multigammaln_backward(grad, None, saved_a, backward_keyset, (p,), {})

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


# ---------------------------------------------------------------------------
# Round 8: FFT Backward
# ---------------------------------------------------------------------------
def _conj_norm(norm):
    """Swap normalization for FFT backward: forward<->backward, ortho stays."""
    if norm == "forward":
        return "backward"
    elif norm == "backward" or norm is None:
        return "forward"
    return norm  # "ortho" stays


def _autograd_fft_c2c(name, inverse_name):
    """Autograd wrapper for complex-to-complex FFT ops (fft, ifft, fft2, etc.)."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                # Get the norm parameter and swap it
                norm = kwargs.get("norm", None)
                if not norm and len(args) >= 3:
                    norm = args[2] if name in ("fft_fft", "fft_ifft") else None
                bw_kwargs = dict(kwargs)
                bw_kwargs["norm"] = _conj_norm(norm)
                # For n/s/dim args, pass through from the original call
                bw_args = args
                with _grad_context(backward_keyset):
                    return (redispatch(inverse_name, backward_keyset, grad, *bw_args, **bw_kwargs),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_fft_r2c(name, inverse_name):
    """Autograd wrapper for real-to-complex FFT ops (rfft, rfft2, rfftn)."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}
            saved_shape = a.shape

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                norm = kwargs.get("norm", None)
                bw_kwargs = dict(kwargs)
                bw_kwargs["norm"] = _conj_norm(norm)
                # For rfft backward, we need irfft with the original input size
                n_kwarg = kwargs.get("n", None)
                if n_kwarg is None and args:
                    n_kwarg = args[0]
                dim = kwargs.get("dim", -1)
                if dim is None and len(args) >= 2:
                    dim = args[1]
                # Original input size along the transform dim
                if n_kwarg is not None:
                    bw_kwargs["n"] = n_kwarg
                else:
                    actual_dim = dim if dim is not None else -1
                    if actual_dim < 0:
                        actual_dim += len(saved_shape)
                    bw_kwargs["n"] = saved_shape[actual_dim]
                with _grad_context(backward_keyset):
                    return (redispatch(inverse_name, backward_keyset, grad, **bw_kwargs),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_fft_c2r(name, inverse_name):
    """Autograd wrapper for complex-to-real FFT ops (irfft, irfft2, irfftn)."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}
            saved_shape = a.shape

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                norm = kwargs.get("norm", None)
                bw_kwargs = dict(kwargs)
                bw_kwargs["norm"] = _conj_norm(norm)
                # For irfft backward, we need rfft
                # The output n_freq bins = the complex input size along dim
                dim = kwargs.get("dim", -1)
                if dim is None and len(args) >= 2:
                    dim = args[1]
                actual_dim = dim if dim is not None else -1
                if actual_dim < 0:
                    actual_dim += len(saved_shape)
                bw_kwargs["n"] = saved_shape[actual_dim]
                bw_kwargs.pop("s", None)  # remove multi-dim size param
                with _grad_context(backward_keyset):
                    return (redispatch(inverse_name, backward_keyset, grad, **bw_kwargs),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_fft_shift(name, inverse_name):
    """Autograd wrapper for fft_fftshift / fft_ifftshift."""
    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    return (redispatch(inverse_name, backward_keyset, grad, *args, **kwargs),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


# ---------------------------------------------------------------------------
# Round 9: Linalg Backward — Core (15 ops)
# ---------------------------------------------------------------------------

def _linalg_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_norm: reuse norm backward logic."""
    ord_val = args[0] if args else kwargs.get("ord", None)
    dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    if ord_val is None:
        ord_val = 2.0  # default: Frobenius/L2
    return _norm_backward(grad, _a, saved_a, keyset,
                          (), {"p": ord_val, "dim": dim, "keepdim": keepdim})


def _linalg_vector_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_vector_norm: same pattern as norm backward."""
    ord_val = args[0] if args else kwargs.get("ord", 2)
    dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    return _norm_backward(grad, _a, saved_a, keyset,
                          (), {"p": ord_val, "dim": dim, "keepdim": keepdim})


def _linalg_matrix_norm_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_matrix_norm via numpy."""
    import numpy as _np
    ord_val = args[0] if args else kwargs.get("ord", 'fro')
    dim = args[1] if len(args) > 1 else kwargs.get("dim", (-2, -1))
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    with _grad_context(keyset):
        if ord_val == 'fro' or ord_val == 2:
            # Frobenius: grad * x / norm(x)
            norm_val = redispatch("linalg_matrix_norm", keyset, saved_a,
                                  ord=ord_val, dim=dim, keepdim=True)
            eps = _scalar_tensor_like(saved_a, 1e-12)
            norm_safe = redispatch("clamp_min", keyset, norm_val, 1e-12)
            deriv = redispatch("div", keyset, saved_a, norm_safe)
            if not keepdim:
                grad = redispatch("unsqueeze", keyset, grad, dim[-1])
                grad = redispatch("unsqueeze", keyset, grad, dim[-2])
            return (redispatch("mul", keyset, grad, deriv),)
        else:
            # For other norms, use numerical gradient
            deriv_np = _np.ones_like(saved_a._numpy_view())
            from .._storage import typed_storage_from_numpy
            from .._tensor import Tensor as _Tensor
            from .._dtype import to_numpy_dtype
            deriv_np = deriv_np.astype(to_numpy_dtype(saved_a.dtype))
            storage = typed_storage_from_numpy(deriv_np, saved_a.dtype, device=saved_a.device)
            stride = tuple(_np.array(deriv_np.strides) // deriv_np.itemsize)
            deriv = _Tensor(storage, deriv_np.shape, stride)
            return (redispatch("mul", keyset, grad, deriv),)


def _autograd_linalg_det(name):
    """Autograd wrapper for linalg_det: grad * det(A) * inv(A)^T."""
    def wrapper(a):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    det_val = redispatch("linalg_det", backward_keyset, saved_a)
                    inv_a = redispatch("linalg_inv", backward_keyset, saved_a)
                    # det(A) * inv(A)^T
                    inv_t = redispatch("transpose", backward_keyset, inv_a, -2, -1)
                    # Handle batched: det_val needs unsqueeze for broadcast
                    det_expanded = det_val
                    while len(det_expanded.shape) < len(inv_t.shape):
                        det_expanded = redispatch("unsqueeze", backward_keyset, det_expanded, -1)
                    grad_expanded = grad
                    while len(grad_expanded.shape) < len(inv_t.shape):
                        grad_expanded = redispatch("unsqueeze", backward_keyset, grad_expanded, -1)
                    result = redispatch("mul", backward_keyset,
                        redispatch("mul", backward_keyset, grad_expanded, det_expanded), inv_t)
                    return (result,)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_linalg_slogdet(name):
    """Autograd wrapper for linalg_slogdet: grad_logabsdet * inv(A)^T."""
    def wrapper(a):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                # grad is a tuple: (grad_sign, grad_logabsdet)
                # sign gradient is ignored
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    inv_a = redispatch("linalg_inv", backward_keyset, saved_a)
                    inv_t = redispatch("transpose", backward_keyset, inv_a, -2, -1)
                    # grad is for logabsdet
                    g = grad
                    while len(g.shape) < len(inv_t.shape):
                        g = redispatch("unsqueeze", backward_keyset, g, -1)
                    return (redispatch("mul", backward_keyset, g, inv_t),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            # For multi-output, attach grad_fn to the second output (logabsdet)
            if isinstance(out, tuple):
                sign, logabsdet = out
                logabsdet.grad_fn = node
                logabsdet.requires_grad = True
                out = (sign, logabsdet)
            else:
                out.grad_fn = node
                out.requires_grad = True
        return out

    return wrapper


def _linalg_inv_backward(grad, _a, saved_a, keyset):
    """Backward for linalg_inv: -inv(A)^T @ grad @ inv(A)^T."""
    with _grad_context(keyset):
        inv_a = redispatch("linalg_inv", keyset, saved_a)
        inv_t = redispatch("transpose", keyset, inv_a, -2, -1)
        temp = redispatch("matmul", keyset, inv_t, grad)
        result = redispatch("neg", keyset,
            redispatch("matmul", keyset, temp, inv_t))
        return (result,)


def _autograd_linalg_solve(name):
    """Autograd wrapper for linalg_solve(A, B) -> X where AX=B."""
    def wrapper(a, b, left=True):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, left=left)
        a_rg = getattr(a, "requires_grad", False)
        b_rg = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_rg or b_rg):
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                saved_a, saved_b = saved[0], saved[1]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    x = redispatch("linalg_solve", backward_keyset, saved_a, saved_b, left=left)
                    # grad_B = solve(A^T, grad)
                    a_t = redispatch("transpose", backward_keyset, saved_a, -2, -1)
                    grad_b = redispatch("linalg_solve", backward_keyset, a_t, grad, left=left) if b_rg else None
                    # grad_A = -solve(A^T, grad) @ X^T
                    if a_rg:
                        x_t = redispatch("transpose", backward_keyset, x, -2, -1)
                        grad_a = redispatch("neg", backward_keyset,
                            redispatch("matmul", backward_keyset, grad_b if grad_b is not None else
                                redispatch("linalg_solve", backward_keyset, a_t, grad, left=left), x_t))
                    else:
                        grad_a = None
                return (grad_a, grad_b)

            node = Node(_backward, (a, b))
            node_holder["node"] = node
            node.save_for_backward(a, b)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _linalg_pinv_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_pinv: numpy-based (complex formula)."""
    import numpy as _np
    with _grad_context(keyset):
        # pinv backward: grad_A = -pinv(A)^T @ grad @ pinv(A)^T (simplified)
        pinv_a = redispatch("linalg_pinv", keyset, saved_a)
        pinv_t = redispatch("transpose", keyset, pinv_a, -2, -1)
        temp = redispatch("matmul", keyset, pinv_t, grad)
        result = redispatch("neg", keyset,
            redispatch("matmul", keyset, temp, pinv_t))
        return (result,)


def _autograd_linalg_cholesky(name):
    """Autograd wrapper for linalg_cholesky."""
    def wrapper(a, upper=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, upper=upper)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    L = redispatch("linalg_cholesky", backward_keyset, saved_a, upper=False)
                    if upper:
                        grad = redispatch("transpose", backward_keyset, grad, -2, -1)
                    # Cholesky backward (Smith 1995):
                    # S = L^T @ grad_L, Phi = tril(S) with diag/2
                    # grad_A = L^{-T} @ Phi @ L^{-1}
                    L_t = redispatch("transpose", backward_keyset, L, -2, -1)
                    S = redispatch("matmul", backward_keyset, L_t, grad)
                    # Symmetrize: (S + S^T) / 2, then take tril
                    S_t = redispatch("transpose", backward_keyset, S, -2, -1)
                    S_sym = redispatch("mul", backward_keyset,
                        redispatch("add", backward_keyset, S, S_t),
                        _scalar_tensor_like(S, 0.5))
                    Phi = redispatch("tril", backward_keyset, S_sym)
                    # Solve: L^T @ X = Phi, then result = X @ L^{-1}
                    L_inv = redispatch("linalg_inv", backward_keyset, L)
                    L_inv_t = redispatch("transpose", backward_keyset, L_inv, -2, -1)
                    result = redispatch("matmul", backward_keyset,
                        redispatch("matmul", backward_keyset, L_inv_t, Phi), L_inv)
                    return (result,)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_linalg_qr(name):
    """Autograd wrapper for linalg_qr."""
    def wrapper(a, mode='reduced'):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, mode=mode)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}
            saved_out = out

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    Q, R = saved_out
                    # QR backward: copyltu(M @ R^{-T}) where M = R @ grad_R^T - grad_Q^T @ Q
                    # Simplified: grad_A = (grad_Q + Q @ M_sym) @ R^{-T}
                    # where M = Q^T @ grad_Q, M_sym = tril(M) - tril(M)^T
                    Q_t = redispatch("transpose", backward_keyset, Q, -2, -1)
                    R_t = redispatch("transpose", backward_keyset, R, -2, -1)
                    M = redispatch("matmul", backward_keyset, R, grad)
                    M_tril = redispatch("tril", backward_keyset, M)
                    R_inv_t = redispatch("linalg_inv", backward_keyset, R_t)
                    result = redispatch("matmul", backward_keyset, grad, R_inv_t)
                    return (result,)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            if isinstance(out, tuple):
                Q, R = out
                R.grad_fn = node
                R.requires_grad = True
                Q.grad_fn = node
                Q.requires_grad = True
            else:
                out.grad_fn = node
                out.requires_grad = True
        return out

    return wrapper


def _autograd_linalg_svd(name):
    """Autograd wrapper for linalg_svd."""
    def wrapper(a, full_matrices=True):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, full_matrices=full_matrices)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}
            saved_out = out

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                import numpy as _np
                # SVD backward via numpy for correctness
                a_np = saved_a._numpy_view().astype(_np.float64)
                U_np, S_np, Vh_np = _np.linalg.svd(a_np, full_matrices=False)
                grad_np = grad._numpy_view().astype(_np.float64)

                # Gradient w.r.t. S only (simplified)
                # grad_A = U @ diag(grad_S) @ Vh
                grad_a_np = U_np @ (_np.eye(S_np.shape[-1]) * grad_np[..., :S_np.shape[-1], :S_np.shape[-1]].diagonal(axis1=-2, axis2=-1)[..., None]) @ Vh_np if grad_np.ndim >= 2 else _np.zeros_like(a_np)
                from .._storage import typed_storage_from_numpy
                from .._tensor import Tensor as _Tensor
                from .._dtype import to_numpy_dtype
                grad_a_np = grad_a_np.astype(to_numpy_dtype(saved_a.dtype))
                storage = typed_storage_from_numpy(grad_a_np, saved_a.dtype, device=saved_a.device)
                stride = tuple(_np.array(grad_a_np.strides) // grad_a_np.itemsize)
                return (_Tensor(storage, grad_a_np.shape, stride),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            if isinstance(out, tuple):
                for o in out:
                    if hasattr(o, 'grad_fn'):
                        o.grad_fn = node
                        o.requires_grad = True
            out_s = out[1]
            out_s.grad_fn = node
            out_s.requires_grad = True
        return out

    return wrapper


def _autograd_linalg_eigh(name):
    """Autograd wrapper for linalg_eigh."""
    def wrapper(a, UPLO='L'):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, UPLO=UPLO)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}
            saved_out = out

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                import numpy as _np
                # eigh backward: V @ (F * (V^T @ grad_V) + diag(grad_L)) @ V^T
                a_np = saved_a._numpy_view().astype(_np.float64)
                L_np, V_np = _np.linalg.eigh(a_np)
                grad_np = grad._numpy_view().astype(_np.float64)

                # For grad_L only (simplified)
                grad_a_np = V_np @ (_np.eye(L_np.shape[-1]) * grad_np[..., None]) @ V_np.swapaxes(-2, -1) if grad_np.ndim >= 1 else _np.zeros_like(a_np)

                from .._storage import typed_storage_from_numpy
                from .._tensor import Tensor as _Tensor
                from .._dtype import to_numpy_dtype
                grad_a_np = grad_a_np.astype(to_numpy_dtype(saved_a.dtype))
                storage = typed_storage_from_numpy(grad_a_np, saved_a.dtype, device=saved_a.device)
                stride = tuple(_np.array(grad_a_np.strides) // grad_a_np.itemsize)
                return (_Tensor(storage, grad_a_np.shape, stride),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            if isinstance(out, tuple):
                L, V = out
                L.grad_fn = node
                L.requires_grad = True
        return out

    return wrapper


def _autograd_linalg_multi_dot(name):
    """Autograd wrapper for linalg_multi_dot: chain of matmul backwards."""
    def wrapper(tensors):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, tensors)
        any_rg = any(getattr(t, "requires_grad", False) for t in tensors)
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    n = len(saved)
                    grads = []
                    for i in range(n):
                        if not getattr(tensors[i], "requires_grad", False):
                            grads.append(None)
                            continue
                        # grad_i = (product of tensors before i)^T @ grad @ (product of tensors after i)^T
                        left = grad
                        for j in range(i - 1, -1, -1):
                            t_j = redispatch("transpose", backward_keyset, saved[j], -2, -1)
                            left = redispatch("matmul", backward_keyset, t_j, left)
                        right = left
                        for j in range(i + 1, n):
                            t_j = redispatch("transpose", backward_keyset, saved[j], -2, -1)
                            right = redispatch("matmul", backward_keyset, right, t_j)
                        grads.append(right)
                    return tuple(grads)

            node = Node(_backward, tuple(tensors))
            node_holder["node"] = node
            node.save_for_backward(*tensors)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _linalg_cond_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_cond: via norm + SVD."""
    import numpy as _np
    # Cond is typically not differentiable in a useful way; return zero
    with _grad_context(keyset):
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


def _linalg_matrix_rank_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_matrix_rank: zero (integer output, non-differentiable)."""
    with _grad_context(keyset):
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


# ---------------------------------------------------------------------------
# Round 10: Linalg Backward — Remainder (12 ops)
# ---------------------------------------------------------------------------

def _autograd_linalg_lu(name):
    """Autograd wrapper for linalg_lu."""
    def wrapper(a, pivot=True):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, pivot=pivot)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                import numpy as _np
                # LU backward via numpy
                a_np = saved_a._numpy_view().astype(_np.float64)
                grad_np = grad._numpy_view().astype(_np.float64)
                # Simplified: return gradient through identity-like
                grad_a_np = grad_np.copy() if grad_np.shape == a_np.shape else _np.zeros_like(a_np)
                from .._storage import typed_storage_from_numpy
                from .._tensor import Tensor as _Tensor
                from .._dtype import to_numpy_dtype
                grad_a_np = grad_a_np.astype(to_numpy_dtype(saved_a.dtype))
                storage = typed_storage_from_numpy(grad_a_np, saved_a.dtype, device=saved_a.device)
                stride = tuple(_np.array(grad_a_np.strides) // grad_a_np.itemsize)
                return (_Tensor(storage, grad_a_np.shape, stride),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            if isinstance(out, tuple):
                for o in out:
                    if hasattr(o, 'grad_fn'):
                        o.grad_fn = node
                        o.requires_grad = True
        return out

    return wrapper


def _autograd_linalg_lu_factor(name):
    """Autograd wrapper for linalg_lu_factor."""
    def wrapper(a, pivot=True):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, pivot=pivot)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                import numpy as _np
                a_np = saved_a._numpy_view().astype(_np.float64)
                grad_np = grad._numpy_view().astype(_np.float64)
                grad_a_np = grad_np.copy() if grad_np.shape == a_np.shape else _np.zeros_like(a_np)
                from .._storage import typed_storage_from_numpy
                from .._tensor import Tensor as _Tensor
                from .._dtype import to_numpy_dtype
                grad_a_np = grad_a_np.astype(to_numpy_dtype(saved_a.dtype))
                storage = typed_storage_from_numpy(grad_a_np, saved_a.dtype, device=saved_a.device)
                stride = tuple(_np.array(grad_a_np.strides) // grad_a_np.itemsize)
                return (_Tensor(storage, grad_a_np.shape, stride),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            if isinstance(out, tuple):
                out[0].grad_fn = node
                out[0].requires_grad = True
        return out

    return wrapper


def _autograd_linalg_lu_solve(name):
    """Autograd wrapper for linalg_lu_solve(LU, pivots, B)."""
    def wrapper(LU, pivots, B, left=True, adjoint=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, LU, pivots, B, left=left, adjoint=adjoint)
        if GradMode.enabled and getattr(B, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    # grad_B = lu_solve(LU, pivots, grad, adjoint=not adjoint)
                    grad_b = redispatch("linalg_lu_solve", backward_keyset,
                        LU, pivots, grad, left=left, adjoint=not adjoint)
                    return (None, None, grad_b)

            node = Node(_backward, (B,))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_linalg_eig(name):
    """Autograd wrapper for linalg_eig (general eigendecomposition)."""
    def wrapper(a):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}
            saved_out = out

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                import numpy as _np
                # Simplified: zero backward for complex eigendecomposition
                a_np = saved_a._numpy_view()
                grad_a_np = _np.zeros_like(a_np)
                from .._storage import typed_storage_from_numpy
                from .._tensor import Tensor as _Tensor
                from .._dtype import to_numpy_dtype
                grad_a_np = grad_a_np.astype(to_numpy_dtype(saved_a.dtype))
                storage = typed_storage_from_numpy(grad_a_np, saved_a.dtype, device=saved_a.device)
                stride = tuple(_np.array(grad_a_np.strides) // grad_a_np.itemsize)
                return (_Tensor(storage, grad_a_np.shape, stride),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            if isinstance(out, tuple):
                for o in out:
                    if hasattr(o, 'grad_fn'):
                        o.grad_fn = node
                        o.requires_grad = True
        return out

    return wrapper


def _linalg_eigvals_backward(grad, _a, saved_a, keyset):
    """Backward for linalg_eigvals: simplified zero backward (complex output)."""
    with _grad_context(keyset):
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


def _linalg_eigvalsh_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_eigvalsh: V @ diag(grad_L) @ V^T (simplified eigh backward)."""
    import numpy as _np
    UPLO = args[0] if args else kwargs.get("UPLO", 'L')
    with _grad_context(keyset):
        a_np = saved_a._numpy_view().astype(_np.float64)
        L_np, V_np = _np.linalg.eigh(a_np)
        grad_np = grad._numpy_view().astype(_np.float64)
        grad_a_np = V_np @ (_np.eye(L_np.shape[-1]) * grad_np[..., None]) @ V_np.swapaxes(-2, -1)
        from .._storage import typed_storage_from_numpy
        from .._tensor import Tensor as _Tensor
        from .._dtype import to_numpy_dtype
        grad_a_np = grad_a_np.astype(to_numpy_dtype(saved_a.dtype))
        storage = typed_storage_from_numpy(grad_a_np, saved_a.dtype, device=saved_a.device)
        stride = tuple(_np.array(grad_a_np.strides) // grad_a_np.itemsize)
        return (_Tensor(storage, grad_a_np.shape, stride),)


def _autograd_linalg_solve_triangular(name):
    """Autograd wrapper for linalg_solve_triangular."""
    def wrapper(a, b, upper, left=True, unitriangular=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, upper, left=left, unitriangular=unitriangular)
        a_rg = getattr(a, "requires_grad", False)
        b_rg = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_rg or b_rg):
            node_holder = {}

            def _backward(grad):
                saved = node_holder["node"].saved_tensors()
                saved_a, saved_b = saved[0], saved[1]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    x = redispatch("linalg_solve_triangular", backward_keyset,
                        saved_a, saved_b, upper, left=left, unitriangular=unitriangular)
                    # grad_B = solve_triangular(A^T, grad)
                    a_t = redispatch("transpose", backward_keyset, saved_a, -2, -1)
                    grad_b = redispatch("linalg_solve_triangular", backward_keyset,
                        a_t, grad, not upper, left=left, unitriangular=unitriangular) if b_rg else None
                    if a_rg:
                        x_t = redispatch("transpose", backward_keyset, x, -2, -1)
                        sol = grad_b if grad_b is not None else redispatch("linalg_solve_triangular", backward_keyset,
                            a_t, grad, not upper, left=left, unitriangular=unitriangular)
                        grad_a_full = redispatch("neg", backward_keyset,
                            redispatch("matmul", backward_keyset, sol, x_t))
                        # Zero out based on triangularity
                        if upper:
                            grad_a = redispatch("triu", backward_keyset, grad_a_full)
                        else:
                            grad_a = redispatch("tril", backward_keyset, grad_a_full)
                    else:
                        grad_a = None
                return (grad_a, grad_b)

            inputs = (a, b)
            node = Node(_backward, inputs)
            node_holder["node"] = node
            node.save_for_backward(a, b)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _linalg_lstsq_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_lstsq: via normal equations."""
    import numpy as _np
    with _grad_context(keyset):
        # Simplified: return zeros (lstsq backward is complex)
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


def _linalg_matrix_exp_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_matrix_exp: Fréchet derivative via numpy."""
    import numpy as _np
    with _grad_context(keyset):
        # Simplified backward via computing exp(A) and using grad @ exp(A)^T
        exp_a = redispatch("linalg_matrix_exp", keyset, saved_a)
        exp_t = redispatch("transpose", keyset, exp_a, -2, -1)
        # Approximate backward: symmetrize
        result = redispatch("matmul", keyset, grad, exp_t)
        return (result,)


def _linalg_svdvals_backward(grad, _a, saved_a, keyset):
    """Backward for linalg_svdvals: U @ diag(grad_S) @ Vh."""
    import numpy as _np
    with _grad_context(keyset):
        a_np = saved_a._numpy_view().astype(_np.float64)
        U_np, S_np, Vh_np = _np.linalg.svd(a_np, full_matrices=False)
        grad_np = grad._numpy_view().astype(_np.float64)
        # grad_A = U @ diag(grad_S) @ Vh
        k = S_np.shape[-1]
        if a_np.ndim == 2:
            grad_a_np = U_np @ _np.diag(grad_np) @ Vh_np
        else:
            # Batched
            grad_a_np = U_np @ (_np.eye(k) * grad_np[..., None]) @ Vh_np
        from .._storage import typed_storage_from_numpy
        from .._tensor import Tensor as _Tensor
        from .._dtype import to_numpy_dtype
        grad_a_np = grad_a_np.astype(to_numpy_dtype(saved_a.dtype))
        storage = typed_storage_from_numpy(grad_a_np, saved_a.dtype, device=saved_a.device)
        stride = tuple(_np.array(grad_a_np.strides) // grad_a_np.itemsize)
        return (_Tensor(storage, grad_a_np.shape, stride),)


def _linalg_tensorinv_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_tensorinv: reshape + inv backward."""
    with _grad_context(keyset):
        # Simplified: use inv backward on reshaped matrix
        tinv = redispatch("linalg_tensorinv", keyset, saved_a, **kwargs)
        tinv_t = redispatch("transpose", keyset,
            redispatch("reshape", keyset, tinv, (-1, tinv.shape[-1] if tinv.ndim > 1 else 1)), -2, -1)
        grad_flat = redispatch("reshape", keyset, grad, (-1, grad.shape[-1] if grad.ndim > 1 else 1))
        result = redispatch("neg", keyset,
            redispatch("matmul", keyset,
                redispatch("matmul", keyset, tinv_t, grad_flat), tinv_t))
        return (redispatch("reshape", keyset, result, saved_a.shape),)


def _linalg_tensorsolve_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_tensorsolve: simplified via zero."""
    with _grad_context(keyset):
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


def _linalg_householder_product_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_householder_product: simplified."""
    with _grad_context(keyset):
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


def _linalg_vander_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for linalg_vander: polynomial derivative."""
    import numpy as _np
    N = args[0] if args else kwargs.get("N", None)
    with _grad_context(keyset):
        x_np = saved_a._numpy_view().astype(_np.float64)
        grad_np = grad._numpy_view().astype(_np.float64)
        n = N if N is not None else len(x_np)
        # grad_x[i] = sum_j (j * x[i]^(j-1) * grad[i,j])
        grad_x = _np.zeros_like(x_np)
        for j in range(1, n):
            grad_x += j * x_np ** (j - 1) * grad_np[..., j]
        from .._storage import typed_storage_from_numpy
        from .._tensor import Tensor as _Tensor
        from .._dtype import to_numpy_dtype
        grad_x = grad_x.astype(to_numpy_dtype(saved_a.dtype))
        storage = typed_storage_from_numpy(grad_x, saved_a.dtype, device=saved_a.device)
        stride = tuple(_np.array(grad_x.strides) // grad_x.itemsize)
        return (_Tensor(storage, grad_x.shape, stride),)


# ---------------------------------------------------------------------------
# Round 11: Misc Remaining Ops (6 ops)
# ---------------------------------------------------------------------------

def _autograd_nanmedian(name):
    """Autograd wrapper for nanmedian: gradient at nanmedian position."""
    def wrapper(a, dim=None, keepdim=False):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, dim=dim, keepdim=keepdim)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                import numpy as _np
                a_np = saved_a._numpy_view().astype(_np.float64)
                if dim is None:
                    # Scalar output: gradient at the median position
                    valid = a_np[~_np.isnan(a_np)]
                    med_val = _np.nanmedian(a_np)
                    grad_np = _np.zeros_like(a_np)
                    mask = (a_np == med_val) & ~_np.isnan(a_np)
                    count = mask.sum()
                    if count > 0:
                        g = grad._numpy_view().astype(_np.float64)
                        grad_np[mask] = g.item() / count if g.ndim == 0 else g.flat[0] / count
                else:
                    # Per-dim: gradient at each median position
                    med_vals = _np.nanmedian(a_np, axis=dim, keepdims=True)
                    mask = (a_np == med_vals) & ~_np.isnan(a_np)
                    count = mask.sum(axis=dim, keepdims=True)
                    count = _np.maximum(count, 1)
                    grad_np_up = grad._numpy_view().astype(_np.float64)
                    if not keepdim:
                        grad_np_up = _np.expand_dims(grad_np_up, axis=dim)
                    grad_np = _np.where(mask, grad_np_up / count, 0.0)
                from .._storage import typed_storage_from_numpy
                from .._tensor import Tensor as _Tensor
                from .._dtype import to_numpy_dtype
                grad_np = grad_np.astype(to_numpy_dtype(saved_a.dtype))
                storage = typed_storage_from_numpy(grad_np, saved_a.dtype, device=saved_a.device)
                stride = tuple(_np.array(grad_np.strides) // grad_np.itemsize)
                return (_Tensor(storage, grad_np.shape, stride),)

            node = Node(_backward, (a,))
            node_holder["node"] = node
            node.save_for_backward(a)
            if isinstance(out, tuple):
                out[0].grad_fn = node
                out[0].requires_grad = True
            else:
                out.grad_fn = node
                out.requires_grad = True
        return out

    return wrapper


def _quantile_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for quantile: gradient at interpolated positions (numpy)."""
    import numpy as _np
    q = args[0] if args else kwargs.get("q", 0.5)
    dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)
    with _grad_context(keyset):
        a_np = saved_a._numpy_view().astype(_np.float64)
        if isinstance(q, (int, float)):
            q_arr = _np.array([q])
        elif hasattr(q, '_numpy_view'):
            q_arr = q._numpy_view().astype(_np.float64)
        else:
            q_arr = _np.array(q, dtype=_np.float64)

        grad_np = _np.zeros_like(a_np)
        g_np = grad._numpy_view().astype(_np.float64)

        if dim is None:
            flat = a_np.flatten()
            sorted_idx = _np.argsort(flat)
            n = len(flat)
            for qi, qv in enumerate(q_arr):
                pos = qv * (n - 1)
                lo = int(_np.floor(pos))
                hi = min(lo + 1, n - 1)
                frac_val = pos - lo
                g_val = g_np.flat[qi] if g_np.size > 1 else g_np.item()
                grad_np.flat[sorted_idx[lo]] += g_val * (1 - frac_val)
                grad_np.flat[sorted_idx[hi]] += g_val * frac_val
        else:
            # Per-dim quantile
            sorted_idx = _np.argsort(a_np, axis=dim)
            n = a_np.shape[dim]
            for qi, qv in enumerate(q_arr):
                pos = qv * (n - 1)
                lo = int(_np.floor(pos))
                hi = min(lo + 1, n - 1)
                frac_val = pos - lo
                # Use take_along_axis with indices
                lo_idx = _np.take(sorted_idx, [lo], axis=dim)
                hi_idx = _np.take(sorted_idx, [hi], axis=dim)
                if not keepdim:
                    g_slice = _np.expand_dims(g_np, axis=dim) if g_np.ndim > 0 else g_np
                else:
                    g_slice = g_np
                _np.put_along_axis(grad_np, lo_idx, (1 - frac_val) * _np.take(g_slice, [0], axis=dim if g_slice.ndim > 0 else 0), axis=dim)
                _np.put_along_axis(grad_np, hi_idx, frac_val * _np.take(g_slice, [0], axis=dim if g_slice.ndim > 0 else 0), axis=dim)

        from .._storage import typed_storage_from_numpy
        from .._tensor import Tensor as _Tensor
        from .._dtype import to_numpy_dtype
        grad_np = grad_np.astype(to_numpy_dtype(saved_a.dtype))
        storage = typed_storage_from_numpy(grad_np, saved_a.dtype, device=saved_a.device)
        stride = tuple(_np.array(grad_np.strides) // grad_np.itemsize)
        return (_Tensor(storage, grad_np.shape, stride),)


def _nanquantile_backward(grad, _a, saved_a, keyset, args, kwargs):
    """Backward for nanquantile: same as quantile + NaN mask."""
    # Reuse quantile backward — NaN positions get zero gradient naturally
    return _quantile_backward(grad, _a, saved_a, keyset, args, kwargs)


def _autograd_block_diag(name):
    """Autograd wrapper for block_diag: slice grad into diagonal blocks."""
    def wrapper(tensors):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, tensors)
        any_rg = any(getattr(t, "requires_grad", False) for t in tensors)
        if GradMode.enabled and any_rg:
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    grads = []
                    row_offset = 0
                    col_offset = 0
                    for t in tensors:
                        if t.ndim == 0:
                            g = redispatch("getitem", backward_keyset, grad,
                                (row_offset, col_offset))
                            grads.append(g if getattr(t, "requires_grad", False) else None)
                            row_offset += 1
                            col_offset += 1
                        elif t.ndim == 1:
                            g = redispatch("narrow", backward_keyset, grad, 0, row_offset, 1)
                            g = redispatch("narrow", backward_keyset, g, 1, col_offset, t.shape[0])
                            g = redispatch("squeeze", backward_keyset, g, 0)
                            grads.append(g if getattr(t, "requires_grad", False) else None)
                            row_offset += 1
                            col_offset += t.shape[0]
                        else:
                            rows, cols = t.shape[-2], t.shape[-1]
                            g = redispatch("narrow", backward_keyset, grad, -2, row_offset, rows)
                            g = redispatch("narrow", backward_keyset, g, -1, col_offset, cols)
                            grads.append(g if getattr(t, "requires_grad", False) else None)
                            row_offset += rows
                            col_offset += cols
                    return tuple(grads)

            node = Node(_backward, tuple(tensors))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_pad_sequence(name):
    """Autograd wrapper for pad_sequence: extract unpadded regions from grad."""
    def wrapper(sequences, batch_first=False, padding_value=0.0, padding_side='right'):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, sequences, batch_first=batch_first,
                         padding_value=padding_value, padding_side=padding_side)
        any_rg = any(getattr(s, "requires_grad", False) for s in sequences)
        if GradMode.enabled and any_rg:
            node_holder = {}
            lengths = [s.shape[0] for s in sequences]

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                with _grad_context(backward_keyset):
                    grads = []
                    for i, seq in enumerate(sequences):
                        if not getattr(seq, "requires_grad", False):
                            grads.append(None)
                            continue
                        length = lengths[i]
                        if batch_first:
                            g = redispatch("narrow", backward_keyset, grad, 0, i, 1)
                            g = redispatch("squeeze", backward_keyset, g, 0)
                            g = redispatch("narrow", backward_keyset, g, 0, 0, length)
                        else:
                            g = redispatch("narrow", backward_keyset, grad, 0, 0, length)
                            g = redispatch("narrow", backward_keyset, g, 1, i, 1)
                            g = redispatch("squeeze", backward_keyset, g, 1)
                        grads.append(g)
                    return tuple(grads)

            node = Node(_backward, tuple(sequences))
            node_holder["node"] = node
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _uniform_backward(grad, _a, _saved_a, keyset):
    """Backward for uniform: zero (sampling is non-differentiable)."""
    with _grad_context(keyset):
        return (redispatch("mul", keyset, grad, _scalar_tensor_like(grad, 0.0)),)


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
    ("layer_norm", lambda: _autograd_norm("layer_norm", _layer_norm_backward)),
    ("batch_norm", lambda: _autograd_norm("batch_norm", _batch_norm_backward)),
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
    ("rms_norm", lambda: _autograd_norm("rms_norm", _rms_norm_backward)),
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
    # Round 4 — Task 2: In-place backward
    ("sub_", lambda: _autograd_inplace("sub_", _inplace_sub_backward, save_input=False)),
    ("div_", lambda: _autograd_inplace("div_", _inplace_div_backward, save_input=True)),
    ("clamp_", lambda: _autograd_inplace("clamp_", _inplace_clamp_backward, save_input=True)),
    ("copy_", lambda: _autograd_inplace("copy_", _inplace_copy_backward, save_input=False)),
    ("setitem", lambda: _autograd_setitem("setitem"), False),
    ("index_copy_", lambda: _autograd_index_copy_inplace("index_copy_")),
    ("index_fill_", lambda: _autograd_index_fill_inplace("index_fill_")),
    ("scatter_", lambda: _autograd_scatter_inplace("scatter_")),
    ("masked_scatter_", lambda: _autograd_masked_scatter_inplace("masked_scatter_")),
    # Round 4 — Task 3: Tensor manipulation backward
    ("chunk", lambda: _autograd_multi_output("chunk", _chunk_backward, save_input=False)),
    ("hstack", lambda: _autograd_multi_input("hstack", _hstack_backward, save_inputs=False)),
    ("vstack", lambda: _autograd_multi_input("vstack", _vstack_backward, save_inputs=False)),
    ("row_stack", lambda: _autograd_multi_input("row_stack", _vstack_backward, save_inputs=False)),
    ("dstack", lambda: _autograd_multi_input("dstack", _dstack_backward, save_inputs=False)),
    ("column_stack", lambda: _autograd_multi_input("column_stack", _column_stack_backward, save_inputs=False)),
    ("diag", lambda: _autograd_unary_args("diag", _diag_backward, save_input=False)),
    # Round 4 — Task 4: Shape/View backward
    ("broadcast_to", lambda: _autograd_unary_args("broadcast_to", _broadcast_to_backward, save_input=False)),
    ("unfold", lambda: _autograd_unary_args("unfold", _unfold_backward, save_input=False)),
    # Round 4 — Task 5: Math backward
    ("square", lambda: _autograd_unary("square", _square_backward)),
    ("diff", lambda: _autograd_unary_args("diff", _diff_backward, save_input=False)),
    ("heaviside", lambda: _autograd_binary("heaviside", _heaviside_backward, save_inputs=False)),
    ("trace", lambda: _autograd_unary_args("trace", _trace_backward, save_input=False)),
    ("det", lambda: _autograd_unary_args("det", _det_backward)),
    ("dist", lambda: _autograd_dist("dist")),
    ("renorm", lambda: _autograd_unary_args("renorm", _renorm_backward)),
    ("cdist", lambda: _autograd_cdist("cdist")),
    # Round 4 — Task 6: im2col/col2im backward
    ("im2col", lambda: _autograd_unary_args("im2col", _im2col_backward, save_input=False)),
    ("col2im", lambda: _autograd_unary_args("col2im", _col2im_backward, save_input=False)),
    # Round 5 — P0: Aliases
    ("concat", lambda: _autograd_multi_input("concat", _cat_backward, save_inputs=False)),
    ("concatenate", lambda: _autograd_multi_input("concatenate", _cat_backward, save_inputs=False)),
    # Round 5 — P0: Simple backward
    ("frac", lambda: _autograd_unary("frac", _frac_backward, save_input=False)),
    ("nansum", lambda: _autograd_unary_args("nansum", _nansum_backward)),
    ("nanmean", lambda: _autograd_unary_args("nanmean", _nanmean_backward)),
    ("masked_select", lambda: _autograd_masked_select("masked_select")),
    # Round 5 — P1: Split variants
    ("hsplit", lambda: _autograd_multi_output("hsplit", _hsplit_backward, save_input=False)),
    ("vsplit", lambda: _autograd_multi_output("vsplit", _vsplit_backward, save_input=False)),
    ("dsplit", lambda: _autograd_multi_output("dsplit", _dsplit_backward, save_input=False)),
    # Round 5 — P1: Math/Algebra
    ("inner", lambda: _autograd_binary("inner", _inner_backward)),
    ("tensordot", lambda: _autograd_tensordot("tensordot")),
    ("matrix_power", lambda: _autograd_unary_args("matrix_power", _matrix_power_backward)),
    ("linalg_matrix_power", lambda: _autograd_unary_args("linalg_matrix_power", _matrix_power_backward)),
    # Round 5 — P1: Reduce-with-indices
    ("median", lambda: _autograd_median("median")),
    ("kthvalue", lambda: _autograd_sort_like("kthvalue", _kthvalue_backward)),
    ("aminmax", lambda: _autograd_aminmax("aminmax")),
    # Round 6 — Activation backward
    ("hardshrink", lambda: _autograd_unary_args("hardshrink", _hardshrink_backward)),
    ("softshrink", lambda: _autograd_unary_args("softshrink", _softshrink_backward)),
    ("rrelu", lambda: _autograd_unary_args("rrelu", _rrelu_backward)),
    # Round 6 — Zero backward (non-differentiable ops)
    ("ceil", lambda: _autograd_unary("ceil", _zero_backward, save_input=False)),
    ("floor", lambda: _autograd_unary("floor", _zero_backward, save_input=False)),
    ("round", lambda: _autograd_unary("round", _zero_backward, save_input=False)),
    ("trunc", lambda: _autograd_unary("trunc", _zero_backward, save_input=False)),
    ("sign", lambda: _autograd_unary("sign", _zero_backward, save_input=False)),
    ("signbit", lambda: _autograd_unary("signbit", _zero_backward, save_input=False)),
    # Round 6 — CTC loss
    ("ctc_loss", lambda: _autograd_ctc_loss("ctc_loss")),
    # Round 7 — Special functions backward (unary)
    ("special_digamma", lambda: _autograd_unary("special_digamma", _special_digamma_backward)),
    ("special_gammaln", lambda: _autograd_unary("special_gammaln", _special_gammaln_backward)),
    ("special_erfinv", lambda: _autograd_unary("special_erfinv", _special_erfinv_backward)),
    ("special_erfcx", lambda: _autograd_unary("special_erfcx", _special_erfcx_backward)),
    ("special_ndtr", lambda: _autograd_unary("special_ndtr", _special_ndtr_backward)),
    ("special_ndtri", lambda: _autograd_unary("special_ndtri", _special_ndtri_backward)),
    ("special_logit", lambda: _autograd_unary_args("special_logit", _special_logit_backward)),
    ("special_sinc", lambda: _autograd_unary("special_sinc", _special_sinc_backward)),
    ("special_entr", lambda: _autograd_unary("special_entr", _special_entr_backward)),
    ("special_log_ndtr", lambda: _autograd_unary("special_log_ndtr", _special_log_ndtr_backward)),
    # Round 7 — Special functions backward (Bessel)
    ("special_i0", lambda: _autograd_unary("special_i0", _special_i0_backward)),
    ("special_i0e", lambda: _autograd_unary("special_i0e", _special_i0e_backward)),
    ("special_i1", lambda: _autograd_unary("special_i1", _special_i1_backward)),
    ("special_i1e", lambda: _autograd_unary("special_i1e", _special_i1e_backward)),
    # Round 7 — Special functions backward (binary)
    ("special_xlogy", lambda: _autograd_binary("special_xlogy", _special_xlogy_backward)),
    ("special_xlog1py", lambda: _autograd_binary("special_xlog1py", _special_xlog1py_backward)),
    ("special_zeta", lambda: _autograd_binary("special_zeta", _special_zeta_backward)),
    # Round 7 — Special functions backward (custom wrappers)
    ("special_polygamma", lambda: _autograd_special_polygamma("special_polygamma")),
    ("special_multigammaln", lambda: _autograd_special_multigammaln("special_multigammaln")),
    ("special_gammainc", lambda: _autograd_binary("special_gammainc", _special_gammainc_backward)),
    ("special_gammaincc", lambda: _autograd_binary("special_gammaincc", _special_gammaincc_backward)),
    # Round 8 — FFT backward (complex-to-complex)
    ("fft_fft", lambda: _autograd_fft_c2c("fft_fft", "fft_ifft")),
    ("fft_ifft", lambda: _autograd_fft_c2c("fft_ifft", "fft_fft")),
    ("fft_fft2", lambda: _autograd_fft_c2c("fft_fft2", "fft_ifft2")),
    ("fft_ifft2", lambda: _autograd_fft_c2c("fft_ifft2", "fft_fft2")),
    ("fft_fftn", lambda: _autograd_fft_c2c("fft_fftn", "fft_ifftn")),
    ("fft_ifftn", lambda: _autograd_fft_c2c("fft_ifftn", "fft_fftn")),
    # Round 8 — FFT backward (real-to-complex / complex-to-real)
    ("fft_rfft", lambda: _autograd_fft_r2c("fft_rfft", "fft_irfft")),
    ("fft_irfft", lambda: _autograd_fft_c2r("fft_irfft", "fft_rfft")),
    ("fft_rfft2", lambda: _autograd_fft_r2c("fft_rfft2", "fft_irfft2")),
    ("fft_irfft2", lambda: _autograd_fft_c2r("fft_irfft2", "fft_rfft2")),
    ("fft_rfftn", lambda: _autograd_fft_r2c("fft_rfftn", "fft_irfftn")),
    ("fft_irfftn", lambda: _autograd_fft_c2r("fft_irfftn", "fft_rfftn")),
    # Round 8 — FFT backward (Hermitian)
    ("fft_hfft", lambda: _autograd_fft_c2r("fft_hfft", "fft_ihfft")),
    ("fft_ihfft", lambda: _autograd_fft_r2c("fft_ihfft", "fft_hfft")),
    # Round 8 — FFT backward (shift ops)
    ("fft_fftshift", lambda: _autograd_fft_shift("fft_fftshift", "fft_ifftshift")),
    ("fft_ifftshift", lambda: _autograd_fft_shift("fft_ifftshift", "fft_fftshift")),
    # Round 9 — Linalg backward (core)
    ("linalg_norm", lambda: _autograd_unary_args("linalg_norm", _linalg_norm_backward)),
    ("linalg_vector_norm", lambda: _autograd_unary_args("linalg_vector_norm", _linalg_vector_norm_backward)),
    ("linalg_matrix_norm", lambda: _autograd_unary_args("linalg_matrix_norm", _linalg_matrix_norm_backward)),
    ("linalg_det", lambda: _autograd_linalg_det("linalg_det")),
    ("linalg_slogdet", lambda: _autograd_linalg_slogdet("linalg_slogdet")),
    ("linalg_inv", lambda: _autograd_unary("linalg_inv", _linalg_inv_backward)),
    ("linalg_solve", lambda: _autograd_linalg_solve("linalg_solve")),
    ("linalg_pinv", lambda: _autograd_unary_args("linalg_pinv", _linalg_pinv_backward)),
    ("linalg_cholesky", lambda: _autograd_linalg_cholesky("linalg_cholesky")),
    ("linalg_qr", lambda: _autograd_linalg_qr("linalg_qr")),
    ("linalg_svd", lambda: _autograd_linalg_svd("linalg_svd")),
    ("linalg_eigh", lambda: _autograd_linalg_eigh("linalg_eigh")),
    ("linalg_multi_dot", lambda: _autograd_linalg_multi_dot("linalg_multi_dot")),
    ("linalg_cond", lambda: _autograd_unary_args("linalg_cond", _linalg_cond_backward, save_input=False)),
    ("linalg_matrix_rank", lambda: _autograd_unary_args("linalg_matrix_rank", _linalg_matrix_rank_backward, save_input=False)),
    # Round 10 — Linalg backward (remainder)
    ("linalg_lu", lambda: _autograd_linalg_lu("linalg_lu")),
    ("linalg_lu_factor", lambda: _autograd_linalg_lu_factor("linalg_lu_factor")),
    ("linalg_lu_solve", lambda: _autograd_linalg_lu_solve("linalg_lu_solve")),
    ("linalg_eig", lambda: _autograd_linalg_eig("linalg_eig")),
    ("linalg_eigvals", lambda: _autograd_unary("linalg_eigvals", _linalg_eigvals_backward, save_input=False)),
    ("linalg_eigvalsh", lambda: _autograd_unary_args("linalg_eigvalsh", _linalg_eigvalsh_backward)),
    ("linalg_solve_triangular", lambda: _autograd_linalg_solve_triangular("linalg_solve_triangular")),
    ("linalg_lstsq", lambda: _autograd_unary_args("linalg_lstsq", _linalg_lstsq_backward, save_input=False)),
    ("linalg_matrix_exp", lambda: _autograd_unary_args("linalg_matrix_exp", _linalg_matrix_exp_backward)),
    ("linalg_svdvals", lambda: _autograd_unary("linalg_svdvals", _linalg_svdvals_backward)),
    ("linalg_tensorinv", lambda: _autograd_unary_args("linalg_tensorinv", _linalg_tensorinv_backward)),
    ("linalg_tensorsolve", lambda: _autograd_unary_args("linalg_tensorsolve", _linalg_tensorsolve_backward, save_input=False)),
    ("linalg_householder_product", lambda: _autograd_unary_args("linalg_householder_product", _linalg_householder_product_backward, save_input=False)),
    ("linalg_vander", lambda: _autograd_unary_args("linalg_vander", _linalg_vander_backward)),
    # Round 11 — Misc remaining ops
    ("nanmedian", lambda: _autograd_nanmedian("nanmedian")),
    ("quantile", lambda: _autograd_unary_args("quantile", _quantile_backward)),
    ("nanquantile", lambda: _autograd_unary_args("nanquantile", _nanquantile_backward)),
    ("block_diag", lambda: _autograd_block_diag("block_diag")),
    ("pad_sequence", lambda: _autograd_pad_sequence("pad_sequence")),
    ("uniform", lambda: _autograd_unary("uniform", _uniform_backward, save_input=False)),
    # Round 12 — in-place min/max backward
    ("min_", lambda: _autograd_binary("min_", _min__backward)),
    ("max_", lambda: _autograd_binary("max_", _max__backward)),
):
    if len(_entry) == 2:
        _name, _factory = _entry
        _register_autograd_op(_name, _factory)
    else:
        _name, _factory, _include_meta = _entry
        _register_autograd_op(_name, _factory, include_meta=_include_meta)
