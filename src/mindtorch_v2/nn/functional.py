"""Functional interface for nn operations."""


def linear(input, weight, bias=None):
    from .._functional import matmul, add
    output = matmul(input, weight.t() if hasattr(weight, 't') else weight)
    if bias is not None:
        output = add(output, bias)
    return output


def relu(input, inplace=False):
    from .._functional import relu as _relu
    return _relu(input)


def sigmoid(input):
    from .._functional import sigmoid as _sigmoid
    return _sigmoid(input)


def tanh(input):
    from .._functional import tanh as _tanh
    return _tanh(input)


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    from .._dispatch import dispatch
    if dim is None:
        dim = -1
    return dispatch("softmax", input.device.type, input, dim)


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    from .._dispatch import dispatch
    if dim is None:
        dim = -1
    return dispatch("log_softmax", input.device.type, input, dim)


def gelu(input, approximate='none'):
    from .._dispatch import dispatch
    return dispatch("gelu", input.device.type, input)


def silu(input, inplace=False):
    from .._dispatch import dispatch
    return dispatch("silu", input.device.type, input)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    from .._dispatch import dispatch
    return dispatch("leaky_relu", input.device.type, input, negative_slope)


def elu(input, alpha=1.0, inplace=False):
    from .._dispatch import dispatch
    return dispatch("elu", input.device.type, input, alpha)


def mish(input, inplace=False):
    from .._dispatch import dispatch
    return dispatch("mish", input.device.type, input)


def prelu(input, weight):
    from .._dispatch import dispatch
    return dispatch("prelu", input.device.type, input, weight)


def dropout(input, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return input
    from .._dispatch import dispatch
    return dispatch("dropout", input.device.type, input, p, training)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    from .._dispatch import dispatch
    return dispatch("layer_norm", input.device.type, input, normalized_shape, weight, bias, eps)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    from .._dispatch import dispatch
    return dispatch("group_norm", input.device.type, input, num_groups, weight, bias, eps)


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    from .._dispatch import dispatch
    return dispatch("batch_norm", input.device.type, input, running_mean, running_var,
                   weight, bias, training, momentum, eps)


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5):
    from .._dispatch import dispatch
    return dispatch("instance_norm", input.device.type, input, weight, bias,
                   running_mean, running_var, use_input_stats, momentum, eps)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    from .._dispatch import dispatch
    return dispatch("embedding", weight.device.type, weight, input, padding_idx, scale_grad_by_freq, sparse)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from .._dispatch import dispatch
    _stride = (stride,) if isinstance(stride, int) else tuple(stride)
    _padding = (padding,) if isinstance(padding, int) else tuple(padding)
    _dilation = (dilation,) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("conv1d", input.device.type, input, weight, bias,
                    _stride, _padding, _dilation, groups)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from .._dispatch import dispatch
    _stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    _padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    _dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("conv2d", input.device.type, input, weight, bias,
                    _stride, _padding, _dilation, groups)


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    from .._dispatch import dispatch
    _stride = (stride,) if isinstance(stride, int) else tuple(stride)
    _padding = (padding,) if isinstance(padding, int) else tuple(padding)
    _output_padding = (output_padding,) if isinstance(output_padding, int) else tuple(output_padding)
    _dilation = (dilation,) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("conv_transpose1d", input.device.type, input, weight, bias,
                    _stride, _padding, _output_padding, groups, _dilation)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    from .._dispatch import dispatch
    _stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    _padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    _output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else tuple(output_padding)
    _dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("conv_transpose2d", input.device.type, input, weight, bias,
                    _stride, _padding, _output_padding, groups, _dilation)


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    raise NotImplementedError("max_pool1d is not yet implemented")


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    from .._dispatch import dispatch
    _kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    _stride = _kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
    _padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    _dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("max_pool2d", input.device.type, input, _kernel_size, _stride,
                    _padding, _dilation, ceil_mode, return_indices)


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True):
    raise NotImplementedError("avg_pool1d is not yet implemented")


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    from .._dispatch import dispatch
    _kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    _stride = _kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
    _padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    return dispatch("avg_pool2d", input.device.type, input, _kernel_size, _stride,
                    _padding, ceil_mode, count_include_pad, divisor_override)


def adaptive_avg_pool1d(input, output_size):
    raise NotImplementedError("adaptive_avg_pool1d is not yet implemented")


def adaptive_avg_pool2d(input, output_size):
    from .._dispatch import dispatch
    if isinstance(output_size, int):
        _output_size = (output_size, output_size)
    else:
        _output_size = tuple(output_size)
    return dispatch("adaptive_avg_pool2d", input.device.type, input, _output_size)


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean', label_smoothing=0.0):
    log_probs = log_softmax(input, dim=1)
    return nll_loss(log_probs, target, weight=weight, ignore_index=ignore_index,
                    reduction=reduction)


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    from .._functional import add, neg, mul, mean, sum as _sum
    diff = add(input, neg(target))
    squared = mul(diff, diff)
    if reduction == 'none':
        return squared
    elif reduction == 'mean':
        return mean(squared)
    elif reduction == 'sum':
        return _sum(squared)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def binary_cross_entropy(input, target, weight=None, size_average=None,
                         reduce=None, reduction='mean'):
    from .._functional import add, neg, mul, log, mean, sum as _sum
    eps = 1e-12
    # -(target * log(input + eps) + (1 - target) * log(1 - input + eps))
    from .._creation import tensor as _tensor
    eps_t = _tensor(eps, device=input.device)
    one_t = _tensor(1.0, device=input.device)
    log_input = log(add(input, eps_t))
    log_one_minus_input = log(add(add(neg(input), one_t), eps_t))
    one_minus_target = add(neg(target), one_t)
    losses = neg(add(mul(target, log_input), mul(one_minus_target, log_one_minus_input)))
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    sig = sigmoid(input)
    return binary_cross_entropy(sig, target, weight=weight, reduction=reduction)


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    from .._functional import mean, sum as _sum, neg
    from .._dispatch import dispatch
    from .._dtype import int64 as int64_dtype
    # Gather log probabilities at target indices: -log_probs[i, target[i]]
    batch_size = input.shape[0]
    # Ensure target is int64 for gather
    if target.dtype != int64_dtype:
        target = target.to(dtype=int64_dtype)
    # unsqueeze target: (batch_size,) -> (batch_size, 1) for gather
    target_2d = target.view((batch_size, 1))
    gathered = dispatch("gather", input.device.type, input, 1, target_2d)
    # squeeze: (batch_size, 1) -> (batch_size,)
    losses = neg(gathered.view((batch_size,)))
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    from .._functional import abs as _abs, add, neg, mean, sum as _sum
    diff = add(input, neg(target))
    diff = _abs(diff)
    if reduction == 'none':
        return diff
    elif reduction == 'mean':
        return mean(diff)
    elif reduction == 'sum':
        return _sum(diff)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean',
                   beta=1.0):
    from .._functional import abs as _abs, add, neg, mean, sum as _sum, mul, where, signbit
    from .._creation import tensor as _tensor
    diff = add(input, neg(target))
    abs_diff = _abs(diff)
    # Compute element-wise: if |diff| < beta: 0.5 * diff^2 / beta, else |diff| - 0.5 * beta
    beta_t = _tensor(beta, device=input.device)
    half_t = _tensor(0.5, device=input.device)
    # signbit(abs_diff - beta) is True when abs_diff < beta
    mask = signbit(add(abs_diff, neg(beta_t)))
    # smooth part: 0.5 * diff^2 / beta
    squared = mul(diff, diff)
    smooth_part = mul(mul(half_t, squared), _tensor(1.0 / beta, device=input.device))
    # linear part: |diff| - 0.5 * beta
    linear_part = add(abs_diff, mul(neg(half_t), beta_t))
    result = where(mask, smooth_part, linear_part)
    if reduction == 'none':
        return result
    elif reduction == 'mean':
        return mean(result)
    elif reduction == 'sum':
        return _sum(result)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def kl_div(input, target, size_average=None, reduce=None, reduction='mean',
           log_target=False):
    from .._functional import mean, sum as _sum, mul, add, neg, exp, log
    from .._creation import tensor as _tensor
    eps_t = _tensor(1e-12, device=input.device)
    if log_target:
        # exp(target) * (target - input)
        exp_target = exp(target)
        diff = add(target, neg(input))
        losses = mul(exp_target, diff)
    else:
        # target * (log(target + eps) - input)
        log_target_val = log(add(target, eps_t))
        diff = add(log_target_val, neg(input))
        losses = mul(target, diff)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def pad(input, pad, mode='constant', value=0):
    from .._dispatch import dispatch
    return dispatch("pad", input.device.type, input, pad, mode, value)


def interpolate(input, size=None, scale_factor=None, mode='nearest',
                align_corners=None, recompute_scale_factor=None, antialias=False):
    from .._dispatch import dispatch
    if size is not None:
        if isinstance(size, int):
            output_size = (size, size)
        else:
            output_size = tuple(size)
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            sf_h = sf_w = float(scale_factor)
        else:
            sf_h, sf_w = float(scale_factor[0]), float(scale_factor[1])
        H, W = input.shape[2], input.shape[3]
        output_size = (int(H * sf_h), int(W * sf_w))
    else:
        raise ValueError("either size or scale_factor must be defined")

    if mode == 'nearest':
        return dispatch("upsample_nearest2d", input.device.type, input, output_size)
    elif mode == 'bilinear':
        ac = align_corners if align_corners is not None else False
        if scale_factor is not None and not recompute_scale_factor:
            if isinstance(scale_factor, (int, float)):
                sh = sw = float(scale_factor)
            else:
                sh, sw = float(scale_factor[0]), float(scale_factor[1])
        else:
            sh, sw = 0.0, 0.0
        return dispatch("upsample_bilinear2d", input.device.type, input, output_size, ac, sh, sw)
    else:
        raise NotImplementedError(f"interpolate mode '{mode}' is not yet implemented")


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None):
    import math
    from .._functional import matmul, mul, add, neg
    from .._creation import ones, tensor as _tensor
    from .._dtype import bool as bool_dtype

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    # query @ key^T
    key_t = key.transpose(-2, -1)
    attn_weight = matmul(query, key_t)
    scale_t = _tensor(scale_factor, device=query.device)
    attn_weight = mul(attn_weight, scale_t)

    if is_causal:
        causal_mask = ones((L, S), dtype=bool_dtype, device=query.device).tril()
        neg_inf = _tensor(float('-inf'), device=query.device)
        from .._dispatch import dispatch
        inv_mask = dispatch("eq", query.device.type, causal_mask, False)
        from .._functional import where as _where
        attn_weight = _where(inv_mask, neg_inf, attn_weight)

    if attn_mask is not None:
        if attn_mask.dtype == bool_dtype:
            neg_inf = _tensor(float('-inf'), device=query.device)
            from .._dispatch import dispatch as _dispatch
            inv_mask = _dispatch("eq", query.device.type, attn_mask, False)
            from .._functional import where as _where
            attn_weight = _where(inv_mask, neg_inf, attn_weight)
        else:
            attn_weight = add(attn_weight, attn_mask)

    attn_weight = softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        attn_weight = dropout(attn_weight, p=dropout_p)
    return matmul(attn_weight, value)


def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
    from .._functional import rms_norm as _rms_norm
    return _rms_norm(input, normalized_shape, weight, eps)


def normalize(input, p=2.0, dim=1, eps=1e-12):
    from .._functional import norm, div
    from .._dispatch import dispatch
    norms = norm(input, p=p, dim=dim, keepdim=True)
    clamped = dispatch("clamp_min", input.device.type, norms, eps)
    return div(input, clamped)


def one_hot(tensor, num_classes=-1):
    from .._dispatch import dispatch
    return dispatch("one_hot", tensor.device.type, tensor, num_classes)


def relu6(input, inplace=False):
    from .._functional import relu6 as _relu6
    return _relu6(input)


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    from .._functional import hardtanh as _hardtanh
    return _hardtanh(input, min_val, max_val)


def logsigmoid(input):
    from .._functional import softplus as _softplus, neg as _neg
    return _neg(_softplus(_neg(input)))
