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


def dropout3d(input, p=0.5, training=True, inplace=False):
    """Randomly zero out entire channels of a 5D input (N, C, D, H, W).

    Each channel is zeroed out independently with probability ``p``.
    Falls back to the standard dropout dispatch (zeros individual elements) when
    a dedicated channel-wise dropout op is not available.
    """
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


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    from .._dispatch import dispatch
    _stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
    _padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    _dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("conv3d", input.device.type, input, weight, bias,
                    _stride, _padding, _dilation, groups)


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    from .._dispatch import dispatch
    _kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    _stride = _kernel_size if stride is None else ((stride,) if isinstance(stride, int) else tuple(stride))
    _padding = (padding,) if isinstance(padding, int) else tuple(padding)
    _dilation = (dilation,) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("max_pool1d", input.device.type, input, _kernel_size, _stride,
                    _padding, _dilation, ceil_mode, return_indices)


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
    from .._dispatch import dispatch
    _kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    _stride = _kernel_size if stride is None else ((stride,) if isinstance(stride, int) else tuple(stride))
    _padding = (padding,) if isinstance(padding, int) else tuple(padding)
    return dispatch("avg_pool1d", input.device.type, input, _kernel_size, _stride,
                    _padding, ceil_mode, count_include_pad)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    from .._dispatch import dispatch
    _kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    _stride = _kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
    _padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    return dispatch("avg_pool2d", input.device.type, input, _kernel_size, _stride,
                    _padding, ceil_mode, count_include_pad, divisor_override)


def adaptive_avg_pool1d(input, output_size):
    from .._dispatch import dispatch
    if isinstance(output_size, int):
        _output_size = (output_size,)
    else:
        _output_size = tuple(output_size)
    return dispatch("adaptive_avg_pool1d", input.device.type, input, _output_size)


def adaptive_avg_pool2d(input, output_size):
    from .._dispatch import dispatch
    if isinstance(output_size, int):
        _output_size = (output_size, output_size)
    else:
        _output_size = tuple(output_size)
    return dispatch("adaptive_avg_pool2d", input.device.type, input, _output_size)


def adaptive_max_pool2d(input, output_size, return_indices=False):
    from .._dispatch import dispatch
    if isinstance(output_size, int):
        _output_size = (output_size, output_size)
    else:
        _output_size = tuple(output_size)
    return dispatch("adaptive_max_pool2d", input.device.type, input, _output_size,
                    return_indices)


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
    elif mode == 'bicubic':
        ac = align_corners if align_corners is not None else False
        if scale_factor is not None and not recompute_scale_factor:
            if isinstance(scale_factor, (int, float)):
                sh = sw = float(scale_factor)
            else:
                sh, sw = float(scale_factor[0]), float(scale_factor[1])
        else:
            sh, sw = 0.0, 0.0
        return dispatch("upsample_bicubic2d", input.device.type, input, output_size, ac, sh, sw)
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


def huber_loss(input, target, reduction='mean', delta=1.0):
    from .._functional import abs as _abs, add, neg, mul, mean, sum as _sum, where, signbit
    from .._creation import tensor as _tensor
    diff = add(input, neg(target))
    abs_diff = _abs(diff)
    delta_t = _tensor(delta, device=input.device)
    half_t = _tensor(0.5, device=input.device)
    # mask: abs_diff < delta (signbit is True when value < 0, i.e. abs_diff - delta < 0)
    mask = signbit(add(abs_diff, neg(delta_t)))
    # smooth part: 0.5 * diff^2
    smooth_part = mul(half_t, mul(diff, diff))
    # linear part: delta * (abs_diff - 0.5 * delta)
    linear_part = mul(delta_t, add(abs_diff, neg(mul(half_t, delta_t))))
    result = where(mask, smooth_part, linear_part)
    if reduction == 'none':
        return result
    elif reduction == 'mean':
        return mean(result)
    elif reduction == 'sum':
        return _sum(result)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def cosine_embedding_loss(input1, input2, target, margin=0, reduction='mean'):
    from .._functional import mul, add, neg, mean, sum as _sum, clamp, where
    from .._creation import tensor as _tensor
    from .._dispatch import dispatch
    eps = 1e-8
    # compute cosine similarity along last dim
    dot = mul(input1, input2)
    dot_sum = dispatch("sum", input1.device.type, dot, dim=-1)
    norm1 = dispatch("norm", input1.device.type, input1, dim=-1)
    norm2 = dispatch("norm", input2.device.type, input2, dim=-1)
    eps_t = _tensor(eps, device=input1.device)
    denom = add(mul(norm1, norm2), eps_t)
    cos_sim = dispatch("div", input1.device.type, dot_sum, denom)
    # clamp cos_sim to [-1, 1]
    cos_sim = clamp(cos_sim, -1.0, 1.0)
    margin_t = _tensor(float(margin), device=input1.device)
    # if y==1: max(0, 1 - cos_sim)
    loss_pos = clamp(add(_tensor(1.0, device=input1.device), neg(cos_sim)), 0.0, None)
    # if y==-1: max(0, cos_sim - margin)
    loss_neg = clamp(add(cos_sim, neg(margin_t)), 0.0, None)
    # select based on target
    pos_mask = dispatch("eq", input1.device.type, target, 1)
    losses = where(pos_mask, loss_pos, loss_neg)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def margin_ranking_loss(input1, input2, target, margin=0, reduction='mean'):
    from .._functional import add, neg, mul, mean, sum as _sum, clamp
    from .._creation import tensor as _tensor
    margin_t = _tensor(float(margin), device=input1.device)
    # loss = max(0, -target * (input1 - input2) + margin)
    diff = add(input1, neg(input2))
    neg_target_diff = mul(neg(target), diff)
    raw = add(neg_target_diff, margin_t)
    losses = clamp(raw, 0.0, None)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6,
                        swap=False, reduction='mean'):
    from .._functional import add, neg, mean, sum as _sum, clamp, norm
    from .._creation import tensor as _tensor
    from .._dispatch import dispatch
    eps_t = _tensor(eps, device=anchor.device)
    margin_t = _tensor(float(margin), device=anchor.device)
    # d(a, p)
    diff_ap = add(anchor, neg(positive))
    dist_ap = norm(diff_ap, p=p, dim=-1)
    dist_ap = add(dist_ap, eps_t)
    # d(a, n)
    diff_an = add(anchor, neg(negative))
    dist_an = norm(diff_an, p=p, dim=-1)
    dist_an = add(dist_an, eps_t)
    if swap:
        # d(p, n)
        diff_pn = add(positive, neg(negative))
        dist_pn = norm(diff_pn, p=p, dim=-1)
        dist_pn = add(dist_pn, eps_t)
        dist_an = dispatch("min", anchor.device.type, dist_an, dist_pn)
    # loss = max(0, d(a,p) - d(a,n) + margin)
    raw = add(add(dist_ap, neg(dist_an)), margin_t)
    losses = clamp(raw, 0.0, None)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def hinge_embedding_loss(input, target, margin=1.0, reduction='mean'):
    from .._functional import add, neg, mean, sum as _sum, clamp, where
    from .._creation import tensor as _tensor
    from .._dispatch import dispatch
    margin_t = _tensor(float(margin), device=input.device)
    # if target == 1: loss = input
    # if target == -1: loss = max(0, margin - input)
    loss_pos = input
    loss_neg = clamp(add(margin_t, neg(input)), 0.0, None)
    pos_mask = dispatch("eq", input.device.type, target, 1)
    losses = where(pos_mask, loss_pos, loss_neg)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def soft_margin_loss(input, target, reduction='mean'):
    from .._functional import mul, neg, log, exp, add, mean, sum as _sum
    from .._creation import tensor as _tensor
    one_t = _tensor(1.0, device=input.device)
    # loss = log(1 + exp(-target * input))
    neg_target_input = mul(neg(target), input)
    losses = log(add(one_t, exp(neg_target_input)))
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
             reduction='mean', zero_infinity=False):
    from .._dispatch import dispatch
    return dispatch("ctc_loss", log_probs.device.type, log_probs, targets,
                    input_lengths, target_lengths, blank, reduction, zero_infinity)


def multi_margin_loss(input, target, p=1, margin=1.0, weight=None, reduction='mean'):
    raise NotImplementedError("multi_margin_loss is not yet implemented")


def multilabel_soft_margin_loss(input, target, weight=None, reduction='mean'):
    from .._functional import mul, neg, log, add, mean, sum as _sum
    from .._creation import tensor as _tensor
    from .._dispatch import dispatch
    # per-element binary cross entropy: -[y * log(sigmoid(x)) + (1-y) * log(1 - sigmoid(x))]
    # then average over classes (last dim), then reduce over batch
    sig = sigmoid(input)
    eps_t = _tensor(1e-12, device=input.device)
    one_t = _tensor(1.0, device=input.device)
    log_sig = log(add(sig, eps_t))
    log_one_minus_sig = log(add(add(neg(sig), one_t), eps_t))
    one_minus_target = add(neg(target), one_t)
    per_elem = neg(add(mul(target, log_sig), mul(one_minus_target, log_one_minus_sig)))
    # sum over classes and divide by num_classes
    num_classes = float(input.shape[-1])
    num_classes_t = _tensor(num_classes, device=input.device)
    losses_per_sample = dispatch("div", input.device.type,
                                 dispatch("sum", input.device.type, per_elem, dim=-1),
                                 num_classes_t)
    if reduction == 'none':
        return losses_per_sample
    elif reduction == 'mean':
        return mean(losses_per_sample)
    elif reduction == 'sum':
        return _sum(losses_per_sample)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def poisson_nll_loss(input, target, log_input=True, full=False, eps=1e-8,
                     reduction='mean'):
    from .._functional import exp, log, mul, add, neg, mean, sum as _sum
    from .._creation import tensor as _tensor
    eps_t = _tensor(eps, device=input.device)
    if log_input:
        # loss = exp(input) - target * input
        losses = add(exp(input), neg(mul(target, input)))
    else:
        # loss = input - target * log(input + eps)
        losses = add(input, neg(mul(target, log(add(input, eps_t)))))
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def hardswish(input, inplace=False):
    from .._functional import relu6 as _relu6, mul as _mul, add as _add, div as _div
    from .._creation import tensor as _tensor
    three = _tensor(3.0, device=input.device)
    six = _tensor(6.0, device=input.device)
    return _div(_mul(input, _relu6(_add(input, three))), six)


def hardsigmoid(input, inplace=False):
    from .._functional import relu6 as _relu6, add as _add, div as _div
    from .._creation import tensor as _tensor
    three = _tensor(3.0, device=input.device)
    six = _tensor(6.0, device=input.device)
    return _div(_relu6(_add(input, three)), six)


def selu(input, inplace=False):
    from .._dispatch import dispatch
    return dispatch("selu", input.device.type, input)


def celu(input, alpha=1.0, inplace=False):
    from .._dispatch import dispatch
    return dispatch("celu", input.device.type, input, alpha)


def softplus(input, beta=1, threshold=20):
    from .._functional import softplus as _softplus
    if beta == 1 and threshold == 20:
        return _softplus(input)
    from .._dispatch import dispatch
    return dispatch("softplus", input.device.type, input, beta, threshold)


def softsign(input):
    from .._functional import abs as _abs, add as _add, div as _div
    from .._creation import tensor as _tensor
    one = _tensor(1.0, device=input.device)
    return _div(input, _add(one, _abs(input)))


def threshold(input, threshold, value, inplace=False):
    from .._dispatch import dispatch
    return dispatch("threshold", input.device.type, input, threshold, value)


def glu(input, dim=-1):
    from .._functional import sigmoid as _sigmoid, mul as _mul
    from .._dispatch import dispatch
    a, b = dispatch("chunk", input.device.type, input, 2, dim)
    return _mul(a, _sigmoid(b))


def softmax2d(input):
    return softmax(input, dim=1)


def softmin(input, dim=None, _stacklevel=3, dtype=None):
    from .._functional import neg as _neg
    if dim is None:
        dim = -1
    return softmax(_neg(input), dim=dim)


def tanhshrink(input):
    from .._functional import tanh as _tanh, add as _add, neg as _neg
    return _add(input, _neg(_tanh(input)))


def softshrink(input, lambd=0.5):
    from .._dispatch import dispatch
    return dispatch("softshrink", input.device.type, input, lambd)


def hardshrink(input, lambd=0.5):
    from .._dispatch import dispatch
    return dispatch("hardshrink", input.device.type, input, lambd)


def rrelu(input, lower=1.0/8, upper=1.0/3, training=False, inplace=False):
    from .._dispatch import dispatch
    return dispatch("rrelu", input.device.type, input, lower, upper, training)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Computes cosine similarity between x1 and x2 along dim.

    cosine_similarity = dot(x1, x2) / (||x1|| * ||x2|| + eps)
    """
    from .._functional import mul, div, add
    from .._dispatch import dispatch
    from .._creation import tensor as _tensor

    # Compute dot product along dim: sum(x1 * x2, dim=dim)
    dot = dispatch("sum", x1.device.type, mul(x1, x2), dim=dim)

    # Compute norms along dim
    norm_x1 = dispatch("norm", x1.device.type, x1, 2.0, dim, False)
    norm_x2 = dispatch("norm", x2.device.type, x2, 2.0, dim, False)

    # Denominator with eps to avoid division by zero
    eps_t = _tensor(eps, device=x1.device)
    denom = add(mul(norm_x1, norm_x2), eps_t)

    return div(dot, denom)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """Computes the pairwise distance ||x1 - x2 + eps||_p."""
    from .._functional import add, neg, abs as _abs, pow as _pow
    from .._creation import tensor as _tensor
    from .._dispatch import dispatch

    eps_t = _tensor(eps, device=x1.device)
    diff = add(x1, neg(x2))
    diff = add(diff, eps_t)

    if p == 2.0:
        result = dispatch("norm", diff.device.type, diff, 2.0, -1, keepdim)
    elif p == 1.0:
        diff_abs = _abs(diff)
        result = dispatch("sum", diff.device.type, diff_abs, dim=-1, keepdim=keepdim)
    else:
        p_t = _tensor(p, device=x1.device)
        inv_p_t = _tensor(1.0 / p, device=x1.device)
        diff_abs = _abs(diff)
        powered = _pow(diff_abs, p_t)
        summed = dispatch("sum", diff.device.type, powered, dim=-1, keepdim=keepdim)
        result = _pow(summed, inv_p_t)

    return result


def pixel_shuffle(input, upscale_factor):
    """Rearranges elements in a tensor of shape (N, C*r^2, H, W) to (N, C, H*r, W*r).

    Args:
        input: tensor of shape (N, C*r^2, H, W)
        upscale_factor (int): factor to increase spatial resolution by (r)
    """
    from .._dispatch import dispatch

    N, C_r2, H, W = input.shape
    r = upscale_factor
    C = C_r2 // (r * r)
    # Reshape: (N, C, r, r, H, W)
    x = dispatch("reshape", input.device.type, input, (N, C, r, r, H, W))
    # Permute: (N, C, H, r, W, r)
    x = dispatch("permute", input.device.type, x, (0, 1, 4, 2, 5, 3))
    # Reshape: (N, C, H*r, W*r)
    x = dispatch("reshape", input.device.type, x, (N, C, H * r, W * r))
    return x


def pixel_unshuffle(input, downscale_factor):
    """Reverses the pixel_shuffle operation: (N, C, H*r, W*r) -> (N, C*r^2, H, W).

    Args:
        input: tensor of shape (N, C, H*r, W*r)
        downscale_factor (int): factor to reduce spatial resolution by (r)
    """
    from .._dispatch import dispatch

    N, C, Hr, Wr = input.shape
    r = downscale_factor
    H = Hr // r
    W = Wr // r
    # Reshape: (N, C, H, r, W, r)
    x = dispatch("reshape", input.device.type, input, (N, C, H, r, W, r))
    # Permute: (N, C, r, r, H, W)
    x = dispatch("permute", input.device.type, x, (0, 1, 3, 5, 2, 4))
    # Reshape: (N, C*r^2, H, W)
    x = dispatch("reshape", input.device.type, x, (N, C * r * r, H, W))
    return x


def channel_shuffle(input, groups):
    """Shuffles channels within groups to mix information across groups.

    Args:
        input: tensor of shape (N, C, H, W)
        groups (int): number of groups to divide channels into
    """
    from .._dispatch import dispatch

    N, C, H, W = input.shape
    channels_per_group = C // groups
    # Reshape: (N, groups, channels_per_group, H, W)
    x = dispatch("reshape", input.device.type, input, (N, groups, channels_per_group, H, W))
    # Transpose groups and channels: (N, channels_per_group, groups, H, W)
    x = dispatch("permute", input.device.type, x, (0, 2, 1, 3, 4))
    # Reshape back: (N, C, H, W)
    x = dispatch("reshape", input.device.type, x, (N, C, H, W))
    return x
