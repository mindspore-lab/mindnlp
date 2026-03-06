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
    if approximate == 'tanh':
        import math
        from .._functional import tanh as _tanh, mul as _mul, add as _add, pow as _pow
        from .._creation import tensor as _tensor
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        coeff = _tensor(math.sqrt(2.0 / math.pi), device=input.device)
        k = _tensor(0.044715, device=input.device)
        half = _tensor(0.5, device=input.device)
        one = _tensor(1.0, device=input.device)
        three = _tensor(3.0, device=input.device)
        inner = _mul(coeff, _add(input, _mul(k, _pow(input, three))))
        return _mul(half, _mul(input, _add(one, _tanh(inner))))
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


def _channel_dropout(input, p, training, ndim_extra):
    from .._dispatch import dispatch
    from .._creation import empty, tensor as _tensor
    from .._functional import mul, div
    from .._dtype import float32
    N, C = input.shape[0], input.shape[1]
    mask_shape = [N, C] + [1] * ndim_extra
    mask = empty(*mask_shape, device=input.device)
    mask = dispatch("uniform", input.device.type, mask)
    keep = dispatch("ge", input.device.type, mask, p)
    keep_float = keep.to(dtype=float32)
    scale = _tensor(1.0 / (1.0 - p), device=input.device)
    return mul(mul(input, keep_float), scale)


def dropout1d(input, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return input
    return _channel_dropout(input, p, training, 1)


def dropout2d(input, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return input
    return _channel_dropout(input, p, training, 2)


def dropout3d(input, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return input
    return _channel_dropout(input, p, training, 3)


def alpha_dropout(input, p=0.5, training=False, inplace=False):
    if not training or p == 0:
        return input
    # SELU self-normalizing dropout constants
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    alpha_p = -alpha * scale
    a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
    b = -a * alpha_p * p
    # Generate mask and apply
    import math
    from .._functional import rand
    mask = (rand(input.shape, dtype=input.dtype, device=input.device) >= p).float()
    result = input * mask + alpha_p * (1 - mask)
    return result * a + b


def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    if not training or p == 0:
        return input
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    alpha_p = -alpha * scale
    a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
    b = -a * alpha_p * p
    from .._functional import rand
    noise_shape = list(input.shape[:2]) + [1] * (input.dim() - 2)
    mask = (rand(noise_shape, dtype=input.dtype, device=input.device) >= p).float()
    result = input * mask + alpha_p * (1 - mask)
    return result * a + b


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
    if label_smoothing > 0:
        from .._functional import sum as _sum, neg, mean
        from .._dispatch import dispatch
        from .._creation import tensor as _tensor
        from .._dtype import float32
        C = input.shape[1]
        nll = nll_loss(log_probs, target, weight=weight, ignore_index=ignore_index,
                       reduction=reduction)
        smooth_loss = neg(dispatch("sum", input.device.type, log_probs, dim=1))
        smooth_loss = dispatch("div", input.device.type, smooth_loss,
                               _tensor(float(C), device=input.device))
        valid = dispatch("ne", input.device.type, target, ignore_index)
        valid_float = valid.to(dtype=float32)
        smooth_loss = dispatch("mul", input.device.type, smooth_loss, valid_float)
        if reduction == 'mean':
            valid_count = dispatch("sum", input.device.type, valid_float)
            smooth_loss = dispatch("div", input.device.type,
                                   dispatch("sum", input.device.type, smooth_loss),
                                   valid_count)
        elif reduction == 'sum':
            smooth_loss = dispatch("sum", input.device.type, smooth_loss)
        ls = label_smoothing
        ls_t = _tensor(ls, device=input.device)
        one_minus_ls = _tensor(1.0 - ls, device=input.device)
        return dispatch("add", input.device.type,
                         dispatch("mul", input.device.type, one_minus_ls, nll),
                         dispatch("mul", input.device.type, ls_t, smooth_loss))
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
    from .._functional import add, neg, mul, mean, sum as _sum, exp, log, abs as _abs
    from .._dispatch import dispatch
    from .._creation import tensor as _tensor
    one_t = _tensor(1.0, device=input.device)
    max_val = dispatch("clamp_min", input.device.type, input, 0.0)
    neg_abs_input = neg(_abs(input))
    log_term = log(add(one_t, exp(neg_abs_input)))
    if pos_weight is not None:
        pw_minus_1 = add(pos_weight, neg(one_t))
        pw_factor = add(one_t, mul(pw_minus_1, target))
        log_term = mul(pw_factor, log_term)
    losses = add(add(max_val, neg(mul(input, target))), log_term)
    if weight is not None:
        losses = mul(losses, weight)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    from .._functional import mean, sum as _sum, neg
    from .._dispatch import dispatch
    from .._dtype import int64 as int64_dtype, float32
    from .._creation import tensor as _tensor
    batch_size = input.shape[0]
    if target.dtype != int64_dtype:
        target = target.to(dtype=int64_dtype)
    valid = dispatch("ne", input.device.type, target, ignore_index)
    target_safe = dispatch("clamp", input.device.type, target, 0, input.shape[1] - 1)
    target_2d = target_safe.view((batch_size, 1))
    gathered = dispatch("gather", input.device.type, input, 1, target_2d)
    losses = neg(gathered.view((batch_size,)))
    if weight is not None:
        w_2d = dispatch("gather", input.device.type,
                        weight.unsqueeze(0).expand((batch_size, weight.shape[0])),
                        1, target_2d)
        w = w_2d.view((batch_size,))
        losses = dispatch("mul", input.device.type, losses, w)
    valid_float = valid.to(dtype=float32)
    losses = dispatch("mul", input.device.type, losses, valid_float)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        if weight is not None:
            total_weight = dispatch("mul", input.device.type, w, valid_float)
            total_weight = _sum(total_weight)
        else:
            total_weight = _sum(valid_float)
        return dispatch("div", input.device.type, _sum(losses), total_weight)
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
    ndim = input.ndim
    # Determine output_size based on input dimensionality
    if ndim == 3:
        # 1D: (N, C, W)
        if size is not None:
            output_size = (size,) if isinstance(size, int) else tuple(size)
        elif scale_factor is not None:
            sf = float(scale_factor) if isinstance(scale_factor, (int, float)) else float(scale_factor[0])
            W = input.shape[2]
            output_size = (int(W * sf),)
        else:
            raise ValueError("either size or scale_factor must be defined")
    else:
        # 2D: (N, C, H, W)
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
        if ndim == 3:
            return dispatch("upsample_nearest1d", input.device.type, input, output_size)
        return dispatch("upsample_nearest2d", input.device.type, input, output_size)
    elif mode == 'linear':
        ac = align_corners if align_corners is not None else False
        sf = 0.0
        if scale_factor is not None and not recompute_scale_factor:
            sf = float(scale_factor) if isinstance(scale_factor, (int, float)) else float(scale_factor[0])
        return dispatch("upsample_linear1d", input.device.type, input, output_size, ac, sf)
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
    from .._functional import add, neg, mul, mean, sum as _sum, clamp, pow as _pow
    from .._creation import tensor as _tensor, zeros as _zeros
    from .._dispatch import dispatch
    from .._dtype import int64 as int64_dtype
    # loss_i = (1/C) * sum_j max(0, margin - (x[y_i] - x[j]))^p   for j != y_i
    batch_size, n_classes = input.shape[0], input.shape[1]
    if target.dtype != int64_dtype:
        target = target.to(dtype=int64_dtype)
    target_2d = target.view((batch_size, 1))
    correct_scores = dispatch("gather", input.device.type, input, 1, target_2d)
    # correct_scores: (batch_size, 1) -> broadcast with input (batch_size, n_classes)
    margin_t = _tensor(float(margin), device=input.device)
    diff = add(margin_t, add(input, neg(correct_scores)))  # margin - (correct - x_j) = margin + x_j - correct
    diff = clamp(diff, 0.0, None)
    if p == 2:
        diff = mul(diff, diff)
    # Zero out the correct class
    from .._functional import where as _where
    from .._creation import arange as _arange
    mask = dispatch("eq", input.device.type,
                    _arange(n_classes, device=input.device).unsqueeze(0),
                    target_2d)
    zero_t = _tensor(0.0, device=input.device)
    diff = _where(mask, zero_t, diff)
    # Sum over classes and divide by n_classes
    losses = dispatch("sum", input.device.type, diff, dim=1)
    n_classes_t = _tensor(float(n_classes), device=input.device)
    losses = dispatch("div", input.device.type, losses, n_classes_t)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


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


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    from .._functional import log as _log, add as _add, neg as _neg, div as _div
    from .._creation import tensor as _tensor
    from .._dispatch import dispatch
    # Sample from Gumbel(0, 1): -log(-log(U)) where U ~ Uniform(0,1)
    u = dispatch("uniform", logits.device.type, logits)
    eps_t = _tensor(eps, device=logits.device)
    gumbels = _neg(_log(_add(_neg(_log(_add(u, eps_t))), eps_t)))
    # (logits + gumbels) / tau
    tau_t = _tensor(float(tau), device=logits.device)
    scores = _div(_add(logits, gumbels), tau_t)
    y_soft = softmax(scores, dim=dim)
    if hard:
        idx = dispatch("argmax", logits.device.type, y_soft, dim)
        y_hard = dispatch("one_hot", logits.device.type, idx, logits.shape[dim])
        # Straight-through: y_hard - y_soft.detach() + y_soft
        ret = _add(_add(y_hard.to(dtype=y_soft.dtype), _neg(y_soft.detach())), y_soft)
        return ret
    return y_soft


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    from .._dispatch import dispatch
    _kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    _dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    _padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    _stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    return dispatch("im2col", input.device.type, input, _kernel_size, _dilation, _padding, _stride)


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    from .._dispatch import dispatch
    _output_size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
    _kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    _dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    _padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    _stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    return dispatch("col2im", input.device.type, input, _output_size, _kernel_size,
                    _dilation, _padding, _stride)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    from .._dispatch import dispatch
    if align_corners is None:
        align_corners = False
    return dispatch("grid_sample", input.device.type, input, grid, mode, padding_mode, align_corners)


def affine_grid(theta, size, align_corners=None):
    from .._dispatch import dispatch
    if align_corners is None:
        align_corners = False
    return dispatch("affine_grid", theta.device.type, theta, size, align_corners)


def alpha_dropout(input, p=0.5, training=False, inplace=False):
    if not training or p == 0:
        return input
    from .._dispatch import dispatch
    from .._creation import empty, tensor as _tensor
    from .._functional import mul, add, neg
    from .._dtype import float32
    import math
    alpha = 1.6732632423543772
    lam = 1.0507009873554805
    alpha_prime = -alpha * lam
    # Compute affine constants to maintain self-normalizing property
    a = 1.0 / math.sqrt((1.0 - p) * (1.0 + p * alpha_prime * alpha_prime))
    b = -a * p * alpha_prime
    mask = empty(*input.shape, device=input.device)
    mask = dispatch("uniform", input.device.type, mask)
    keep = dispatch("ge", input.device.type, mask, p)
    keep_float = keep.to(dtype=float32)
    # Where kept: input; where dropped: alpha_prime
    from .._functional import where as _where
    alpha_prime_t = _tensor(alpha_prime, device=input.device)
    dropped = _where(keep, input, alpha_prime_t)
    # Apply affine transform: a * dropped + b
    a_t = _tensor(a, device=input.device)
    b_t = _tensor(b, device=input.device)
    return add(mul(a_t, dropped), b_t)


def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction='mean'):
    from .._functional import add, neg, mul, mean, sum as _sum, log, div
    from .._dispatch import dispatch
    from .._creation import tensor as _tensor
    var = dispatch("clamp_min", var.device.type, var, eps)
    # 0.5 * (log(var) + (input - target)^2 / var)
    diff = add(input, neg(target))
    diff_sq = mul(diff, diff)
    log_var = log(var)
    losses = mul(_tensor(0.5, device=input.device),
                 add(log_var, div(diff_sq, var)))
    if full:
        import math
        losses = add(losses, _tensor(0.5 * math.log(2.0 * math.pi), device=input.device))
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return mean(losses)
    elif reduction == 'sum':
        return _sum(losses)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def bilinear(input1, input2, weight, bias=None):
    from .._functional import matmul, mul, add
    from .._dispatch import dispatch
    # weight: (out_features, in1_features, in2_features)
    # input1: (..., in1_features), input2: (..., in2_features)
    orig_shape = input1.shape[:-1]
    in1 = input1.shape[-1]
    in2 = input2.shape[-1]
    out_f = weight.shape[0]
    # Flatten batch dims
    batch = 1
    for s in orig_shape:
        batch *= s
    x1 = input1.reshape((batch, in1))
    x2 = input2.reshape((batch, in2))
    # weight_perm: (in1, out_f, in2)
    weight_perm = dispatch("permute", weight.device.type, weight, (1, 0, 2))
    # x1 @ weight_perm.reshape(in1, out_f*in2) -> (batch, out_f*in2)
    w_2d = weight_perm.reshape((in1, out_f * in2))
    intermediate = matmul(x1, w_2d)
    # reshape to (batch, out_f, in2)
    intermediate = intermediate.reshape((batch, out_f, in2))
    # element-wise multiply by x2 and sum over in2
    x2_expanded = x2.unsqueeze(1).expand((batch, out_f, in2))
    result = mul(intermediate, x2_expanded)
    result = dispatch("sum", result.device.type, result, dim=2)
    if bias is not None:
        result = add(result, bias)
    # Reshape back to (..., out_f)
    out_shape = tuple(orig_shape) + (out_f,)
    return result.reshape(out_shape)


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                  scale_grad_by_freq=False, mode='mean', sparse=False,
                  per_sample_weights=None, include_last_offset=False,
                  padding_idx=None):
    from .._dispatch import dispatch
    from .._dtype import int64 as int64_dtype
    if input.ndim == 2:
        # 2D input: each row is a bag
        num_bags = input.shape[0]
        bag_size = input.shape[1]
        results = []
        for i in range(num_bags):
            row_indices = input[i]
            row_indices = row_indices.to(dtype=int64_dtype).contiguous()
            flat = row_indices.view((bag_size,))
            embeddings = dispatch("embedding", weight.device.type, weight, flat, padding_idx, False, False)
            if per_sample_weights is not None:
                pw = per_sample_weights[i].unsqueeze(1).expand(embeddings.shape)
                embeddings = dispatch("mul", weight.device.type, embeddings, pw)
            if mode == 'sum':
                bag_result = dispatch("sum", weight.device.type, embeddings, dim=0)
            elif mode == 'mean':
                bag_result = dispatch("mean", weight.device.type, embeddings, dim=0)
            elif mode == 'max':
                bag_result = dispatch("amax", weight.device.type, embeddings, dim=0)
            else:
                raise ValueError(f"Invalid mode: {mode}")
            results.append(bag_result.unsqueeze(0))
        return dispatch("cat", weight.device.type, results, 0)
    else:
        # 1D input with offsets
        if offsets is None:
            raise ValueError("offsets required for 1D input")
        input_flat = input.to(dtype=int64_dtype).contiguous()
        total = input_flat.shape[0]
        input_flat = input_flat.view((total,))
        num_bags = offsets.shape[0]
        results = []
        for i in range(num_bags):
            start_idx = int(offsets[i])
            end_idx = int(offsets[i + 1]) if i + 1 < num_bags else total
            count = end_idx - start_idx
            if count == 0:
                from .._creation import zeros
                results.append(zeros(1, weight.shape[1], device=weight.device))
                continue
            bag_indices = input_flat[start_idx:end_idx]
            embeddings = dispatch("embedding", weight.device.type, weight, bag_indices, padding_idx, False, False)
            if mode == 'sum':
                bag_result = dispatch("sum", weight.device.type, embeddings, dim=0)
            elif mode == 'mean':
                bag_result = dispatch("mean", weight.device.type, embeddings, dim=0)
            elif mode == 'max':
                bag_result = dispatch("amax", weight.device.type, embeddings, dim=0)
            else:
                raise ValueError(f"Invalid mode: {mode}")
            results.append(bag_result.unsqueeze(0))
        return dispatch("cat", weight.device.type, results, 0)


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0):
    from .._functional import mul, add, pow as _pow, div
    from .._dispatch import dispatch
    from .._creation import tensor as _tensor
    C = input.shape[1]
    half_n = size // 2
    sq = mul(input, input)
    # Pad channels with zeros and compute sliding window sum
    # Build channel sum by accumulating
    from .._creation import zeros
    sum_sq = zeros(*input.shape, device=input.device)
    for c in range(C):
        c_start = max(0, c - half_n)
        c_end = min(C, c + half_n + 1)
        for j in range(c_start, c_end):
            sum_sq_slice = sum_sq[:, c:c+1]
            sq_slice = sq[:, j:j+1]
            new_val = add(sum_sq_slice, sq_slice)
            # update via setitem
            dispatch("setitem", input.device.type, sum_sq, (slice(None), slice(c, c+1)), new_val)
    # norm_factor = (k + alpha * sum_sq) ^ beta
    alpha_t = _tensor(alpha, device=input.device)
    k_t = _tensor(k, device=input.device)
    beta_t = _tensor(beta, device=input.device)
    norm_factor = _pow(add(k_t, mul(alpha_t, sum_sq)), beta_t)
    return div(input, norm_factor)


def pdist(input, p=2.0):
    from .._functional import add, neg, abs as _abs, pow as _pow, norm
    from .._dispatch import dispatch
    from .._creation import tensor as _tensor
    N = input.shape[0]
    results = []
    for i in range(N):
        for j in range(i + 1, N):
            diff = add(input[i], neg(input[j]))
            if p == 2.0:
                d = dispatch("norm", diff.device.type, diff, 2.0, None, False)
            elif p == 1.0:
                d = dispatch("sum", diff.device.type, _abs(diff))
            elif p == float('inf'):
                d = dispatch("amax", diff.device.type, _abs(diff))
            else:
                p_t = _tensor(p, device=input.device)
                inv_p_t = _tensor(1.0 / p, device=input.device)
                d = _pow(dispatch("sum", diff.device.type, _pow(_abs(diff), p_t)), inv_p_t)
            results.append(d.unsqueeze(0))
    if not results:
        from .._creation import zeros
        return zeros(0, device=input.device)
    return dispatch("cat", input.device.type, results, 0)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    from .._dispatch import dispatch
    _stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
    _padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    _output_padding = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else tuple(output_padding)
    _dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("conv_transpose3d", input.device.type, input, weight, bias,
                    _stride, _padding, _output_padding, groups, _dilation)


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    from .._dispatch import dispatch
    _kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    _stride = _kernel_size if stride is None else ((stride, stride, stride) if isinstance(stride, int) else tuple(stride))
    _padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    _dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    return dispatch("max_pool3d", input.device.type, input, _kernel_size, _stride,
                    _padding, _dilation, ceil_mode, return_indices)


def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True):
    from .._dispatch import dispatch
    _kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    _stride = _kernel_size if stride is None else ((stride, stride, stride) if isinstance(stride, int) else tuple(stride))
    _padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    return dispatch("avg_pool3d", input.device.type, input, _kernel_size, _stride,
                    _padding, ceil_mode, count_include_pad)


def adaptive_avg_pool3d(input, output_size):
    from .._dispatch import dispatch
    if isinstance(output_size, int):
        _output_size = (output_size, output_size, output_size)
    else:
        _output_size = tuple(output_size)
    return dispatch("adaptive_avg_pool3d", input.device.type, input, _output_size)
