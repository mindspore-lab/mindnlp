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


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    from .._dispatch import dispatch
    return dispatch("embedding", weight.device.type, weight, input, padding_idx, scale_grad_by_freq, sparse)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    raise NotImplementedError("conv1d is not yet implemented")


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    raise NotImplementedError("conv2d is not yet implemented")


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    raise NotImplementedError("conv_transpose1d is not yet implemented")


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    raise NotImplementedError("conv_transpose2d is not yet implemented")


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    raise NotImplementedError("max_pool1d is not yet implemented")


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    raise NotImplementedError("max_pool2d is not yet implemented")


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True):
    raise NotImplementedError("avg_pool1d is not yet implemented")


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    raise NotImplementedError("avg_pool2d is not yet implemented")


def adaptive_avg_pool1d(input, output_size):
    raise NotImplementedError("adaptive_avg_pool1d is not yet implemented")


def adaptive_avg_pool2d(input, output_size):
    raise NotImplementedError("adaptive_avg_pool2d is not yet implemented")


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
    import numpy as np
    from .._functional import mean, sum as _sum
    from .._creation import tensor
    pn = input.numpy()
    tn = target.numpy()
    eps = 1e-12
    losses = -(tn * np.log(pn + eps) + (1 - tn) * np.log(1 - pn + eps))
    result = tensor(losses.tolist(), device=input.device)
    if reduction == 'none':
        return result
    elif reduction == 'mean':
        return mean(result)
    elif reduction == 'sum':
        return _sum(result)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    sig = sigmoid(input)
    return binary_cross_entropy(sig, target, weight=weight, reduction=reduction)


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    import numpy as np
    from .._functional import mean, sum as _sum
    from .._creation import tensor
    log_probs = input.numpy()
    targets = target.numpy().astype(int)
    batch_size = log_probs.shape[0]
    losses = np.array([-log_probs[i, targets[i]] for i in range(batch_size)], dtype=np.float32)
    result = tensor(losses.tolist(), device=input.device)
    if reduction == 'none':
        return result
    elif reduction == 'mean':
        return mean(result)
    elif reduction == 'sum':
        return _sum(result)
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
    from .._functional import abs as _abs, add, neg, mean, sum as _sum
    diff = add(input, neg(target))
    abs_diff = _abs(diff)
    # Compute element-wise: if |diff| < beta: 0.5 * diff^2 / beta, else |diff| - 0.5 * beta
    # Use numpy for the conditional since we're on CPU
    import numpy as np
    d = abs_diff.numpy()
    result_arr = np.where(d < beta, 0.5 * d ** 2 / beta, d - 0.5 * beta).astype(np.float32)
    from .._creation import tensor
    result = tensor(result_arr.tolist())
    if reduction == 'none':
        return result
    elif reduction == 'mean':
        return mean(result)
    elif reduction == 'sum':
        return _sum(result)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def kl_div(input, target, size_average=None, reduce=None, reduction='mean',
           log_target=False):
    import numpy as np
    from .._functional import mean, sum as _sum
    from .._creation import tensor
    log_pred = input.numpy()
    tn = target.numpy()
    if log_target:
        losses = np.exp(tn) * (tn - log_pred)
    else:
        losses = tn * (np.log(tn + 1e-12) - log_pred)
    result = tensor(losses.tolist(), device=input.device)
    if reduction == 'none':
        return result
    elif reduction == 'mean':
        return mean(result)
    elif reduction == 'sum':
        return _sum(result)
    raise ValueError(f"Invalid reduction mode: {reduction}")


def pad(input, pad, mode='constant', value=0):
    from .._dispatch import dispatch
    return dispatch("pad", input.device.type, input, pad, mode, value)


def interpolate(input, size=None, scale_factor=None, mode='nearest',
                align_corners=None, recompute_scale_factor=None, antialias=False):
    raise NotImplementedError("interpolate is not yet implemented")


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None):
    raise NotImplementedError("scaled_dot_product_attention is not yet implemented")
