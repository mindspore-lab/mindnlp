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
    raise NotImplementedError("softmax is not yet implemented")


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    raise NotImplementedError("log_softmax is not yet implemented")


def gelu(input, approximate='none'):
    raise NotImplementedError("gelu is not yet implemented")


def silu(input, inplace=False):
    raise NotImplementedError("silu is not yet implemented")


def leaky_relu(input, negative_slope=0.01, inplace=False):
    raise NotImplementedError("leaky_relu is not yet implemented")


def elu(input, alpha=1.0, inplace=False):
    raise NotImplementedError("elu is not yet implemented")


def mish(input, inplace=False):
    raise NotImplementedError("mish is not yet implemented")


def prelu(input, weight):
    raise NotImplementedError("prelu is not yet implemented")


def dropout(input, p=0.5, training=True, inplace=False):
    if not training:
        return input
    raise NotImplementedError("dropout is not yet implemented")


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    raise NotImplementedError("layer_norm is not yet implemented")


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    raise NotImplementedError("group_norm is not yet implemented")


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    raise NotImplementedError("batch_norm is not yet implemented")


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    raise NotImplementedError("embedding is not yet implemented")


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
    raise NotImplementedError("cross_entropy is not yet implemented")


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    raise NotImplementedError("mse_loss is not yet implemented")


def binary_cross_entropy(input, target, weight=None, size_average=None,
                         reduce=None, reduction='mean'):
    raise NotImplementedError("binary_cross_entropy is not yet implemented")


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    raise NotImplementedError("binary_cross_entropy_with_logits is not yet implemented")


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    raise NotImplementedError("nll_loss is not yet implemented")


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    raise NotImplementedError("l1_loss is not yet implemented")


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean',
                   beta=1.0):
    raise NotImplementedError("smooth_l1_loss is not yet implemented")


def kl_div(input, target, size_average=None, reduce=None, reduction='mean',
           log_target=False):
    raise NotImplementedError("kl_div is not yet implemented")


def pad(input, pad, mode='constant', value=0):
    raise NotImplementedError("pad is not yet implemented")


def interpolate(input, size=None, scale_factor=None, mode='nearest',
                align_corners=None, recompute_scale_factor=None, antialias=False):
    raise NotImplementedError("interpolate is not yet implemented")


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None):
    raise NotImplementedError("scaled_dot_product_attention is not yet implemented")
