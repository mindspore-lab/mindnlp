"""nn functional"""
import math
import warnings
from typing import Optional, Tuple, List
import numpy as np
import mindspore
from mindspore import ops, Tensor
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.function.random_func import _get_seed, _set_prim_op_user_data
from mindspore.ops.operations import nn_ops

from mindnlp.configs import DEVICE_TARGET, ON_ORANGE_PI, use_pyboost

def gelu(input, approximate='none'):
    if use_pyboost():
        return mindspore.mint.nn.functional.gelu(input, approximate=approximate)
    return ops.gelu(input, approximate)

def relu(input):
    if use_pyboost():
        return mindspore.mint.nn.functional.relu(input)
    return ops.relu(input)

def tanh(input):
    if use_pyboost():
        return mindspore.mint.nn.functional.tanh(input)
    return ops.tanh(input)

def sigmoid(input):
    if use_pyboost() and not ON_ORANGE_PI:
        return mindspore.mint.nn.functional.sigmoid(input)
    return ops.sigmoid(input)

def silu(input):
    if DEVICE_TARGET == 'CPU' or ON_ORANGE_PI:
        return input * sigmoid(input)
    if use_pyboost():
        return mindspore.mint.nn.functional.silu(input)
    return ops.silu(input)

def mish(input):
    return ops.mish(input)

def relu6(input):
    return ops.relu6(input)

def elu(input, alpha=1.0):
    if use_pyboost():
        return mindspore.mint.nn.functional.elu(input, alpha)
    return ops.elu(input, alpha)

def glu(input, dim=-1):
    return ops.glu(input, dim)

def softplus(input, beta=1, threshold=20):
    if use_pyboost():
        return mindspore.mint.nn.functional.softplus(input, beta, threshold)
    return ops.softplus(input, beta, threshold)

def logsigmoid(input):
    return ops.logsigmoid(input)

def leaky_relu(input, alpha=0.2):
    if use_pyboost():
        return mindspore.mint.nn.functional.leaky_relu(input, alpha)
    return ops.leaky_relu(input, alpha)

def prelu(input, weight):
    return ops.prelu(input, weight)

def celu(input, alpha=1., inplace=False):
    return ops.celu(input, alpha)

def selu(input):
    return ops.selu(input)

def hardsigmoid(input, inplace=False):
    return ops.hardsigmoid(input)

def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    return ops.hardswish(input)

def avg_pool1d(input_array, pool_size, stride, padding=0, ceil_mode=False, count_include_pad=True):
    """
    Perform 1D average pooling on the input array of shape (N, C, L) without using explicit for loops.

    Parameters:
    - input_array (numpy array): The input array to be pooled, shape (N, C, L).
    - pool_size (int): The size of the pooling window.
    - stride (int): The stride of the pooling window.
    - padding (int): The amount of zero-padding to add to both sides of the input array.
    - ceil_mode (bool): If True, use ceil instead of floor to compute the output length.
    - count_include_pad (bool): If True, include padding in the average calculation.

    Returns:
    - numpy array: The result of the average pooling operation.
    """
    N, C, L = input_array.shape

    # Add padding to the input array
    if padding > 0:
        input_array = ops.pad(input_array, ((0, 0), (0, 0), (padding, padding)), mode='constant', value=(0, 0))

    # Calculate the output length
    if ceil_mode:
        output_length = int(np.ceil((L + 2 * padding - pool_size) / stride).astype(int) + 1)
    else:
        output_length = int(np.floor((L + 2 * padding - pool_size) / stride).astype(int) + 1)

    # Initialize the output array
    output_array = ops.zeros((N, C, output_length))

    # Generate the starting indices of the pooling windows
    indices = ops.range(0, output_length, 1) * stride
    indices = indices[:, None] + ops.range(0, pool_size, 1)

    # Ensure indices are within bounds
    indices = ops.minimum(indices, input_array.shape[2] - 1)

    # Use advanced indexing to extract the pooling windows
    windows = input_array[:, :, indices]

    # Calculate the mean along the pooling window dimension
    if count_include_pad:
        output_array = ops.mean(windows, axis=-1)
    else:
        valid_counts = ops.sum(windows != 0, dim=-1)
        valid_counts = ops.maximum(valid_counts, 1)  # Avoid division by zero
        output_array = ops.sum(windows, dim=-1) / valid_counts

    return output_array

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0):
    """
    Perform 2D average pooling on the input array.

    Parameters:
    - input_array (numpy array): The input array to be pooled, shape (N, C, H, W).
    - pool_size (tuple): The size of the pooling window (pool_height, pool_width).
    - stride (tuple): The stride of the pooling window (stride_height, stride_width).
    - padding (int or tuple): The amount of zero-padding to add to all sides of the input array.
    - ceil_mode (bool): If True, use ceil instead of floor to compute the output length.
    - count_include_pad (bool): If True, include padding in the average calculation.

    Returns:
    - numpy array: The result of the average pooling operation.
    """
    if use_pyboost():
        return mindspore.ops.function.nn_func.avg_pool2d_ext(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    return ops.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def dropout(input, p=0.5, training=True):
    if not training or p == 0:
        return input
    if use_pyboost() and not ON_ORANGE_PI:
        return mindspore.mint.nn.functional.dropout(input, p, training)
    return ops.dropout(input, p, training)

def dropout2d(input, p=0.5, training=False):
    return ops.dropout2d(input, p, training)

def drop_and_mask(keep_prob, seed=None):
    seed0, seed1 = _get_seed(seed, "dropout")
    dropout_op = ops.Dropout(keep_prob=keep_prob, Seed0=seed0, Seed1=seed1)
    dropout_op = _set_prim_op_user_data(dropout_op, "random_cache", False)
    out, mask = dropout_op(input)
    return out, mask

dense_ = ops.Dense()
def linear(input, weight, bias=None):
    if ON_ORANGE_PI:
        input = input.to(mindspore.float16)
        weight = weight.to(mindspore.float16)
        if bias is not None:
            bias = bias.to(mindspore.float16)
            return dense_(input, weight) + bias
        return dense_(input, weight)
    if use_pyboost():
        return mindspore.mint.nn.functional.linear(input, weight, bias)
    return dense_(input, weight, bias)


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    if input.shape != target.shape:
        target = target.unsqueeze(1).expand_as(input).to(input.dtype)
    if use_pyboost():
        return mindspore.mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, reduction, pos_weight)
    return ops.binary_cross_entropy_with_logits(input, target.astype(input.dtype), weight, pos_weight, reduction)

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    uniform_samples = _get_cache_prim(ops.UniformReal)()(logits.shape)
    gumbels = -ops.log(-ops.log(uniform_samples + eps) + eps) # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = softmax(gumbels, dim)

    if hard:
        # Straight through.
        index = y_soft.argmax(dim)
        y_hard = one_hot(index, logits.shape[dim])
        ret = ops.stop_gradient(y_hard - y_soft) + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def log_softmax(input, dim=-1, dtype=None):
    out = ops.log_softmax(input, dim)
    if dtype is not None:
        out = out.to(dtype)
    return out

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False):
    if use_pyboost():
        return mindspore.ops.auto_generate.gen_ops_prim.embedding_op(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq)
    return ops.gather(weight, input, 0)

def rms_norm(input, weight, eps=1e-5):
    return ops.rms_norm(input, weight, eps)[0]

def apply_rotary_pos_emb(query, key, cos, sin, position_ids, cos_format=0):
    return mindspore.ops.auto_generate.gen_ops_def.apply_rotary_pos_emb_(
        query, key, cos, sin, position_ids, cos_format
    )

def pad(input, pad, mode='constant', value=0.0):
    if use_pyboost():
        return mindspore.mint.nn.functional.pad(input, pad, mode, value)
    if mode in ['reflect', 'circular']:
        return ops.pad(input, pad, mode)
    return ops.pad(input, pad, mode, value)

def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    return _inner_nll_loss(input, target, weight, ignore_index, reduction, label_smoothing)
    # if label_smoothing != 0.0 or target.ndim != 1:
    # if weight is None:
    #     weight = ops.ones(input.shape[-1], dtype=input.dtype)
    # _nll_loss = _get_cache_prim(ops.NLLLoss)(reduction, ignore_index)
    # return _nll_loss(input, target, weight)[0]

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    input = input.to(mindspore.float32)
    class_dim = 0 if input.ndim == 1 else 1
    if target.dtype in [mindspore.float32, mindspore.float16]:
        return _cross_entropy(input, target, class_dim, weight, reduction, label_smoothing)
    return nll_loss(log_softmax(input, class_dim), target, weight, ignore_index, reduction, label_smoothing)


def _cross_entropy(inputs, target, target_dim, weight=None, reduction='mean', label_smoothing=0.0):
    """cross entropy inner function"""
    class_dim = 0 if inputs.ndim == 1 else 1
    n_classes = inputs.shape[class_dim]
    inputs = log_softmax(inputs, class_dim)
    if label_smoothing > 0.0:
        target = target * (1 - label_smoothing) + label_smoothing / n_classes

    if weight is None:
        weight = ops.ones_like(inputs)
    elif inputs.ndim != 1:
        broadcast_shape = [1 for _ in range(inputs.ndim)]
        broadcast_shape[1] = weight.shape[0]
        weight = weight.reshape(broadcast_shape)

    if reduction == 'mean':
        return -(inputs * target * weight).sum() / (inputs.size / n_classes)
    if reduction == 'sum':
        return -(inputs * target * weight).sum()
    return -(inputs * target * weight).sum(class_dim)


def _inner_nll_loss(inputs, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    ndim = inputs.ndim
    if ndim == 2:
        ret = _nll_loss(inputs, target, -1, weight, ignore_index, reduction, label_smoothing)
    elif ndim == 4:
        ret = _nll_loss(inputs, target, 1, weight, ignore_index, reduction, label_smoothing)
    elif ndim == 1:
        ret = _nll_loss(inputs, target, 0, weight, ignore_index, reduction, label_smoothing)
    else:
        n = inputs.shape[0]
        c = inputs.shape[1]
        out_size = (n,) + inputs.shape[2:]
        inputs = inputs.view((n, c, 1, -1))
        target = target.view((n, 1, -1))
        if reduction != 'none':
            ret = _nll_loss(inputs, target, 1, weight, ignore_index, reduction, label_smoothing)
        else:
            ret = _nll_loss(inputs, target, 1, weight, ignore_index, label_smoothing=label_smoothing)
            ret = ret.view(out_size)
    return ret


def _nll_loss(inputs, target, target_dim=-1, weight=None, ignore_index=None, reduction='none', label_smoothing=0.0):
    """nll loss inner function"""
    if target.ndim == inputs.ndim - 1:
        target = target.expand_dims(target_dim)
    if ignore_index is not None:
        non_pad_mask = ops.equal(target, ignore_index)
        target = target.masked_fill(non_pad_mask, ops.cast(0, target.dtype))
    else:
        non_pad_mask = target
    if weight is not None:
        loss_weights = ops.gather(weight, target, 0)
        orig_shape = inputs.shape
        if inputs.ndim != 2:
            inputs = inputs.view(orig_shape[:2] + (-1,))
            weight = weight.view(weight.shape + (1,))
        weighted_inputs = inputs * weight
        weighted_inputs = weighted_inputs.view(orig_shape)
        loss = ops.neg(ops.gather_d(weighted_inputs, target_dim, target))
        smooth_loss = ops.neg(weighted_inputs.sum(axis=target_dim, keepdims=True))
    else:
        loss = ops.neg(ops.gather_d(inputs, target_dim, target))
        smooth_loss = ops.neg(inputs.sum(axis=target_dim, keepdims=True))
        loss_weights = ops.ones_like(loss)

    if ignore_index is not None:
        loss = loss.masked_fill(non_pad_mask, ops.cast(0, loss.dtype))
        loss_weights = loss_weights.masked_fill(non_pad_mask, ops.cast(0, loss_weights.dtype))
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, ops.cast(0, smooth_loss.dtype))

    loss = loss.squeeze(target_dim)
    smooth_loss = smooth_loss.squeeze(target_dim)

    if reduction == 'sum':
        loss = loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        loss = loss.sum() / loss_weights.sum()
        smooth_loss = smooth_loss.sum() / loss_weights.sum()

    eps_i = label_smoothing / inputs.shape[target_dim]
    if label_smoothing != 0:
        loss = (1. - label_smoothing) * loss + eps_i * smooth_loss

    return loss

def mse_loss(input, target, reduction='mean'):
    return ops.mse_loss(input, target, reduction)

def l1_loss(input, target, reduction='mean'):
    return ops.l1_loss(input, target, reduction)

def smooth_l1_loss(input, target, beta=1.0, reduction='none'):
    input = input.to(mindspore.float32)
    target = target.to(mindspore.float32)
    return ops.smooth_l1_loss(input, target, beta, reduction)

def kl_div(logits, labels, reduction='mean', log_target=False):
    if log_target:
        labels = ops.log(labels)
    return ops.kl_div(logits, labels, reduction)

def manual_softmax(x, dim=-1):
    exp_x = ops.exp(x - ops.max(x, axis=dim, keepdims=True)[0])
    return exp_x / ops.sum(exp_x, dim=dim, keepdim=True)

def softmax(input, dim=-1, *, dtype=None):
    if use_pyboost():
        return mindspore.mint.nn.functional.softmax(input, dim, dtype=dtype)
    if dtype is not None:
        input = input.to(dtype)
    if dim is None:
        dim = -1
    if ON_ORANGE_PI:
        return manual_softmax(input, dim)
    softmax_ = _get_cache_prim(ops.Softmax)(dim)
    return softmax_(input)

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if weight is None:
        weight = ops.ones(normalized_shape, dtype=input.dtype)
    if bias is None:
        bias = ops.zeros(normalized_shape, dtype=input.dtype)
    if use_pyboost():
        return mindspore.mint.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    if weight is not None:
        begin_axis = input.ndim - weight.ndim
    else:
        begin_axis = -1
    _layer_norm = _get_cache_prim(ops.LayerNorm)(begin_axis, begin_axis, epsilon=eps)
    return _layer_norm(input, weight, bias)[0]

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    return ops.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor)

def normalize(input, p=2.0, dim=1, eps=1e-6):
    r"""
    Normalize a tensor along a specified dimension.
    
    Args:
        input (Tensor): The input tensor to be normalized.
        p (float, optional): The power parameter for the normalization. Default is 2.0.
        dim (int, optional): The dimension along which to normalize the tensor. Default is 1.
    
    Returns:
        None
    
    Raises:
        TypeError: If the input is not a tensor.
        ValueError: If the specified dimension is out of range or if the power parameter is not a positive number.
    
    This function normalizes the input tensor along the specified dimension using the power parameter 'p'.
    The normalization is performed by dividing each element of the tensor by the Lp norm of the tensor along the specified dimension.
    The Lp norm is defined as the p-th root of the sum of the absolute values raised to the power of 'p'.
    The resulting tensor will have the same shape as the input tensor.
    """
    return input / ops.norm(input, ord=p, dim=dim, keepdim=True)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):

    if running_mean is None:
        running_mean = ops.ones(input.shape[1])
    if running_var is None:
        running_var = ops.zeros(input.shape[1])
    if weight is None:
        weight = ops.ones(input.shape[1])
    if bias is None:
        bias = ops.zeros(input.shape[1])

    if use_pyboost():
        return mindspore.mint.nn.functional.batch_norm(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            training,
            momentum,
            eps
        )
    return ops.batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps
    )

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    pad_mode = 'pad'
    if not isinstance(padding, (int, tuple)):
        pad_mode = padding

    return ops.conv2d(input, weight, bias=bias, stride=stride, pad_mode=pad_mode, padding=padding, dilation=dilation, groups=groups)

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if use_pyboost():
        return mindspore.mint.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    return ops.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode, return_indices=return_indices)

def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if stride is None:
        stride = kernel_size

    kernel_size = (1, kernel_size)
    stride = (1, stride)
    padding = (0, padding)
    dilation = (1, dilation)

    input_2d = input.unsqueeze(2)

    if return_indices:
        output_2d, indices_2d = max_pool2d(input_2d, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        output_1d = output_2d.squeeze(2)
        indices_1d = indices_2d.squeeze(2)
        return output_1d, indices_1d
    else:
        output_2d = max_pool2d(input_2d, kernel_size, stride, padding, dilation, ceil_mode)
        output_1d = output_2d.squeeze(2)
        return output_1d


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    if use_pyboost():
        return mindspore.mint.nn.functional.group_norm(input, num_groups, weight, bias, eps)

    input_shape = input.shape
    N = input_shape[0]
    C = input_shape[1]
    input_reshaped = input.view(1, N * num_groups, -1 if N!=0 else 1)
    outputs = batch_norm(input_reshaped, None, None, None, None, True, 0., eps)
    out = outputs.view(input_shape)
    affine_param_shape = [1] * input.ndim
    affine_param_shape[1] = C
    affine_param_shape = tuple(affine_param_shape)
    if weight is not None and bias is not None:
        out = bias.view(affine_param_shape).addcmul(out, weight.view(affine_param_shape), 1)
    elif weight is not None:
        out = out.mul(weight.view(affine_param_shape))
    elif bias is not None:
        out = out.add(bias.view(affine_param_shape))
    return out


def _in_projection(
    q,
    k,
    v,
    w_q,
    w_k,
    w_v,
    b_q=None,
    b_k=None,
    b_v=None,
):
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    assert w_q.shape == (
        Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            # proj = linear(q, w, b)
            # # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            # proj = proj.unflatten(-1, (3, E)).unsqueeze(0).swapaxes(0, -2).squeeze(-2)
            # return proj[0], proj[1], proj[2]
            return linear(q, w, b).chunk(3, axis=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            # q_proj = linear(q, w_q, b_q)
            # kv_proj = linear(k, w_kv, b_kv)
            # # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            # kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).swapaxes(0, -2).squeeze(-2)
            # return (q_proj, kv_proj[0], kv_proj[1])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, axis=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal):
    embed_size = query.shape[-1]
    scaling_factor = ops.sqrt(ops.sqrt(mindspore.Tensor(embed_size, dtype=query.dtype)))
    query = query / scaling_factor

    if is_causal:
        L = query.shape[-2], S = key.shape[-2]
        attn_mask = ops.ones((L, S), mindspore.bool_).tril()

    attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = softmax(attn, -1)
    if dropout_p > 0.:
        attn = ops.dropout(attn, dropout_p)
    output = ops.matmul(attn, value)

    return output


def _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.ndim == 3:
        # Batched Inputs
        is_batched = True
        assert key.ndim == 3 and value.ndim == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.ndim}-D and {value.ndim}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.ndim == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.ndim}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.ndim in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.ndim}-D tensor instead")
    elif query.ndim == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.ndim == 2 and value.ndim == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.ndim}-D and {value.ndim}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.ndim == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.ndim}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.ndim in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.ndim}-D tensor instead")
            if attn_mask.ndim == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.ndim}-D query tensor")

    return is_batched


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, mindspore.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.ndim == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.ndim == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.ndim} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = ops.cat([k, bias_k.repeat(1, bsz, 1)])
        v = ops.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).swapaxes(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.shape[0] == bsz * num_heads, \
            f"expecting static_k.shape[0] of {bsz * num_heads}, but got {static_k.shape[0]}"
        assert static_k.shape[2] == head_dim, \
            f"expecting static_k.shape[2] of {head_dim}, but got {static_k.shape[2]}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.shape[0] == bsz * num_heads, \
            f"expecting static_v.shape[0] of {bsz * num_heads}, but got {static_v.shape[0]}"
        assert static_v.shape[2] == head_dim, \
            f"expecting static_v.shape[2] of {head_dim}, but got {static_v.shape[2]}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = ops.cat([k, ops.zeros(zero_attn_shape, dtype=k.dtype)], axis=1)
        v = ops.cat([v, ops.zeros(zero_attn_shape, dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            broadcast_to((-1, num_heads, -1, -1)).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = ops.baddbmm(attn_mask, q_scaled, k.swapaxes(-2, -1))
        else:
            attn_output_weights = ops.bmm(q_scaled, k.swapaxes(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = ops.bmm(attn_output_weights, v)

        attn_output = attn_output.swapaxes(0, 1).view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.shape[1])

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(axis=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.shape[0] == 1 and attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).view(bsz * tgt_len, embed_dim)

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.shape[1])
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None

def _canonical_mask(
        mask: Optional[mindspore.Tensor],
        mask_name: str,
        other_type: Optional[int],
        other_name: str,
        target_type: int,
        check_other: bool = True,
) -> Optional[mindspore.Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = ops.is_floating_point(mask)
        if _mask_dtype != mindspore.bool_ and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            zero_tensor = ops.zeros_like(mask, dtype=target_type)
            mask = ops.where(mask, mindspore.Tensor(float("-inf"), target_type), zero_tensor)
            # mask = (
            #     ops.zeros_like(mask, dtype=target_type)
            #     .masked_fill_(mask, float("-inf"))
            # )
    return mask

def _none_or_dtype(input: Optional[mindspore.Tensor]) -> Optional[int]:
    if input is None:
        return None
    elif isinstance(input, mindspore.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or mindspore.Tensor")

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    if use_pyboost():
        return mindspore.mint.nn.functional.unfold(input, kernel_size, dilation, padding, stride)
    return ops.unfold(input, kernel_size, dilation, padding, stride)

def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    if use_pyboost():
        return mindspore.mint.nn.functional.fold(input, output_size, kernel_size, dilation, padding, stride)
    return ops.fold(input, output_size, kernel_size, dilation, padding, stride)

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, tuple):
        pad = (0, 0, padding[0], padding[0])
    elif isinstance(padding, int):
        pad = (0, 0) + (padding,) * 2
    if not isinstance(padding, (int, tuple)):
        pad_mode = padding
        pad = (0,) * 4

    _conv2d = _get_cache_prim(ops.Conv2D)(out_channel=weight.shape[0] * groups,
                                        kernel_size=(1, weight.shape[-1]),
                                        mode=1,
                                        pad_mode=pad_mode,
                                        pad=pad,
                                        stride=(1, stride),
                                        dilation=(1, dilation),
                                        group=groups)

    input = input.expand_dims(2)
    output = _conv2d(input, weight.expand_dims(2))

    if bias is not None:
        output = ops.bias_add(output, bias)

    output = output.squeeze(2)
    return output

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    ctc_loss_op = _get_cache_prim(nn_ops.CTCLossV2)(blank=blank, reduction="none", zero_infinity=zero_infinity)
    loss, _ = ctc_loss_op(log_probs, targets, input_lengths, target_lengths)
    if zero_infinity:
        loss = ops.where(ops.isinf(loss), 0., loss)
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'mean':
        input_type = loss.dtype
        target_length_t = target_lengths.clip(1., None)
        loss = loss.astype("float32")
        loss = loss / target_length_t
        loss = loss.mean()
        loss = loss.astype(input_type)
    return loss

def one_hot(tensor, num_classes=-1):
    if use_pyboost():
        return mindspore.mint.nn.functional.one_hot(tensor, num_classes)
    return ops.one_hot(tensor, num_classes)

def pixel_shuffle(input, upscale_factor):
    return ops.pixel_shuffle(input, upscale_factor)

def pixel_unshuffle(input, downscale_factor):
    return ops.pixel_shuffle(input, downscale_factor)

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    if use_pyboost():
        return mindspore.mint.nn.functional.grid_sample(input, grid, mode, padding_mode, align_corners)
    return ops.grid_sample(input, grid, mode, padding_mode, align_corners)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    if DEVICE_TARGET == 'Ascend':
        zero_norm_mask = ((x1.sum(dim) == 0).int() & (x2.sum(dim) == 0).int()).bool()
    else:
        zero_norm_mask = (x1.sum(dim) == 0) & (x2.sum(dim) == 0)

    cosine_sim = ops.cosine_similarity(x1, x2, dim, eps)
    return ops.select(zero_norm_mask, ops.ones_like(cosine_sim), cosine_sim)

# def pairwise_distance():
#     return ops.pairwise_distance

def make_attention_mask(
    query_input: Tensor,
    key_input: Tensor,
    dtype=mindspore.float32,
):
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
    attention weights will be `[batch..., heads, len_q, len_kv]` and this
    function will produce `[batch..., 1, len_q, len_kv]`.

    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = ops.greater_equal(
        ops.expand_dims(query_input, axis=-1), ops.expand_dims(key_input, axis=-2)
    )
    mask = ops.expand_dims(mask, axis=-3)
    return mask.astype(dtype)


def make_causal_mask(
    x: Tensor, dtype=mindspore.float32
) -> Tensor:
    """Make a causal mask for self-attention.

    In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
    will be `[batch..., heads, len, len]` and this function will produce a
    causal mask of shape `[batch..., 1, len, len]`.

    Args:
      x: input array of shape `[batch..., len]`
      extra_batch_dims: number of batch dims to add singleton axes for, none by
        default
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
    """
    idxs = ops.broadcast_to(ops.range(0, x.shape[-1], 1), x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        dtype=dtype,
    )
