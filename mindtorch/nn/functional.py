"""nn functional"""
import math
import numbers
import warnings
from typing import Optional, Tuple, List

import mindtorch
from mindtorch.executor import execute
from mindtorch._C import default_generator
from mindtorch.nn.modules.utils import _pair

from ..configs import ON_A2, ON_A1

generator_step_ = 12

def gelu(input, *, approximate='none'):
    return execute('gelu', input, approximate)

def relu(input, inplace=False):
    if inplace:
        execute('inplace_relu', input)
        return input
    return execute('relu', input)

def tanh(input, inplace=False):
    if inplace:
        execute('inplace_tanh', input)
        return input
    return execute('tanh', input)

def sigmoid(input):
    return execute('sigmoid', input)

def silu(input, inplace=False):
    if inplace:
        execute('inplace_silu', input)
        return input
    return execute('silu', input)

def mish(input):
    return execute('mish', input)

def relu6(input):
    return execute('relu6', input)

def elu(input, alpha=1.0):
    return execute('elu', input, alpha)

def glu(input, dim=-1):
    if input.device.type == 'cuda':
        x, y = input.chunk(2, dim)
        return x * sigmoid(y)
    return execute('glu', input, dim)

def softplus(input, beta=1, threshold=20):
    return execute('softplus', input, beta, threshold)

def logsigmoid(input):
    return execute('logsigmoid', input)[0]

def leaky_relu(input, alpha=0.2):
    return execute('leaky_relu', input, alpha)

def prelu(input, weight):
    return execute('prelu', input, weight)

def celu(input, alpha=1., inplace=False):
    return ops.celu(input, alpha)

def selu(input):
    return ops.selu(input)

def hardsigmoid(input, inplace=False):
    return ops.hardsigmoid(input)

def hardswish(input: mindtorch.Tensor, inplace: bool = False) -> mindtorch.Tensor:
    return execute('hswish', input)

def hardshrink(input, lambd=0.5):
    return execute('hard_shrink', input, lambd)

def avg_pool1d(input, kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=True):
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
    return execute('avg_pool1d', input, kernel_size, stride, padding, ceil_mode, count_include_pad)

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
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
    return execute('avg_pool2d', input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return ops.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


def adaptive_avg_pool1d(input, output_size):
    return execute('adaptive_avg_pool1d', input, output_size)

def adaptive_avg_pool2d(input, output_size):
    return execute('adaptive_avg_pool2d', input, output_size)

def dropout(input, p=0.5, training=True, inplace=False):
    if not training or p==0:
        return input
    seed, offset = default_generator._step(generator_step_)
    seed._device = input.device
    offset._device = input.device
    out, _ = execute('dropout', input, p, seed, offset)
    if inplace:
        input.copy_(out)
        return input
    return out

def dropout2d(input, p=0.5, training=False):
    if not training or p==0:
        return input
    out, _ = execute('dropout2d', input, p)
    return out

def linear(input, weight, bias=None):
    return execute('dense', input, weight, bias)

def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    if input.shape != target.shape:
        target = target.unsqueeze(1).expand_as(input).to(input.dtype)
    
    return execute('binary_cross_entropy_with_logits', input, target, weight, pos_weight, reduction)

def gumbel_softmax(logits: mindtorch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> mindtorch.Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -mindtorch.empty_like(logits, memory_format=mindtorch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = mindtorch.zeros_like(
            logits, memory_format=mindtorch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def log_softmax(input, dim=None, dtype=None):
    return execute('log_softmax', input, dim, dtype)

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return execute('embedding', input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq)

def rms_norm(input, normalized_shape, weight, eps=None):
    if eps is None:
        eps = mindtorch.finfo(input.dtype).eps
    if weight is None:
        weight = mindtorch.ones(normalized_shape, dtype=input.dtype, device=input.device)
    return execute('rms_norm', input, weight, eps)

def fast_gelu(x):
    return ops.fast_gelu(x)

def swiglu(x, dim=-1):
    return ops.swiglu(x, dim)

def apply_rotary_pos_emb(query, key, cos, sin, position_ids, cos_format=0):
    return mindspore.ops.auto_generate.gen_ops_def.apply_rotary_pos_emb_(
        query, key, cos, sin, position_ids, cos_format
    )

def custom_circular_pad(x, pad):
    """手动实现 torch.nn.functional.pad 的 circular 模式。
    
    参数:
        x: 输入张量，形状为 (B, C, D1, D2, ...)
        pad: 填充参数，格式为 (left_N, right_N, left_{N-1}, right_{N-1}, ..., left_1, right_1)
              表示从最后维度开始向前定义填充大小
    
    返回:
        循环填充后的张量
    """
    ndim = x.dim()
    n_pad_dims = len(pad) // 2
    assert n_pad_dims <= ndim, "填充参数超过了张量的维度"
    
    # 按从最后维度向前处理填充
    for dim in range(ndim-1, ndim-1-n_pad_dims, -1):
        # 当前维度的左右填充量
        idx = 2 * (ndim - 1 - dim)  # 在pad元组中的起始位置
        left_pad = pad[idx]
        right_pad = pad[idx + 1]
        
        if left_pad == 0 and right_pad == 0:
            continue  # 跳过该维度
            
        size = x.shape[dim]  # 当前维度的原始长度
        new_size = left_pad + size + right_pad
        
        # 生成循环索引: (index - left_pad) mod size
        index = (mindtorch.arange(new_size, device=x.device) + new_size - left_pad) % size
        index = (index + x.shape[dim]) % x.shape[dim]
        x = mindtorch.index_select(x, dim, index)

    return x

def _reflection_pad(input, pad):
    """reflection pad"""
    out = input
    if len(pad) == 2:
        out = execute('reflection_pad_1d', input, pad)
    elif len(pad) == 4:
        out = execute('reflection_pad_2d', input, pad)
    else:
        out = execute('reflection_pad_3d', input, pad)
    return out

def _replication_pad(input, pad):
    """replication pad"""
    out = input
    if len(pad) == 2:
        out = execute('replication_pad_1d', input, pad)
    elif len(pad) == 4:
        out = execute('replication_pad_2d', input, pad)
    else:
        out = execute('replication_pad_3d', input, pad)
    return out

def _circular_pad(input_x, padding):
    """circular pad"""
    if isinstance(padding, tuple):
        padding = mindtorch.tensor(padding, dtype=mindtorch.int64, device=input_x.device)
    elif isinstance(padding, list):
        padding = mindtorch.tensor(padding, dtype=mindtorch.int64, device=input_x.device)
    is_expand = False
    if padding.shape[0] // 2 + 1 == input_x.ndim:
        input_x = input_x.expand_dims(0)
        is_expand = True
    out = execute('pad_v3', input_x, padding, "circular", None)
    if is_expand:
        out = out.squeeze(0)
    return out

def pad(input, pad, mode='constant', value=None):
    if isinstance(pad, tuple):
        pad = tuple(p if isinstance(p, int) else p.item() for p in pad)

    if input.device.type in ['cpu', 'meta', 'cuda'] or ON_A1:
        new_pad = ()
        for idx, pad_v in enumerate(pad):
            if not isinstance(pad_v, int):
                pad_v = pad_v.item()
            if pad_v < 0:
                dim = input.ndim - 1 - idx // 2
                input = input.narrow(dim, 0, input.shape[dim] + pad_v)
                pad_v = 0
            new_pad += (pad_v,)
        if sum(new_pad) == 0:
            return input
        if mode == 'circular':
            return custom_circular_pad(input, pad)
        elif mode == 'reflect':
            return execute('pad_v3', input, new_pad, mode)
        if value is None:
            value = 0
        if mode == "replicate":
            mode = "edge"
            return execute('pad_v3', input, new_pad, mode)
        if input.dtype.is_floating_point:
            value = float(value)
        elif input.dtype == mindtorch.bool:
            value = bool(value)
        elif input.dtype in [mindtorch.int32, mindtorch.int64]:
            value = int(value)
        if input.device.type == 'cuda' and mode == 'constant' and value == 0 and len(new_pad) > 6:
            paddings = ()
            for i in range(input.ndim-1, -1, -1):
                paddings += ((new_pad[2*i], new_pad[2*i+1]),)
            return execute('pad', input, paddings)
        return execute('pad_v3', input, new_pad, mode, value)
    out = input
    if (isinstance(pad, tuple) and not pad):
        return out
    if sum(pad) == 0:
        return out
    if mode == "constant":
        value = 0 if value is None else value
        out = execute('constant_pad_nd', input, pad, value)
    else:
        if value is not None and value != 0:
            raise ValueError(f"Padding mode {mode} doesn\'t take in value argument.")
        if mode == "circular":
            out = _circular_pad(input, pad)
        elif mode == "reflect":
            out = _reflection_pad(input, pad)
        elif mode == "replicate":
            out = _replication_pad(input, pad)
        else:
            raise ValueError(f"Pad filling mode must be 'constant' 'circular' 'reflect' or 'replicate'.")
    return out

def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    if input.device.type in ['npu', 'cpu']:
        return _nllloss_nd(input, target, weight, ignore_index, reduction)
    return _inner_nll_loss(input, target, weight, ignore_index, reduction)

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
        target = target.unsqueeze(target_dim)
    if ignore_index is not None:
        non_pad_mask = mindtorch.eq(target, ignore_index)
        target = target.masked_fill(non_pad_mask, mindtorch.cast(0, target.dtype))
    else:
        non_pad_mask = target
    if weight is not None:
        loss_weights = mindtorch.index_select(weight, 0, target)
        orig_shape = inputs.shape
        if inputs.ndim != 2:
            inputs = inputs.view(orig_shape[:2] + (-1,))
            weight = weight.view(weight.shape + (1,))
        weighted_inputs = inputs * weight
        weighted_inputs = weighted_inputs.view(orig_shape)
        loss = mindtorch.neg(mindtorch.gather(weighted_inputs, target_dim, target))
        smooth_loss = mindtorch.neg(weighted_inputs.sum(axis=target_dim, keepdims=True))
    else:
        loss = mindtorch.neg(mindtorch.gather(inputs, target_dim, target))
        smooth_loss = mindtorch.neg(inputs.sum(axis=target_dim, keepdims=True))
        loss_weights = mindtorch.ones_like(loss)

    if ignore_index is not None:
        loss = loss.masked_fill(non_pad_mask, mindtorch.cast(0, loss.dtype))
        loss_weights = loss_weights.masked_fill(non_pad_mask, mindtorch.cast(0, loss_weights.dtype))
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, mindtorch.cast(0, smooth_loss.dtype))

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

def _nllloss_nd(input, target, weight=None, ingore_index=-100, reduction='mean'):
    """nllloss_nd inner function"""
    input_dim = input.ndim
    class_dim = 0 if input_dim == 1 else 1
    n_classes = input.shape[class_dim]
    if weight is None:
        weight = mindtorch.ones(n_classes, dtype=input.dtype, device=input.device)
    if input_dim < 1:
        raise ValueError(f"input dim should be less than 1, but got {input_dim}")
    if input_dim != 1 and input.shape[0] != target.shape[0]:
        raise ValueError(f"input bacth_size should be equal to target batch_size, but got {input.shape[0]} and "
                         f"{target.shape[0]}")
    if input_dim == 1 or input_dim == 2:
        return execute('nllloss', input, target, weight, reduction, ingore_index)[0]
    if input_dim == 4:
        return execute('nllloss_2d', input, target, weight, reduction, ingore_index)[0]
    # input_dim==3 or input_dim>4
    n = input.shape[0]
    c = input.shape[1]
    out_size = (n,) + input.shape[2:]
    if input.numel() > 0:
        input = input.view((n, c, 1, -1))
    else:
        input = input.view((n, c, 0, 0))
    if target.numel() > 0:
        target = target.view((n, 1, -1))
    else:
        target = target.view((n, 0, 0))
    if reduction != 'none':
        return execute('nllloss_2d', input, target, weight, reduction, ingore_index)[0]
    ret = execute('nllloss_2d', input, target, weight, reduction, ingore_index)[0]
    return ret.view(out_size)


def cross_entropy_gpu(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    class_dim = 0 if input.ndim == 1 else 1
    if target.dtype.is_floating_point:
        return _cross_entropy(input, target, class_dim, weight, reduction, label_smoothing)
    return nll_loss(log_softmax(input, class_dim), target, weight, ignore_index, reduction)

def _cross_entropy(inputs, target, target_dim, weight=None, reduction='mean', label_smoothing=0.0):
    """cross entropy inner function"""
    class_dim = 0 if inputs.ndim == 1 else 1
    n_classes = inputs.shape[class_dim]
    inputs = log_softmax(inputs, class_dim)
    if label_smoothing > 0.0:
        target = target * (1 - label_smoothing) + label_smoothing / n_classes

    if weight is None:
        weight = mindtorch.ones_like(inputs)
    elif inputs.ndim != 1:
        broadcast_shape = [1 for _ in range(inputs.ndim)]
        broadcast_shape[1] = weight.shape[0]
        weight = weight.reshape(broadcast_shape)

    if reduction == 'mean':
        return -(inputs * target * weight).sum() / (inputs.nel / n_classes)
    if reduction == 'sum':
        return -(inputs * target * weight).sum()
    return -(inputs * target * weight).sum(class_dim)


def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    if label_smoothing < 0.0 or label_smoothing > 1.0:
        raise ValueError(f"For cross_entropy, label_smoothing must in [0, 1]")
    if input.ndim == 0 or input.shape[0] == 0:
        raise ValueError(f"For cross_entropy, input don't support 0-dim and shape[0].")
    if input.device.type == 'cuda':
        return cross_entropy_gpu(input, target, weight, ignore_index, reduction, label_smoothing)
    class_dim = 0 if input.ndim == 1 else 1
    n_classes = input.shape[class_dim]
    input = log_softmax(input, class_dim, dtype=input.dtype)
    # for probabilities
    target_dtype = target.dtype
    if target_dtype in [mindtorch.float32, mindtorch.float16, mindtorch.bfloat16]:
        return _cross_entropy_for_probabilities(input, target, weight, reduction, label_smoothing, class_dim,
                                                n_classes)
    return _cross_entropy_for_class_indices(input, target, weight, ignore_index, reduction, label_smoothing,
                                            class_dim, n_classes)

def _cross_entropy_for_probabilities(input, target, weight, reduction, label_smoothing, class_dim, n_classes):
    """cross_entropy inner function for class probabilities"""
    if input.shape != target.shape:
        raise ValueError("For cross_entropy that target is probabilities, input shape should equal to target shape.")
    if label_smoothing > 0.0:
        target = target * (1 - label_smoothing) + label_smoothing / n_classes
    loss = input * target
    if weight is not None:
        weight_ = weight
        ori_shape = loss.shape
        if input.ndim > 2:
            loss = loss.view(ori_shape[:2] + (-1,))
            weight_ = weight_.view(1, -1, 1)
        loss = loss * weight_
        loss = loss.view(ori_shape)
    if reduction == "mean":
        return -mindtorch.div(loss.sum(), (input.size / n_classes))
    if reduction == "sum":
        return -loss.sum()
    if reduction == "none":
        return -loss.sum(class_dim)
    raise ValueError(f"redution value {reduction} not valid.")


def _cross_entropy_for_class_indices(input, target, weight, ingore_index, reduction, label_smoothing, class_dim,
                                     n_classes):
    """cross_entropy inner function for class indices"""
    nllloss = _nllloss_nd(input, target, weight, ingore_index, reduction)
    if label_smoothing > 0.0:
        if weight is not None:
            weight_ = weight
            input_ = input
            ori_shape = input.shape
            if input.ndim > 2:
                input_ = input.view(ori_shape[:2] + (-1,))
                weight_ = weight_.view(1, -1, 1)
            loss = input_ * weight_
            loss = loss.view(ori_shape)
            smooth_loss = -loss.sum(class_dim)
        else:
            smooth_loss = -input.sum(class_dim)
        ignore_mask = mindtorch.eq(target, ingore_index)
        smooth_loss = mindtorch.masked_fill(smooth_loss, ignore_mask, 0.)
        if reduction == "mean":
            true_mask = ~ignore_mask
            if weight is not None:
                weight_sum = mindtorch.gather(weight, 0, mindtorch.masked_select(mindtorch.masked_select(target, true_mask))).sum()
                if weight_sum == 0:
                    ret = smooth_loss.sum()
                else:
                    ret = smooth_loss.sum() / weight_sum
            else:
                weight_sum = true_mask.sum()
                if weight_sum == 0:
                    ret = smooth_loss.sum()
                else:
                    ret = smooth_loss.sum() / weight_sum
        elif reduction == "sum":
            ret = smooth_loss.sum()
        elif reduction == "none":
            ret = smooth_loss
        else:
            raise ValueError(f"redution value {reduction} not valid.")
        return (1 - label_smoothing) * nllloss + ret * (label_smoothing / n_classes)
    return nllloss


def mse_loss(input, target, reduction='mean'):
    return execute('mse_loss', input, target, reduction)

def l1_loss(input, target, reduction='mean'):
    return execute('l1_loss', input, target, reduction)

def smooth_l1_loss(input, target, beta=1.0, reduction='none'):
    input = input.to(mindtorch.float32)
    target = target.to(mindtorch.float32)
    return ops.smooth_l1_loss(input, target, beta, reduction)

def kl_div(input, target, reduction='mean', log_target=False):
    if reduction == 'batchmean':
        reduced = execute('kl_div', input, target, 'sum', log_target)
    else:
        reduced = execute('kl_div', input, target, reduction, log_target)

    if reduction == 'batchmean' and input.ndim != 0:
        reduced = mindtorch.div(reduced, input.shape[0])

    return reduced

def softmax(input, dim=-1, *, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    out = execute('softmax', input, dim)
    return out

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if weight is None:
        weight = mindtorch.ones(normalized_shape, dtype=input.dtype, device=input.device)
    if bias is None:
        bias = mindtorch.zeros(normalized_shape, dtype=input.dtype, device=input.device)
    return execute('layer_norm', input, normalized_shape, weight, bias, eps)[0]


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got "
                    f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "output size in (o1, o2, ...,oK) format."
                )
            output_size = [s.item() if not isinstance(s, numbers.Number) else s for s in size]
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "Input and scale_factor must have the same number of spatial dimensions, but "
                    f"got input with spatial dimensions of {list(input.shape[2:])} and "
                    f"scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "scale_factor in (s1, s2, ...,sK) format."
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
        scale_factors = [float(scale_factor) for scale_factor in scale_factors]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if (
        recompute_scale_factor is not None
        and recompute_scale_factor
        and size is not None
    ):
        raise ValueError(
            "recompute_scale_factor is not meaningful with an explicit size."
        )

    # "area" mode always requires an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if mode in ["area", "bilinear", "bicubic", "nearest-exact"] and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        assert scale_factors is not None
        # make scale_factor a tensor in tracing so constant doesn't get baked in
        output_size = [
            (
                math.floor(
                    float(input.size(i + 2) * scale_factors[i])
                )
            )
            for i in range(dim)
        ]
        scale_factors = None

    if antialias and not (mode in ("bilinear", "bicubic") and input.ndim == 4):
        raise ValueError(
            "Anti-alias option is restricted to bilinear and bicubic modes and requires a 4-D tensor as input"
        )

    if input.dim() == 3 and mode == "nearest":
        return execute('upsample_nearest1d', input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        return execute('upsample_nearest2d', input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest":
        return execute('upsample_nearest3d', input, output_size, scale_factors)

    if input.dim() == 3 and mode == "nearest-exact":
        return torch._C._nn._upsample_nearest_exact1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest-exact":
        nearest_exact = _get_cache_prim(ops.ResizeNearestNeighborV2)(
            align_corners=False,
            half_pixel_centers=True)
        return nearest_exact(input, output_size)
    if input.dim() == 5 and mode == "nearest-exact":
        warnings.warn('interpolate do not support `nearest-exact` for 5-D input, use `nearest` instead.')
        return execute('upsample_nearest3d', input, output_size, scale_factors)

    if input.dim() == 3 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool1d(input, output_size)
    if input.dim() == 4 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool2d(input, output_size)
    if input.dim() == 5 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool3d(input, output_size)

    if input.dim() == 3 and mode == "linear":
        assert align_corners is not None
        return execute(
            'upsample_linear1d', input, output_size, scale_factors, align_corners
        )
    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None
        # if antialias:
        #     return torch._C._nn._upsample_bilinear2d_aa(
        #         input, output_size, align_corners, scale_factors
        #     )
        return execute(
            'upsample_bilinear2d', input, output_size, scale_factors, align_corners
        )
    if input.dim() == 5 and mode == "trilinear":
        assert align_corners is not None
        return upsample_trilinear3d_impl(
            input, output_size, scale_factors, align_corners
        )
    if input.dim() == 4 and mode == "bicubic":
        assert align_corners is not None
        # if antialias:
        #     return torch._C._nn._upsample_bicubic2d_aa(
        #         input, output_size, align_corners, scale_factors
        #     )
        return execute(
            'upsample_bicubic2d', input, output_size, scale_factors, align_corners
        )

    if input.dim() == 3 and mode == "bilinear":
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        f" (got {input.dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact"
        f" (got {mode})"
    )

def normalize(input, p=2.0, dim=1, eps=1e-6):
    r"""
    Normalize a tensor along a specified dimension.

    Args:
        input (mindtorch.Tensor): The input tensor to be normalized.
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
    return input / mindtorch.norm(input, p=p, dim=dim, keepdim=True)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):

    if running_mean is None:
        running_mean = mindtorch.ones(input.shape[1], dtype=input.dtype, device=input.device)
    if running_var is None:
        running_var = mindtorch.zeros(input.shape[1], dtype=input.dtype, device=input.device)
    if weight is None:
        weight = mindtorch.ones(input.shape[1], dtype=input.dtype, device=input.device)
    if bias is None:
        bias = mindtorch.zeros(input.shape[1], dtype=input.dtype, device=input.device)

    return execute(
        'batch_norm',
        input,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps
    )[0]

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(padding, str):
        return execute('conv1d_padding', input, weight, bias, stride, padding, dilation, groups)
    return execute('conv1d', input, weight, bias, stride, padding, dilation, groups)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(padding, str):
        return execute('conv2d_padding', input, weight, bias, stride, padding, dilation, groups)
    return execute('conv2d', input, weight, bias, stride, padding, dilation, groups)

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(padding, str):
        return execute('conv3d_padding', input, weight, bias, stride, padding, dilation, groups)
    return execute('conv3d', input, weight, bias, stride, padding, dilation, groups)

    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, (tuple, list)):
        pad = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    elif isinstance(padding, int):
        pad = (padding,) * 6
    if not isinstance(padding, (int, tuple, list)):
        pad_mode = padding
        pad = (0,) * 6

    out_channels = weight.shape[0]
    kernel_size = weight.shape[2:]
    conv3d_op = ops.Conv3D(out_channels,
                            kernel_size,
                            mode=1,
                            pad_mode=pad_mode,
                            pad=pad,
                            stride=tuple(stride),
                            dilation=dilation,
                            group=groups)
    output = conv3d_op(input, weight)
                            
    if bias is not None:
        output = ops.bias_add(output, bias)
    return output



def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x_2d = input.unsqueeze(2)  # (batch, in_channels, 1, L_in)
    
    # 2. 增加卷积核的高度维度
    weight_2d = weight.unsqueeze(2)  # (in_channels, out_channels, 1, kernel_size)
    
    # 3. 二维转置卷积
    output_2d = conv_transpose2d(
        x_2d,
        weight_2d,
        bias,
        stride=(1,) + stride,
        padding=(0,) + padding,
        output_padding=(0,) + output_padding,
        groups=groups,
        dilation=(1,) + dilation
    )  # 输出形状: (batch, out_channels, 1, L_out)
    # 4. 移除高度维度恢复一维
    return output_2d.squeeze(2)

def _deconv_output_length(pad_mode, filter_size, stride_size, dilation_size, padding):
    """Calculate the width and height of output."""
    length = 0
    filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)
    if pad_mode == 'valid':
        if filter_size - stride_size > 0:
            length = filter_size - stride_size
    elif pad_mode == 'pad':
        length = - padding + filter_size - stride_size

    return length

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    return execute('conv_transpose2d', input, weight, bias, stride, padding, output_padding, groups, dilation)

    # pad_mode = 'pad'
    # pad = padding
    # if isinstance(padding, tuple):
    #     pad = (0, 0, padding[0], padding[0])
    # elif isinstance(padding, int):
    #     pad = (0, 0) + (padding,) * 2
    # if not isinstance(padding, (int, tuple)):
    #     pad_mode = padding
    #     pad = (0,) * 4

    # in_channel, out_channels = weight.shape[0], weight.shape[1] * groups
    # kernel_size = weight.shape[2:]

    # conv2d_transpose_op = ops.Conv2DTranspose(out_channel=out_channels,
    #                                             kernel_size=kernel_size,
    #                                             mode=1,
    #                                             pad_mode=pad_mode,
    #                                             pad=pad,
    #                                             stride=stride,
    #                                             dilation=dilation,
    #                                             group=groups)
    # n, _, h, w = input.shape
    # h_add = _deconv_output_length(pad_mode, kernel_size[0], stride[0], dilation[0], pad[0] + pad[1])
    # w_add = _deconv_output_length(pad_mode, kernel_size[1], stride[1], dilation[1], pad[2] + pad[3])

    # out = conv2d_transpose_op(input, weight,
    #                           (n, out_channels, h * stride[0] + h_add, w * stride[1] + w_add))
    # if bias is not None:
    #     out = ops.bias_add(out, bias)
    # return out

def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if input.device.type == 'npu':
        in_channel, out_channel = weight.shape[0], weight.shape[1]
        kernel_size = weight.shape[2:]
        conv_transpose3d_op = ops.Conv3DTranspose(
            in_channel,
            out_channel,
            kernel_size,
            mode=1,
            pad_mode='valid',
            pad=padding,
            stride=stride,
            dilation=dilation,
            group=1,
            output_padding=output_padding,
            data_format="NCDHW"
        )
        if groups > 1:
            outputs = ()
            for i in range(groups):
                output = conv_transpose3d_op(input.half(), weight.half())            
                if bias is not None:
                    output = output + bias
                outputs = outputs + (output,)
            out = ops.concat(outputs, 1)
        else:
            out = conv_transpose3d_op(input, weight)
            if bias is not None:
                out = out + bias
        return out
    else:
        in_channel, out_channel = weight.shape[0], weight.shape[1] * groups
        kernel_size = weight.shape[2:]
        conv_transpose3d_op = ops.Conv3DTranspose(
            in_channel,
            out_channel,
            kernel_size,
            mode=1,
            pad_mode='valid',
            pad=padding,
            stride=stride,
            dilation=dilation,
            group=groups,
            output_padding=output_padding,
            data_format="NCDHW"
        )

        out = conv_transpose3d_op(input, weight)
        if bias is not None:
            out = out + bias
        return out


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    input_ndim = input.ndim
    if input_ndim == 3:
        input = input.unsqueeze(1)
    out = execute('max_pool2d', input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    if input_ndim == 3:
        out = out.squeeze(1)
    return out

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
    if weight is None:
        weight = mindtorch.ones([input.shape[1]], dtype=input.dtype, device=input.device)
    if bias is None:
        bias = mindtorch.zeros([input.shape[1]], dtype=input.dtype, device=input.device)
    if input.device.type == 'npu':
        return execute('group_norm', input, num_groups, weight, bias, eps)[0]

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
        out = mindtorch.addcmul(bias.view(affine_param_shape), out, weight.view(affine_param_shape), value=1)

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
    q: mindtorch.Tensor,
    k: mindtorch.Tensor,
    v: mindtorch.Tensor,
    w: mindtorch.Tensor,
    b: Optional[mindtorch.Tensor] = None,
) -> List[mindtorch.Tensor]:
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
            return linear(q, w, b).chunk(3, dim=-1)
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
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def repeat_kv(hidden_states, n_rep: int):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if ON_A2:
        return mindtorch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


ATTN_MASK_NPU_CACHE = {}


def get_attn_mask_npu(device):
    """Get or create attention mask for the specified device."""
    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = mindtorch.ones(2048, 2048, dtype=mindtorch.bool, device=device).triu(diagonal=1)
    return ATTN_MASK_NPU_CACHE[device]

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> mindtorch.Tensor:

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale


    if query.device.type == 'npu' and ON_A2:
        if attn_mask is not None and not is_causal:
            attn_mask = ~attn_mask

        sparse_mode = 0

        if is_causal:
            assert attn_mask is None
            attn_mask = get_attn_mask_npu(query.device)
            sparse_mode = 3

        head_num = query.shape[1]
        if enable_gqa and is_causal:
            output = execute('prompt_flash_attention', query, key, value, attn_mask,
                             actual_seq_lengths=None, actual_seq_lengths_kv=None, pse_shift=None,
                             deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None,
                             quant_offset2=None, num_heads=head_num, scale_value=scale_factor,
                             pre_tokens=2147483647, next_tokens=0, input_layout='BNSD',
                             num_key_value_heads=key.shape[1], sparse_mode=sparse_mode, inner_precise=1)
            # output = execute('incre_flash_attention', query, [key], [value], attn_mask,
            #                  actual_seq_lengths=None, pse_shift=None, dequant_scale1=None, quant_scale1=None,
            #                  dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None,
            #                  antiquant_offset=None, block_table=None, kv_padding_size=None, num_heads=head_num,
            #                  input_layout='BNSD', scale_value=scale_factor, num_key_value_heads=key.shape[1],
            #                  block_size=0, inner_precise=1)
            return output
        else:
            output = execute('flash_attention_score', query, key, value, head_num=head_num, input_layout='BNSD', real_shift=None, padding_mask=None, attn_mask=attn_mask,
                            scale_value=scale_factor, keep_prob=1 - dropout_p, pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0,
                            drop_mask=None, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=sparse_mode)

            sfm_max, sfm_sum, sfm_out, atten_out = output

            return atten_out

    if enable_gqa:
        key = repeat_kv(key, query.size(-3) // key.size(-3)).contiguous()
        value = repeat_kv(value, query.size(-3) // value.size(-3)).contiguous()
    
    attn_bias_shape = (L, S) if attn_mask is None else attn_mask.shape
    attn_bias = mindtorch.zeros(attn_bias_shape, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = mindtorch.ones(L, S, dtype=mindtorch.bool, device=query.device).tril(diagonal=0)
        attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), mindtorch.finfo(attn_bias.dtype).min)
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == mindtorch.bool:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.squeeze(0)
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), mindtorch.finfo(attn_bias.dtype).min)
        else:
            attn_bias = attn_mask + attn_bias
        
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax(attn_weight, dim=-1)
    attn_weight = dropout(attn_weight, dropout_p, training=True)
    return attn_weight @ value


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
    query: mindtorch.Tensor,
    key: mindtorch.Tensor,
    value: mindtorch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[mindtorch.Tensor],
    in_proj_bias: Optional[mindtorch.Tensor],
    bias_k: Optional[mindtorch.Tensor],
    bias_v: Optional[mindtorch.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: mindtorch.Tensor,
    out_proj_bias: Optional[mindtorch.Tensor],
    training: bool = True,
    key_padding_mask: Optional[mindtorch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[mindtorch.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[mindtorch.Tensor] = None,
    k_proj_weight: Optional[mindtorch.Tensor] = None,
    v_proj_weight: Optional[mindtorch.Tensor] = None,
    static_k: Optional[mindtorch.Tensor] = None,
    static_v: Optional[mindtorch.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[mindtorch.Tensor, Optional[mindtorch.Tensor]]:
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
          If a Floatmindtorch.Tensor is provided, it will be directly added to the value.
          If a Boolmindtorch.Tensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a Boolmindtorch.Tensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a Floatmindtorch.Tensor
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
    if isinstance(embed_dim, mindtorch.Tensor):
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
            attn_output_weights = mindtorch.baddbmm(attn_mask, q_scaled, k.swapaxes(-2, -1))
        else:
            attn_output_weights = mindtorch.bmm(q_scaled, k.swapaxes(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = mindtorch.bmm(attn_output_weights, v)

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
        mask: Optional[mindtorch.Tensor],
        mask_name: str,
        other_type: Optional[int],
        other_name: str,
        target_type: int,
        check_other: bool = True,
) -> Optional[mindtorch.Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = mindtorch.is_floating_point(mask)
        if _mask_dtype != mindtorch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            zero_tensor = mindtorch.zeros_like(mask, dtype=target_type, device=mask.device)
            mask = mindtorch.where(mask, mindtorch.tensor(float("-inf"), dtype=target_type, device=mask.device), zero_tensor)
            # mask = (
            #     ops.zeros_like(mask, dtype=target_type)
            #     .masked_fill_(mask, float("-inf"))
            # )
    return mask

def _none_or_dtype(input: Optional[mindtorch.Tensor]) -> Optional[int]:
    if input is None:
        return None
    elif isinstance(input, mindtorch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or mindtorch.Tensor")

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    return execute('im2col', input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))

def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    return execute('col2im', input, output_size, kernel_size, dilation, padding, stride)

# def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
#     return execute('ctc_loss', log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    """
    使用向量化操作手动实现 CTC Loss，提升计算效率。
    支持批处理，并在内部使用张量操作避免循环。

    参数:
        log_probs: Tensor of size (T, N, C), 其中 T=input length, N=batch size, C=number of classes (包括空白符).
                   通常应经过 log_softmax 处理。
        targets: Tensor of size (N, S) 或 (sum(target_lengths)), 表示目标序列。不包含空白符。
        input_lengths: Tensor or tuple of size (N), 表示每个输入序列的实际长度。
        target_lengths: Tensor or tuple of size (N), 表示每个目标序列的实际长度。
        blank (int, optional): 空白符的类别索引。默认为 0。
        reduction (str, optional): 指定损失的缩减方式：'none' | 'mean' | 'sum'. 默认为 'mean'.
        zero_infinity (bool, optional): 是否将无限损失（及其梯度）归零。默认为 False。

    返回:
        Tensor: 计算出的 CTC 损失。
    """
    T, N, C = log_probs.size()
    device = log_probs.device
    dtype = log_probs.dtype

    # 初始化损失张量
    losses = mindtorch.zeros(N, device=device, dtype=dtype)
    
    # 处理 targets 的格式 (N, S) 或 (sum(target_lengths))
    if targets.dim() == 1:
        # targets 是 1D 的 concatenated 形式
        targets_ = targets
    else:
        # targets 是 2D 的 (N, S) 形式
        targets_ = targets.view(-1)

    # 遍历批次中的每个样本
    for n in range(N):
        T_n = input_lengths[n]
        S_n = target_lengths[n]
        
        if S_n == 0:
            # 如果目标长度为0，则损失为 -log(在空白符上的概率和)
            blank_log_probs = log_probs[:T_n, n, blank]
            losses[n] = -mindtorch.sum(blank_log_probs)
            if zero_infinity and mindtorch.isinf(losses[n]):
                losses[n] = 0.0
            continue

        # 获取当前样本的目标序列
        if targets.dim() == 1:
            start_index = sum(target_lengths[:n])
            end_index = start_index + S_n
            target_seq = targets_[start_index:end_index]
        else:
            target_seq = targets[n, :S_n]

        # 构建扩展目标序列 (长度 L = 2 * S_n + 1)
        extended_targets = mindtorch.zeros(2 * S_n + 1, device=device, dtype=mindtorch.long)
        extended_targets[0] = blank
        extended_targets[1::2] = target_seq
        extended_targets[2::2] = blank
        L = len(extended_targets)

        # 初始化前向变量 alpha, 形状为 (T_n, L)
        alpha = mindtorch.full((T_n, L), mindtorch.finfo(dtype).min, device=device, dtype=dtype)
        
        # 初始化第一个时间步
        alpha[0, 0] = log_probs[0, n, extended_targets[0]]  # 从空白符开始
        if L > 1:
            alpha[0, 1] = log_probs[0, n, extended_targets[1]]  # 从第一个真实字符开始

        # 前向递归计算 alpha
        for t in range(1, T_n):
            # 获取当前时间步对所有扩展目标字符的 log_probs
            # log_probs_t 形状: (L,)
            log_probs_t = log_probs[t, n, extended_targets]
            
            # 初始化当前时间步的 alpha_prev，用于向量化计算
            alpha_prev = alpha[t-1]
            
            # 情况1: 从 s 转移过来 (停留)
            stay_log_prob = alpha_prev + log_probs_t
            
            # 情况2: 从 s-1 转移过来 (移动一步)
            move_one_log_prob = mindtorch.empty_like(stay_log_prob)
            move_one_log_prob[0] = -float('inf') # 第一个位置没有 s-1
            move_one_log_prob[1:] = alpha_prev[:-1] + log_probs_t[1:]
            
            # 情况3: 从 s-2 转移过来 (移动两步，需满足条件)
            move_two_log_prob = mindtorch.empty_like(stay_log_prob)
            move_two_log_prob[:2] = -float('inf') # 前两个位置没有 s-2
            # 条件: s >= 2, 且当前字符不是空白符，且当前字符与 s-2 处的字符不同
            condition = (extended_targets[2:] != extended_targets[:-2]) & (extended_targets[2:] != blank)
            # 将条件应用到 alpha_prev[s-2] 上
            eligible_s2 = alpha_prev[:-2].clone()
            eligible_s2[~condition] = -float('inf') # 不满足条件的设为 -inf
            move_two_log_prob[2:] = eligible_s2 + log_probs_t[2:]
            
            # 使用 logaddexp 合并三种情况的概率
            # 首先合并 stay 和 move_one
            combined_stay_move = mindtorch.logaddexp(stay_log_prob, move_one_log_prob)
            # 再合并 move_two
            alpha[t] = mindtorch.logaddexp(combined_stay_move, move_two_log_prob)

        # 计算最终的总对数似然 (最后时间步的最后两个位置)
        if L > 1:
            total_log_prob = mindtorch.logaddexp(alpha[T_n-1, L-1], alpha[T_n-1, L-2])
        else:
            total_log_prob = alpha[T_n-1, L-1]
        
        losses[n] = -total_log_prob

        # 处理 zero_infinity
        if zero_infinity and mindtorch.isinf(losses[n]):
            losses[n] = 0.0

    # 根据 reduction 参数返回损失
    if reduction == 'none':
        return losses
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'mean':
        return losses.mean()
    else:
        raise ValueError("reduction should be 'none', 'sum', or 'mean'.")

def one_hot(tensor, num_classes=-1):
    return execute('one_hot', tensor, num_classes)

def pixel_shuffle(input, upscale_factor):
    return execute('pixel_shuffle', input, upscale_factor)

def pixel_unshuffle(input, downscale_factor):
    return ops.pixel_unshuffle(input, downscale_factor)

def getWH(input):
    """Get [W, H] tensor from input"""
    H, W = input.size()[-2:]
    return mindtorch.tensor([[W, H]], dtype=mindtorch.float32, device=input.device)

def center_of(input):
    """return [(W-1)/2, (H-1)/2] tensor of input img"""
    if input.dim() == 4:
        H, W = input.size()[-2:]
        shape = [[W, H]]
    else:
        D, H, W = input.size()[-3:]
        shape = [[W, H, D]]
    return mindtorch.tensor(shape, dtype=mindtorch.float32, device=input.device).sub_(1).div_(2)

def u(s, a: float = -0.75):
    s2, s3 = s**2, s**3
    l1 = (a+2)*s3 - (a+3)*s2 + 1
    l2 = a*s3 - (5*a)*s2 + (8*a)*s - 4*a
    return l1.where(s <= 1, l2)

def bicubic_grid_sample(input, grid, padding_mode: str = 'zeros', align_corners: bool = False):
    """bicubic_grid_sample"""
    kernel_size = 4
    if not align_corners:
        grid = grid * getWH(input) / getWH(input).sub_(1)
    center = center_of(input)
    abs_loc = ((grid + 1) * center).unsqueeze(-1)

    locs = abs_loc.floor() + mindtorch.tensor([-1, 0, 1, 2], device=grid.device)

    loc_w, loc_h = locs.detach().flatten(0, 2).unbind(dim=-2)
    loc_w = loc_w.reshape(-1, 1, kernel_size).expand(-1, kernel_size, -1)
    loc_h = loc_h.reshape(-1, kernel_size, 1).expand(-1, -1, kernel_size)
    loc_grid = mindtorch.stack([loc_w, loc_h], dim=-1)
    loc_grid = loc_grid.view(grid.size(0), -1, 1, 2)/center - 1

    selected = grid_sample(input, loc_grid.detach(), mode='nearest',
                             padding_mode=padding_mode, align_corners=True)
    patch = selected.view(input.size()[:2]+grid.size()[1:3]+(kernel_size,)*2)

    mat_r, mat_l = u(mindtorch.abs(abs_loc - locs.detach())).unbind(dim=-2)
    output = mindtorch.einsum('bhwl,bchwlr,bhwr->bchw', mat_l, patch, mat_r)
    return output

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    align_corners = False if align_corners is None else align_corners
    if input.ndim == 4:
        if mode == 'bicubic':
            return bicubic_grid_sample(input, grid, padding_mode, align_corners)
        return execute('grid_sampler_2d', input, grid, mode, padding_mode, align_corners)
    return execute('grid_sampler_3d', input, grid, mode, padding_mode, align_corners)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    dot_product = mindtorch.sum(x1 * x2, dim=dim)
    
    # 2. 计算L2范数 (||x|| 和 ||y||)
    norm_vec1 = mindtorch.norm(x1, p=2, dim=dim)
    norm_vec2 = mindtorch.norm(x2, p=2, dim=dim)
    
    # 3. 计算余弦相似度: (x · y) / (||x|| * ||y|| + eps)
    cosine_sim = dot_product / (norm_vec1 * norm_vec2 + eps)
    
    return cosine_sim

# def pairwise_distance():
#     return ops.pairwise_distance

def make_attention_mask(
    query_input: mindtorch.Tensor,
    key_input: mindtorch.Tensor,
    dtype=mindtorch.float32,
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
    x: mindtorch.Tensor, dtype=mindtorch.float32
) -> mindtorch.Tensor:
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
    idxs = ops.broadcast_to(ops.arange(x.shape[-1], dtype=mindspore.int32), x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        dtype=dtype,
    )

def rotary_position_embedding(x, cos, sin, mode=0):
    return ops.rotary_position_embedding(x, cos, sin, mode)