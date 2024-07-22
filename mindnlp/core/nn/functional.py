"""nn functional"""
import mindspore.mint.nn.functional
import numpy as np
import mindspore
from mindspore import ops
from mindnlp.configs import USE_PYBOOST

def gelu(input, approximate='none'):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.gelu(input, approximate)
    return ops.gelu(input, approximate)

def relu(input):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.relu(input)
    return ops.relu(input)

def tanh(input):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.tanh(input)
    return ops.tanh(input)

def sigmoid(input):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.sigmoid(input)
    return ops.sigmoid(input)

def silu(input):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.silu(input)
    return ops.silu(input)

def mish(input):
    return ops.mish(input)

def relu6(input):
    return ops.relu6(input)

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
    indices = ops.arange(output_length) * stride
    indices = indices[:, None] + ops.arange(pool_size)

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

def dropout(input, p=0.5, training=True):
    if USE_PYBOOST:
        return mindspore.mint.dropout(input, p, training)
    return ops.dropout(input, p, training)

dense_ = ops.Dense()
def linear(input, weight, bias=None):
    if USE_PYBOOST:
        return mindspore.mint.linear(input, weight, bias)
    return dense_(input, weight, bias)

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean'):
    return ops.cross_entropy(input, target, weight, ignore_index, reduction)

def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, reduction, pos_weight)
    return ops.binary_cross_entropy_with_logits(input, target, weight, reduction, pos_weight)

def log_softmax(input, dim=-1):
    return ops.log_softmax(input, dim)

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False):
    if USE_PYBOOST:
        return mindspore.ops.auto_generate.gen_ops_prim.embedding_op(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq)
    return ops.gather(weight, input, 0)

def rms_norm(input, weight, eps=1e-5):
    return ops.rms_norm(input, weight, eps)[0]

def apply_rotary_pos_emb(query, key, cos, sin, position_ids, cos_format=0):
    return mindspore.ops.auto_generate.gen_ops_def.apply_rotary_pos_emb_(
        query, key, cos, sin, position_ids, cos_format
    )

def pad(input, pad, mode='constant', value=0.0):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.pad(input, pad, mode, value)
    return ops.pad(input, pad, mode, value)
