"""nn functional"""
import numpy as np
from mindspore import ops

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
