"""Parameter initialization functions."""

import math
import numpy as np
import mindspore
from .._tensor import Tensor


def _calculate_fan_in_and_fan_out(tensor):
    """Calculate fan_in and fan_out for a tensor."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity."""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
                  'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def uniform_(tensor, a=0.0, b=1.0):
    """Fill tensor with values from uniform distribution U(a, b)."""
    arr = np.random.uniform(a, b, tensor.shape).astype(np.float32)
    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


def normal_(tensor, mean=0.0, std=1.0):
    """Fill tensor with values from normal distribution N(mean, std)."""
    arr = np.random.normal(mean, std, tensor.shape).astype(np.float32)
    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


def constant_(tensor, val):
    """Fill tensor with constant value."""
    arr = np.full(tensor.shape, val, dtype=np.float32)
    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


def ones_(tensor):
    """Fill tensor with ones."""
    return constant_(tensor, 1.0)


def zeros_(tensor):
    """Fill tensor with zeros."""
    return constant_(tensor, 0.0)


def xavier_uniform_(tensor, gain=1.0):
    """Xavier uniform initialization (Glorot uniform)."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor, gain=1.0):
    """Xavier normal initialization (Glorot normal)."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """Kaiming uniform initialization (He uniform)."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """Kaiming normal initialization (He normal)."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fill tensor with truncated normal distribution."""
    from scipy import stats
    # Use scipy truncated normal
    lower = (a - mean) / std
    upper = (b - mean) / std
    arr = stats.truncnorm.rvs(lower, upper, loc=mean, scale=std, size=tensor.shape)
    arr = arr.astype(np.float32)
    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


def orthogonal_(tensor, gain=1.0):
    """Fill tensor with orthogonal matrix."""
    rows = tensor.shape[0]
    cols = int(np.prod(tensor.shape[1:]))

    if rows < cols:
        raise ValueError("Orthogonal init only works for tensors with rows >= cols")

    # Generate random matrix
    flattened = np.random.normal(0, 1, (rows, cols)).astype(np.float32)
    # QR decomposition
    q, r = np.linalg.qr(flattened)
    # Make Q uniform
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph
    q *= gain

    # Reshape and assign
    arr = q.reshape(tensor.shape).astype(np.float32)
    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


def sparse_(tensor, sparsity, std=0.01):
    """Fill tensor with sparse initialization."""
    if tensor.dim() != 2:
        raise ValueError("Only 2D tensors are supported")

    rows, cols = tensor.shape
    num_zeros = int(np.ceil(sparsity * rows))

    arr = np.random.normal(0, std, (rows, cols)).astype(np.float32)
    for col_idx in range(cols):
        row_indices = np.random.permutation(rows)[:num_zeros]
        arr[row_indices, col_idx] = 0

    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


def eye_(tensor):
    """Fill tensor with identity matrix."""
    if tensor.dim() != 2:
        raise ValueError("Only 2D tensors are supported")

    arr = np.eye(tensor.shape[0], tensor.shape[1], dtype=np.float32)
    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


def dirac_(tensor, groups=1):
    """Fill tensor with Dirac delta function initialization."""
    dimensions = tensor.dim()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only 3D, 4D and 5D tensors are supported")

    sizes = tensor.shape
    if sizes[0] % groups != 0:
        raise ValueError("dim 0 must be divisible by groups")

    arr = np.zeros(sizes, dtype=np.float32)
    out_channels_per_grp = sizes[0] // groups
    in_channels_per_grp = sizes[1]
    min_dim = min(out_channels_per_grp, in_channels_per_grp)

    for g in range(groups):
        for d in range(min_dim):
            if dimensions == 3:  # Conv1d
                arr[g * out_channels_per_grp + d, d, sizes[2] // 2] = 1
            elif dimensions == 4:  # Conv2d
                arr[g * out_channels_per_grp + d, d, sizes[2] // 2, sizes[3] // 2] = 1
            elif dimensions == 5:  # Conv3d
                arr[g * out_channels_per_grp + d, d, sizes[2] // 2, sizes[3] // 2, sizes[4] // 2] = 1

    flat = arr.ravel()
    tensor._storage._ms_tensor = mindspore.Tensor(flat)
    tensor._version += 1
    return tensor


# Aliases without trailing underscore (for compatibility)
uniform = uniform_
normal = normal_
constant = constant_
ones = ones_
zeros = zeros_
xavier_uniform = xavier_uniform_
xavier_normal = xavier_normal_
kaiming_uniform = kaiming_uniform_
kaiming_normal = kaiming_normal_
trunc_normal = trunc_normal_
orthogonal = orthogonal_
sparse = sparse_
eye = eye_
dirac = dirac_
