"""Weight initialization functions.

Mirrors PyTorch's torch.nn.init module. All functions operate in-place on
the given tensor and return it. Computations are done under no_grad() to
avoid recording these operations in the autograd graph.

These functions are pure Python composing from Tensor in-place methods
(uniform_, normal_, fill_, zero_, etc.) which internally dispatch to the
appropriate backend (CPU/NPU).
"""

import math
import warnings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _no_grad_uniform_(tensor, a, b):
    from .._autograd.grad_mode import no_grad
    with no_grad():
        return tensor.uniform_(a, b)


def _no_grad_normal_(tensor, mean, std):
    from .._autograd.grad_mode import no_grad
    with no_grad():
        return tensor.normal_(mean, std)


def _no_grad_fill_(tensor, val):
    from .._autograd.grad_mode import no_grad
    with no_grad():
        return tensor.fill_(val)


def _no_grad_zero_(tensor):
    from .._autograd.grad_mode import no_grad
    with no_grad():
        return tensor.zero_()


def _calculate_fan_in_and_fan_out(tensor):
    """Calculate fan_in and fan_out for a weight tensor.

    Supports Linear (2D), Conv (3D+), and scalar/vector tensors.
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(
            f"Mode {mode} not supported, please use one of {valid_modes}"
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity.

    Args:
        nonlinearity: the non-linear function (nn.functional name)
        param: optional parameter for the non-linear function

    Returns:
        float: the recommended gain value
    """
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d',
        'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, (int, float)):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def uniform_(tensor, a=0.0, b=1.0):
    r"""Fill the input Tensor with values drawn from U(a, b).

    Args:
        tensor: an n-dimensional Tensor
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor, mean=0.0, std=1.0):
    r"""Fill the input Tensor with values drawn from N(mean, std^2).

    Args:
        tensor: an n-dimensional Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    return _no_grad_normal_(tensor, mean, std)


def constant_(tensor, val):
    r"""Fill the input Tensor with the value ``val``.

    Args:
        tensor: an n-dimensional Tensor
        val: the value to fill the tensor with

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    return _no_grad_fill_(tensor, val)


def ones_(tensor):
    r"""Fill the input Tensor with the scalar value 1.

    Args:
        tensor: an n-dimensional Tensor

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    return _no_grad_fill_(tensor, 1.0)


def zeros_(tensor):
    r"""Fill the input Tensor with the scalar value 0.

    Args:
        tensor: an n-dimensional Tensor

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    return _no_grad_zero_(tensor)


def eye_(tensor):
    r"""Fill the 2-dimensional input Tensor with the identity matrix.

    Args:
        tensor: a 2-dimensional Tensor

    Returns:
        Tensor: the input tensor
    """
    if tensor.dim() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    from .._autograd.grad_mode import no_grad
    with no_grad():
        from .._creation import eye as _eye
        _eye(tensor.shape[0], tensor.shape[1],
             dtype=tensor.dtype, device=tensor.device, out=tensor)
    return tensor


def dirac_(tensor, groups=1):
    r"""Fill the {3, 4, 5}-dimensional input Tensor with the Dirac delta function.

    Args:
        tensor: a {3, 4, 5}-dimensional Tensor
        groups: number of groups in the conv layer (default: 1)

    Returns:
        Tensor: the input tensor
    """
    dimensions = tensor.dim()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    sizes = tensor.shape
    if sizes[0] % groups != 0:
        raise ValueError('dimension 0 must be divisible by groups')

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    from .._autograd.grad_mode import no_grad
    import numpy as np
    with no_grad():
        arr = np.zeros(sizes, dtype=np.float32)
        for g in range(groups):
            for d in range(min_dim):
                if dimensions == 3:
                    arr[g * out_chans_per_grp + d, d, sizes[2] // 2] = 1.0
                elif dimensions == 4:
                    arr[g * out_chans_per_grp + d, d, sizes[2] // 2, sizes[3] // 2] = 1.0
                elif dimensions == 5:
                    arr[g * out_chans_per_grp + d, d, sizes[2] // 2, sizes[3] // 2, sizes[4] // 2] = 1.0
        from .._creation import tensor as create_tensor
        src = create_tensor(arr, dtype=tensor.dtype, device=tensor.device)
        tensor.copy_(src)
    return tensor


def xavier_uniform_(tensor, gain=1.0):
    r"""Fill the input Tensor with values using Xavier uniform initialization.

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional Tensor
        gain: an optional scaling factor

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)


def xavier_normal_(tensor, gain=1.0):
    r"""Fill the input Tensor with values using Xavier normal initialization.

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional Tensor
        gain: an optional scaling factor

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fill the input Tensor with values using Kaiming uniform initialization.

    Also known as He initialization.

    Args:
        tensor: an n-dimensional Tensor
        a: the negative slope of the rectifier used after this layer
            (only used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``
        nonlinearity: the non-linear function

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fill the input Tensor with values using Kaiming normal initialization.

    Also known as He initialization.

    Args:
        tensor: an n-dimensional Tensor
        a: the negative slope of the rectifier used after this layer
            (only used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``
        nonlinearity: the non-linear function

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return _no_grad_normal_(tensor, 0.0, std)


def orthogonal_(tensor, gain=1.0):
    r"""Fill the input Tensor with a (semi) orthogonal matrix.

    The input tensor must have at least 2 dimensions. For tensors with
    more than 2 dimensions, the extra dimensions are flattened.

    Args:
        tensor: an n-dimensional Tensor, where n >= 2
        gain: optional scaling factor

    Returns:
        Tensor: the input tensor
    """
    if tensor.dim() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    from .._autograd.grad_mode import no_grad
    import numpy as np

    with no_grad():
        rows = tensor.shape[0]
        cols = 1
        for s in tensor.shape[1:]:
            cols *= s

        # Generate random matrix
        flat_shape = (rows, cols) if rows >= cols else (cols, rows)
        # Use numpy for QR decomposition (not available through dispatch yet)
        from .._random import _get_cpu_rng
        rng = _get_cpu_rng()
        a = rng.standard_normal(flat_shape)
        q, r = np.linalg.qr(a)
        # Make Q uniform according to https://arxiv.org/abs/math-ph/0609050
        d = np.diag(r)
        ph = np.sign(d)
        q *= ph

        if rows < cols:
            q = q.T

        q = q.reshape(tensor.shape)
        q *= gain

        # Copy result back to the tensor
        if tensor.device.type == "cpu":
            from .._dtype import to_numpy_dtype
            arr = tensor._numpy_view()
            arr[:] = q.astype(to_numpy_dtype(tensor.dtype))
        else:
            # For non-CPU: create a CPU tensor, copy data, then move to device
            from .._creation import tensor as create_tensor
            from .._dtype import to_numpy_dtype
            q_arr = q.astype(to_numpy_dtype(tensor.dtype))
            q_tensor = create_tensor(q_arr, dtype=tensor.dtype, device=tensor.device)
            tensor.copy_(q_tensor)

    return tensor


def sparse_(tensor, sparsity, std=0.01):
    r"""Fill the 2D input Tensor as a sparse matrix.

    The non-zero elements will be drawn from N(0, std^2).

    Args:
        tensor: an n-dimensional Tensor
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution

    Returns:
        Tensor: the input tensor
    """
    if tensor.dim() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    from .._autograd.grad_mode import no_grad
    import numpy as np

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    with no_grad():
        # Build the sparse matrix in numpy, then transfer to device
        from .._random import _get_cpu_rng
        from .._dtype import to_numpy_dtype
        rng = _get_cpu_rng()
        arr = rng.normal(0, std, (rows, cols)).astype(to_numpy_dtype(tensor.dtype))
        for col_idx in range(cols):
            row_indices = rng.permutation(rows)[:num_zeros]
            arr[row_indices, col_idx] = 0.0
        from .._creation import tensor as create_tensor
        src = create_tensor(arr, dtype=tensor.dtype, device=tensor.device)
        tensor.copy_(src)

    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fill the input Tensor with values drawn from a truncated normal distribution.

    Values are effectively drawn from N(mean, std^2) and any values outside
    [a, b] are redrawn until they fall within the bounds.

    The method used for generating the random values works best when
    a <= mean <= b.

    Args:
        tensor: an n-dimensional Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Returns:
        Tensor: the input tensor
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    from .._autograd.grad_mode import no_grad

    with no_grad():
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        # Uses the inverse CDF method:
        # X = Phi^{-1}(U * Phi((b-mean)/std) + (1-U) * Phi((a-mean)/std)) * std + mean

        # Normalize bounds
        l = (a - mean) / std
        u = (b - mean) / std

        # Fill with uniform [0, 1]
        tensor.uniform_(0, 1)

        # Use erfinv to transform: erfinv maps (-1, 1) -> (-inf, inf)
        # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        # Phi_inv(p) = sqrt(2) * erfinv(2*p - 1)

        # Compute Phi(l) and Phi(u)
        _sqrt2 = math.sqrt(2.0)
        # Phi(x) = 0.5 * erfc(-x / sqrt(2))
        # We use the approximation: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        phi_l = 0.5 * (1.0 + math.erf(l / _sqrt2))
        phi_u = 0.5 * (1.0 + math.erf(u / _sqrt2))

        # Rescale uniform samples to [Phi(l), Phi(u)]
        # tensor = tensor * (phi_u - phi_l) + phi_l
        tensor.mul_(phi_u - phi_l)
        tensor.add_(phi_l)

        # Clamp to (0+eps, 1-eps) to avoid erfinv(1) = inf
        eps = 1e-7
        tensor.clamp_(eps, 1.0 - eps)

        # Apply inverse CDF: Phi^{-1}(p) = sqrt(2) * erfinv(2*p - 1)
        # Transform to erfinv domain: 2*tensor - 1
        tensor.mul_(2.0)
        tensor.sub_(1.0)

        # erfinv
        tensor.erfinv_()

        # Scale by sqrt(2) * std + mean
        tensor.mul_(_sqrt2 * std)
        tensor.add_(mean)

        # Final clamp to [a, b] for safety
        tensor.clamp_(a, b)

    return tensor
