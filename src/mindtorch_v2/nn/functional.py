"""Functional operations for neural networks."""

from .._dispatch import dispatch
from .._tensor import Tensor


def relu(input, inplace=False):
    """Apply ReLU activation."""
    return dispatch("relu", input)


def gelu(input, approximate='none'):
    """Apply GELU activation."""
    return dispatch("gelu", input, approximate=approximate)


def silu(input, inplace=False):
    """Apply SiLU/Swish activation."""
    return dispatch("silu", input)


def sigmoid(input):
    """Apply sigmoid activation."""
    return dispatch("sigmoid", input)


def tanh(input):
    """Apply tanh activation."""
    return dispatch("tanh", input)


def softmax(input, dim=None, dtype=None):
    """Apply softmax."""
    return dispatch("softmax", input, dim=dim)


def log_softmax(input, dim=None, dtype=None):
    """Apply log softmax."""
    return dispatch("log_softmax", input, dim=dim)


def linear(input, weight, bias=None):
    """Apply linear transformation: y = xW^T + b."""
    output = dispatch("matmul", input, weight.t())
    if bias is not None:
        output = dispatch("add", output, bias)
    return output


def embedding(input, weight, padding_idx=None, max_norm=None,
              norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Look up embeddings."""
    return dispatch("embedding", input, weight)


def dropout(input, p=0.5, training=True, inplace=False):
    """Apply dropout."""
    if not training or p == 0:
        return input
    return dispatch("dropout", input, p=p, training=training)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Apply layer normalization."""
    return dispatch("layer_norm", input, normalized_shape, weight, bias, eps)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    """Apply LeakyReLU activation."""
    import numpy as np
    x = input.numpy()
    result = np.where(x > 0, x, x * negative_slope)
    return Tensor(result.astype(np.float32))


def elu(input, alpha=1.0, inplace=False):
    """Apply ELU activation."""
    import numpy as np
    x = input.numpy()
    result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return Tensor(result.astype(np.float32))


def prelu(input, weight):
    """Apply PReLU activation."""
    import numpy as np
    x = input.numpy()
    w = weight.numpy()
    result = np.where(x > 0, x, x * w)
    return Tensor(result.astype(np.float32))


def mish(input, inplace=False):
    """Apply Mish activation: x * tanh(softplus(x))."""
    import numpy as np
    x = input.numpy()
    result = x * np.tanh(np.log1p(np.exp(x)))
    return Tensor(result.astype(np.float32))


def hardswish(input, inplace=False):
    """Apply Hardswish activation."""
    import numpy as np
    x = input.numpy()
    result = x * np.clip(x + 3, 0, 6) / 6
    return Tensor(result.astype(np.float32))


def hardsigmoid(input, inplace=False):
    """Apply Hardsigmoid activation."""
    import numpy as np
    x = input.numpy()
    result = np.clip(x / 6 + 0.5, 0, 1)
    return Tensor(result.astype(np.float32))


def softplus(input, beta=1.0, threshold=20.0):
    """Apply Softplus activation."""
    import numpy as np
    x = input.numpy()
    # Use threshold for numerical stability
    result = np.where(beta * x > threshold, x, np.log1p(np.exp(beta * x)) / beta)
    return Tensor(result.astype(np.float32))


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    """Apply Hardtanh activation."""
    import numpy as np
    x = input.numpy()
    result = np.clip(x, min_val, max_val)
    return Tensor(result.astype(np.float32))


def relu6(input, inplace=False):
    """Apply ReLU6 activation."""
    import numpy as np
    x = input.numpy()
    result = np.clip(x, 0, 6)
    return Tensor(result.astype(np.float32))


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None):
    """Scaled dot product attention with autograd support."""
    import math
    from .._functional import matmul

    # Get shapes
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale

    # Compute attention weights: query @ key.T
    # query: (..., L, E), key: (..., S, E) -> key.T: (..., E, S)
    key_t = key.transpose(-2, -1)
    attn_weights = matmul(query, key_t) * scale_factor

    # Apply causal mask if needed
    if is_causal:
        import numpy as np
        causal_mask = np.triu(np.ones((L, S), dtype=np.float32) * float('-inf'), k=1)
        causal_mask_tensor = Tensor(causal_mask)
        attn_weights = attn_weights + causal_mask_tensor

    # Apply attention mask if provided
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    # Softmax along last dimension (use local softmax function)
    attn_weights = softmax(attn_weights, dim=-1)

    # Apply dropout if needed (skip for inference/training without dropout)
    # TODO: add dropout support

    # Compute output: attn_weights @ value
    output = matmul(attn_weights, value)

    return output


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """1D convolution."""
    import numpy as np
    from scipy import signal

    # Simple implementation using numpy
    x = input.numpy()
    w = weight.numpy()

    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding

    # Pad input
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')

    batch, in_ch, in_len = x.shape
    out_ch, _, k_len = w.shape
    out_len = (in_len - k_len) // stride + 1

    output = np.zeros((batch, out_ch, out_len), dtype=np.float32)

    for b in range(batch):
        for oc in range(out_ch):
            for ic in range(in_ch):
                for i in range(out_len):
                    start = i * stride
                    output[b, oc, i] += np.sum(x[b, ic, start:start+k_len] * w[oc, ic])

    if bias is not None:
        output += bias.numpy().reshape(1, -1, 1)

    return Tensor(output)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """2D convolution."""
    import numpy as np

    x = input.numpy()
    w = weight.numpy()

    stride = stride if isinstance(stride, tuple) else (stride, stride)
    padding = padding if isinstance(padding, tuple) else (padding, padding)

    # Pad input
    if padding[0] > 0 or padding[1] > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    batch, in_ch, in_h, in_w = x.shape
    out_ch, _, k_h, k_w = w.shape
    out_h = (in_h - k_h) // stride[0] + 1
    out_w = (in_w - k_w) // stride[1] + 1

    output = np.zeros((batch, out_ch, out_h, out_w), dtype=np.float32)

    for b in range(batch):
        for oc in range(out_ch):
            for ic in range(in_ch):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride[0]
                        w_start = j * stride[1]
                        output[b, oc, i, j] += np.sum(
                            x[b, ic, h_start:h_start+k_h, w_start:w_start+k_w] * w[oc, ic]
                        )

    if bias is not None:
        output += bias.numpy().reshape(1, -1, 1, 1)

    return Tensor(output)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """3D convolution - stub implementation."""
    raise NotImplementedError("conv3d not yet implemented")


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    """1D transposed convolution - stub implementation."""
    raise NotImplementedError("conv_transpose1d not yet implemented")


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    """2D transposed convolution - stub implementation."""
    raise NotImplementedError("conv_transpose2d not yet implemented")


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    """3D transposed convolution - stub implementation."""
    raise NotImplementedError("conv_transpose3d not yet implemented")


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    """Sample from input using grid coordinates - stub implementation."""
    raise NotImplementedError("grid_sample not yet implemented")


def interpolate(input, size=None, scale_factor=None, mode='nearest',
                align_corners=None, recompute_scale_factor=None, antialias=False):
    """Interpolate/resize tensor - stub implementation."""
    raise NotImplementedError("interpolate not yet implemented")


def pad(input, pad, mode='constant', value=0):
    """Pad tensor - basic implementation."""
    import numpy as np
    x = input.numpy()

    # Convert pad from (left, right, top, bottom, ...) to numpy format
    # PyTorch pad: (left, right, top, bottom, front, back, ...)
    # NumPy pad: ((before_1, after_1), (before_2, after_2), ...)
    ndim = x.ndim
    pad_list = list(pad)

    # Pad is applied from last dimension to first
    np_pad = [(0, 0)] * ndim
    dim_idx = ndim - 1
    for i in range(0, len(pad_list), 2):
        if dim_idx >= 0:
            np_pad[dim_idx] = (pad_list[i], pad_list[i+1])
            dim_idx -= 1

    if mode == 'constant':
        result = np.pad(x, np_pad, mode='constant', constant_values=value)
    elif mode == 'reflect':
        result = np.pad(x, np_pad, mode='reflect')
    elif mode == 'replicate':
        result = np.pad(x, np_pad, mode='edge')
    elif mode == 'circular':
        result = np.pad(x, np_pad, mode='wrap')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return Tensor(result.astype(x.dtype))


def affine_grid(theta, size, align_corners=None):
    """Generate affine grid - stub implementation."""
    raise NotImplementedError("affine_grid not yet implemented")


def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """Upsample tensor - stub, delegates to interpolate."""
    return interpolate(input, size=size, scale_factor=scale_factor,
                      mode=mode, align_corners=align_corners)


def upsample_nearest(input, size=None, scale_factor=None):
    """Upsample using nearest neighbor - stub."""
    return interpolate(input, size=size, scale_factor=scale_factor, mode='nearest')


def upsample_bilinear(input, size=None, scale_factor=None):
    """Upsample using bilinear interpolation - stub."""
    return interpolate(input, size=size, scale_factor=scale_factor,
                      mode='bilinear', align_corners=True)


