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


def celu(input, alpha=1.0, inplace=False):
    """Apply CELU activation: max(0,x) + min(0, alpha*(exp(x/alpha)-1))."""
    import numpy as np
    x = input.numpy()
    result = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return Tensor(result.astype(np.float32))


def selu(input, inplace=False):
    """Apply SELU activation."""
    import numpy as np
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = input.numpy()
    result = scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
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
                                  is_causal=False, scale=None, enable_gqa=False):
    """Scaled dot product attention with autograd support.

    Args:
        enable_gqa: Enable grouped query attention optimization (ignored, provided for compatibility)
    """
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

    # Handle string padding values
    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            _, _, k_len = w.shape
            padding = (k_len - 1) // 2
        else:
            raise ValueError(f"Unknown padding mode: {padding}")
    elif isinstance(padding, tuple):
        padding = padding[0]

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
    """2D convolution with groups support using scipy for efficiency."""
    import numpy as np
    from scipy.ndimage import convolve

    x = input.numpy()
    w = weight.numpy()

    stride = stride if isinstance(stride, tuple) else (stride, stride)
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    # Handle string padding values (can be single string or tuple of strings)
    if isinstance(padding, str):
        if padding == 'valid':
            padding = (0, 0)
        elif padding == 'same':
            _, _, k_h, k_w = w.shape
            padding = ((k_h - 1) // 2, (k_w - 1) // 2)
        else:
            raise ValueError(f"Unknown padding mode: {padding}")
    elif isinstance(padding, tuple) and len(padding) > 0 and isinstance(padding[0], str):
        # Handle tuple of strings like ('valid', 'valid')
        _, _, k_h, k_w = w.shape
        pad_h = 0 if padding[0] == 'valid' else (k_h - 1) // 2
        pad_w = 0 if padding[1] == 'valid' else (k_w - 1) // 2
        padding = (pad_h, pad_w)
    elif not isinstance(padding, tuple):
        padding = (padding, padding)

    # Pad input
    if padding[0] > 0 or padding[1] > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    batch, in_ch, in_h, in_w = x.shape
    out_ch, ch_per_group, k_h, k_w = w.shape

    # Handle dilation
    if dilation[0] > 1 or dilation[1] > 1:
        dilated_k_h = (k_h - 1) * dilation[0] + 1
        dilated_k_w = (k_w - 1) * dilation[1] + 1
        dilated_w = np.zeros((out_ch, ch_per_group, dilated_k_h, dilated_k_w), dtype=w.dtype)
        dilated_w[:, :, ::dilation[0], ::dilation[1]] = w
        w = dilated_w
        k_h, k_w = dilated_k_h, dilated_k_w

    out_h = (in_h - k_h) // stride[0] + 1
    out_w = (in_w - k_w) // stride[1] + 1

    # Calculate channels per group
    in_ch_per_group = in_ch // groups
    out_ch_per_group = out_ch // groups

    output = np.zeros((batch, out_ch, out_h, out_w), dtype=np.float32)

    # Use im2col approach for efficiency
    # Extract patches and reshape for matrix multiplication
    for g in range(groups):
        in_start = g * in_ch_per_group
        in_end = in_start + in_ch_per_group
        out_start = g * out_ch_per_group
        out_end = out_start + out_ch_per_group

        # Weight for this group: (out_ch_per_group, in_ch_per_group, k_h, k_w)
        w_group = w[out_start:out_end]

        for b in range(batch):
            # Extract patches: for each output position, extract the input patch
            patches = np.zeros((out_h * out_w, in_ch_per_group * k_h * k_w), dtype=np.float32)
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride[0]
                    w_start = j * stride[1]
                    patch = x[b, in_start:in_end, h_start:h_start+k_h, w_start:w_start+k_w]
                    patches[i * out_w + j] = patch.flatten()

            # Reshape weight: (out_ch_per_group, in_ch_per_group * k_h * k_w)
            w_reshaped = w_group.reshape(out_ch_per_group, -1)

            # Matrix multiply: (out_h * out_w, out_ch_per_group)
            result = patches @ w_reshaped.T

            # Reshape to output
            output[b, out_start:out_end] = result.T.reshape(out_ch_per_group, out_h, out_w)

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
    """Pad tensor - supports both positive (padding) and negative (cropping) values.

    Args:
        input: Input tensor
        pad: Padding sizes (left, right, top, bottom, ...) - negative values crop
        mode: 'constant', 'reflect', 'replicate', or 'circular'
        value: Fill value for constant padding
    """
    import numpy as np
    x = input.numpy()

    # Convert pad from (left, right, top, bottom, ...) to numpy format
    # PyTorch pad: (left, right, top, bottom, front, back, ...)
    # NumPy pad: ((before_1, after_1), (before_2, after_2), ...)
    ndim = x.ndim
    pad_list = list(pad)

    # Separate positive padding and negative cropping
    np_pad = [(0, 0)] * ndim
    slices = [slice(None)] * ndim

    dim_idx = ndim - 1
    for i in range(0, len(pad_list), 2):
        if dim_idx >= 0:
            left_pad = pad_list[i]
            right_pad = pad_list[i + 1]

            # Handle negative padding (cropping)
            left_crop = 0
            right_crop = 0
            if left_pad < 0:
                left_crop = -left_pad
                left_pad = 0
            if right_pad < 0:
                right_crop = -right_pad
                right_pad = 0

            np_pad[dim_idx] = (left_pad, right_pad)

            # Create slice for cropping (if needed)
            if left_crop > 0 or right_crop > 0:
                end = x.shape[dim_idx] - right_crop if right_crop > 0 else None
                slices[dim_idx] = slice(left_crop, end)

            dim_idx -= 1

    # First crop if needed
    x = x[tuple(slices)]

    # Then pad if any positive padding exists
    has_padding = any(p[0] > 0 or p[1] > 0 for p in np_pad)
    if has_padding:
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
    else:
        result = x

    return Tensor(result.astype(input.numpy().dtype))


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


def one_hot(tensor, num_classes=-1):
    """Convert tensor to one-hot encoding.

    Args:
        tensor: LongTensor of shape (*) containing class indices
        num_classes: Total number of classes. If -1, infer from tensor.

    Returns:
        Tensor of shape (*, num_classes) with one-hot encoding.
    """
    import numpy as np
    indices = tensor.numpy().astype(np.int64)

    if num_classes < 0:
        num_classes = int(indices.max()) + 1

    # Flatten indices for easier processing
    flat_indices = indices.flatten()
    num_samples = len(flat_indices)

    # Create one-hot encoded array
    one_hot_array = np.zeros((num_samples, num_classes), dtype=np.float32)
    one_hot_array[np.arange(num_samples), flat_indices] = 1.0

    # Reshape to match input shape + num_classes
    output_shape = indices.shape + (num_classes,)
    result = one_hot_array.reshape(output_shape)

    return Tensor(result)


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """Batch normalization."""
    import numpy as np

    x = input.numpy()
    mean = running_mean.numpy() if running_mean is not None else np.mean(x, axis=(0, 2, 3) if x.ndim == 4 else 0, keepdims=True)
    var = running_var.numpy() if running_var is not None else np.var(x, axis=(0, 2, 3) if x.ndim == 4 else 0, keepdims=True)

    # Reshape for broadcasting
    if x.ndim == 4:  # NCHW
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
    elif x.ndim == 3:  # NCL
        mean = mean.reshape(1, -1, 1)
        var = var.reshape(1, -1, 1)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Scale and shift
    if weight is not None:
        w = weight.numpy()
        if x.ndim == 4:
            w = w.reshape(1, -1, 1, 1)
        elif x.ndim == 3:
            w = w.reshape(1, -1, 1)
        x_norm = x_norm * w

    if bias is not None:
        b = bias.numpy()
        if x.ndim == 4:
            b = b.reshape(1, -1, 1, 1)
        elif x.ndim == 3:
            b = b.reshape(1, -1, 1)
        x_norm = x_norm + b

    return Tensor(x_norm.astype(np.float32))


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """Group normalization."""
    import numpy as np

    x = input.numpy()
    N, C = x.shape[:2]
    spatial_shape = x.shape[2:]

    # Reshape to (N, num_groups, C // num_groups, *spatial)
    G = num_groups
    x_grouped = x.reshape(N, G, C // G, *spatial_shape)

    # Compute mean and var over (C // G, *spatial) for each group
    axes = tuple(range(2, x_grouped.ndim))
    mean = np.mean(x_grouped, axis=axes, keepdims=True)
    var = np.var(x_grouped, axis=axes, keepdims=True)

    # Normalize
    x_norm = (x_grouped - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, *spatial_shape)

    # Scale and shift
    if weight is not None:
        w = weight.numpy().reshape(1, C, *([1] * len(spatial_shape)))
        x_norm = x_norm * w

    if bias is not None:
        b = bias.numpy().reshape(1, C, *([1] * len(spatial_shape)))
        x_norm = x_norm + b

    return Tensor(x_norm.astype(np.float32))


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5):
    """Instance normalization."""
    import numpy as np

    x = input.numpy()
    N, C = x.shape[:2]
    spatial_shape = x.shape[2:]

    # Compute mean and var over spatial dimensions for each sample and channel
    axes = tuple(range(2, x.ndim))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Scale and shift
    if weight is not None:
        w = weight.numpy().reshape(1, C, *([1] * len(spatial_shape)))
        x_norm = x_norm * w

    if bias is not None:
        b = bias.numpy().reshape(1, C, *([1] * len(spatial_shape)))
        x_norm = x_norm + b

    return Tensor(x_norm.astype(np.float32))


def rms_norm(input, normalized_shape, weight=None, eps=1e-5):
    """Root mean square layer normalization."""
    import numpy as np

    x = input.numpy()

    # Compute RMS over normalized dimensions
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # Normalize over the last len(normalized_shape) dimensions
    axes = tuple(range(-len(normalized_shape), 0))
    rms = np.sqrt(np.mean(x ** 2, axis=axes, keepdims=True) + eps)
    x_norm = x / rms

    # Scale
    if weight is not None:
        x_norm = x_norm * weight.numpy()

    return Tensor(x_norm.astype(np.float32))


# ============================================================
# Loss Functions
# ============================================================

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean', label_smoothing=0.0):
    """Cross entropy loss with softmax.

    Args:
        input: Tensor of shape (N, C) or (N, C, d1, d2, ...) containing logits
        target: Tensor of shape (N,) or (N, d1, d2, ...) containing class indices,
                or (N, C) or (N, C, d1, d2, ...) containing probabilities
        weight: Optional weight tensor of shape (C,) for class weighting
        ignore_index: Class index to ignore in loss computation
        reduction: 'none', 'mean', or 'sum'
        label_smoothing: Label smoothing epsilon

    Returns:
        Loss tensor
    """
    return nll_loss(log_softmax(input, dim=1), target, weight=weight,
                    ignore_index=ignore_index, reduction=reduction)


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    """Negative log likelihood loss.

    Args:
        input: Tensor of shape (N, C) containing log-probabilities
        target: Tensor of shape (N,) containing class indices
        weight: Optional weight tensor of shape (C,)
        ignore_index: Class index to ignore
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    import numpy as np

    log_probs = input.numpy()
    targets = target.numpy().astype(np.int64)

    # Create mask for valid (non-ignored) indices BEFORE computing loss
    # This handles both positive and negative ignore_index values
    ignore_mask = targets == ignore_index

    # Replace ignore_index values with 0 temporarily for safe indexing
    # The losses at these positions will be masked out anyway
    safe_targets = np.where(ignore_mask, 0, targets)

    # Handle different input shapes
    if log_probs.ndim == 2:
        # Standard case: (N, C)
        N, C = log_probs.shape
        # Gather log probs for target classes
        batch_indices = np.arange(N)
        losses = -log_probs[batch_indices, safe_targets]
    elif log_probs.ndim == 1:
        # Single sample case
        losses = np.array([-log_probs[safe_targets]])
    else:
        # Higher dimensional case: (N, C, d1, d2, ...)
        N, C = log_probs.shape[:2]
        spatial_shape = log_probs.shape[2:]
        # Reshape for easier indexing
        log_probs_flat = log_probs.transpose(0, *range(2, log_probs.ndim), 1).reshape(-1, C)
        targets_flat = safe_targets.flatten()
        losses = -log_probs_flat[np.arange(len(targets_flat)), targets_flat]
        losses = losses.reshape(N, *spatial_shape)

    # Apply ignore_index mask - zero out losses for ignored indices
    losses = np.where(ignore_mask.flatten() if losses.ndim == 1 else ignore_mask, 0.0, losses)
    valid_count = np.sum(~ignore_mask)

    # Apply weight if provided
    if weight is not None:
        w = weight.numpy()
        target_weights = w[targets.flatten()].reshape(losses.shape)
        losses = losses * target_weights

    # Apply reduction
    if reduction == 'none':
        return Tensor(losses.astype(np.float32))
    elif reduction == 'sum':
        return Tensor(np.array(np.sum(losses), dtype=np.float32))
    else:  # mean
        if ignore_index >= 0 and valid_count > 0:
            return Tensor(np.array(np.sum(losses) / valid_count, dtype=np.float32))
        return Tensor(np.array(np.mean(losses), dtype=np.float32))


def binary_cross_entropy(input, target, weight=None, size_average=None,
                         reduce=None, reduction='mean'):
    """Binary cross entropy loss.

    Args:
        input: Tensor of probabilities (after sigmoid)
        target: Tensor of binary targets
        weight: Optional weight tensor
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    import numpy as np

    probs = input.numpy()
    targets = target.numpy()

    # Clamp for numerical stability
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)

    # BCE: -target * log(prob) - (1 - target) * log(1 - prob)
    losses = -targets * np.log(probs) - (1 - targets) * np.log(1 - probs)

    if weight is not None:
        losses = losses * weight.numpy()

    if reduction == 'none':
        return Tensor(losses.astype(np.float32))
    elif reduction == 'sum':
        return Tensor(np.array(np.sum(losses), dtype=np.float32))
    else:  # mean
        return Tensor(np.array(np.mean(losses), dtype=np.float32))


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                      reduce=None, reduction='mean', pos_weight=None):
    """Binary cross entropy loss with logits (more numerically stable).

    Args:
        input: Tensor of logits (before sigmoid)
        target: Tensor of binary targets
        weight: Optional weight tensor
        reduction: 'none', 'mean', or 'sum'
        pos_weight: Weight for positive examples

    Returns:
        Loss tensor
    """
    import numpy as np

    logits = input.numpy()
    targets = target.numpy()

    # Numerically stable: max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
    max_val = np.maximum(logits, 0)
    losses = max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits)))

    if pos_weight is not None:
        pw = pos_weight.numpy() if hasattr(pos_weight, 'numpy') else pos_weight
        losses = losses * (1 + (pw - 1) * targets)

    if weight is not None:
        losses = losses * weight.numpy()

    if reduction == 'none':
        return Tensor(losses.astype(np.float32))
    elif reduction == 'sum':
        return Tensor(np.array(np.sum(losses), dtype=np.float32))
    else:  # mean
        return Tensor(np.array(np.mean(losses), dtype=np.float32))


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Mean squared error loss.

    Args:
        input: Predictions tensor
        target: Target tensor
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    import numpy as np

    preds = input.numpy()
    targets = target.numpy()

    losses = (preds - targets) ** 2

    if reduction == 'none':
        return Tensor(losses.astype(np.float32))
    elif reduction == 'sum':
        return Tensor(np.array(np.sum(losses), dtype=np.float32))
    else:  # mean
        return Tensor(np.array(np.mean(losses), dtype=np.float32))


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """L1 (mean absolute error) loss.

    Args:
        input: Predictions tensor
        target: Target tensor
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    import numpy as np

    preds = input.numpy()
    targets = target.numpy()

    losses = np.abs(preds - targets)

    if reduction == 'none':
        return Tensor(losses.astype(np.float32))
    elif reduction == 'sum':
        return Tensor(np.array(np.sum(losses), dtype=np.float32))
    else:  # mean
        return Tensor(np.array(np.mean(losses), dtype=np.float32))


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0):
    """Smooth L1 loss (Huber loss).

    Args:
        input: Predictions tensor
        target: Target tensor
        reduction: 'none', 'mean', or 'sum'
        beta: Threshold at which to change between L1 and L2 loss

    Returns:
        Loss tensor
    """
    import numpy as np

    preds = input.numpy()
    targets = target.numpy()

    diff = np.abs(preds - targets)
    losses = np.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    if reduction == 'none':
        return Tensor(losses.astype(np.float32))
    elif reduction == 'sum':
        return Tensor(np.array(np.sum(losses), dtype=np.float32))
    else:  # mean
        return Tensor(np.array(np.mean(losses), dtype=np.float32))


def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
    """Kullback-Leibler divergence loss.

    Args:
        input: Tensor of log-probabilities
        target: Tensor of probabilities (or log-probabilities if log_target=True)
        reduction: 'none', 'mean', 'sum', or 'batchmean'
        log_target: If True, target is log-probabilities

    Returns:
        Loss tensor
    """
    import numpy as np

    log_p = input.numpy()
    if log_target:
        log_q = target.numpy()
        q = np.exp(log_q)
    else:
        q = target.numpy()
        log_q = np.log(q + 1e-10)

    # KL(q||p) = q * (log_q - log_p)
    losses = q * (log_q - log_p)

    if reduction == 'none':
        return Tensor(losses.astype(np.float32))
    elif reduction == 'sum':
        return Tensor(np.array(np.sum(losses), dtype=np.float32))
    elif reduction == 'batchmean':
        return Tensor(np.array(np.sum(losses) / losses.shape[0], dtype=np.float32))
    else:  # mean
        return Tensor(np.array(np.mean(losses), dtype=np.float32))



