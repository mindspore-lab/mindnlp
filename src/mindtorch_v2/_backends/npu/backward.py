"""NPU backward functions using ACLNN large kernels.

Each function takes mindtorch tensors (not raw pointers) and returns
gradient tensors.  This avoids compositing multiple small dispatch ops
and instead calls the single fused ACLNN backward kernel directly.
"""

from ..._storage import npu_typed_storage_from_ptr
from . import aclnn
from . import runtime as npu_runtime
from . import state as npu_state


# ---------------------------------------------------------------
# Shared helpers (re-used from ops.py patterns)
# ---------------------------------------------------------------

def _unwrap_storage(tensor):
    if tensor.storage().device.type != "npu":
        raise ValueError("Expected NPU storage for NPU backward op")
    return tensor.storage()


def _wrap_tensor(storage, shape, stride):
    from ..._tensor import Tensor
    return Tensor(storage, shape, stride)


def _dtype_itemsize(dtype):
    size = getattr(dtype, "itemsize", None)
    if size is not None:
        return int(size)
    name = getattr(dtype, "name", None) or str(dtype).split(".")[-1]
    return {"float16": 2, "float32": 4, "float64": 8, "bfloat16": 2,
            "int8": 1, "int16": 2, "int32": 4, "int64": 8,
            "uint8": 1, "bool": 1}.get(name, 4)


def _numel(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size


def _get_runtime_stream(tensor):
    """Return (runtime, stream) for a tensor's device."""
    dev_idx = tensor.device.index or 0
    runtime = npu_runtime.get_runtime(dev_idx)
    stream = npu_state.current_stream(dev_idx)
    return runtime, stream


def _alloc_like(tensor, runtime, shape=None, dtype=None):
    """Allocate an output buffer with the same shape/dtype as *tensor*.

    Returns (ptr, shape, stride, numel).
    """
    out_shape = shape if shape is not None else tensor.shape
    out_dtype = dtype if dtype is not None else tensor.dtype
    out_stride = npu_runtime._contiguous_stride(out_shape)
    numel = _numel(out_shape)
    byte_size = max(numel, 1) * _dtype_itemsize(out_dtype)
    ptr = npu_runtime._alloc_device(byte_size, runtime=runtime)
    return ptr, out_shape, out_stride, numel


def _wrap_output(ptr, numel, shape, stride, dtype, device):
    """Wrap a raw device pointer into a mindtorch tensor."""
    storage = npu_typed_storage_from_ptr(ptr, max(numel, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride)


# ---------------------------------------------------------------
# Simple activation backward (need forward input)
# ---------------------------------------------------------------

def npu_relu_backward(grad, saved_input):
    """ReLU backward via aclnnThresholdBackward with threshold=0."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.threshold_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        out_stride,
        grad.dtype,
        0.0,  # threshold
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_gelu_backward(grad, saved_input):
    """GELU backward via aclnnGeluBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.gelu_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        out_stride,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_silu_backward(grad, saved_input):
    """SiLU backward via aclnnSiluBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.silu_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        out_stride,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


# ---------------------------------------------------------------
# Output-based activation backward (need forward output from input)
# ---------------------------------------------------------------

def npu_sigmoid_backward(grad, saved_input):
    """Sigmoid backward: compute sigmoid(input) first, then call
    aclnnSigmoidBackward(grad, sigmoid_output)."""
    runtime, stream = _get_runtime_stream(grad)

    # Compute sigmoid(saved_input) -> fwd_output
    fwd_ptr, fwd_shape, fwd_stride, fwd_numel = _alloc_like(saved_input, runtime)
    aclnn.sigmoid(
        _unwrap_storage(saved_input).data_ptr(),
        fwd_ptr,
        saved_input.shape,
        saved_input.stride,
        saved_input.dtype,
        runtime,
        stream=stream.stream,
    )

    # Backward: grad_input = aclnnSigmoidBackward(grad, sigmoid_output)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)
    aclnn.sigmoid_backward(
        _unwrap_storage(grad).data_ptr(),
        fwd_ptr,
        out_ptr,
        grad.shape,
        grad.stride,
        fwd_stride,
        out_stride,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    # Keep fwd_ptr alive via storage until backward completes
    _fwd_storage = npu_typed_storage_from_ptr(fwd_ptr, max(fwd_numel, 1), saved_input.dtype, device=saved_input.device)

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_tanh_backward(grad, saved_input):
    """Tanh backward: compute tanh(input) first, then call
    aclnnTanhBackward(grad, tanh_output)."""
    runtime, stream = _get_runtime_stream(grad)

    # Compute tanh(saved_input) -> fwd_output
    fwd_ptr, fwd_shape, fwd_stride, fwd_numel = _alloc_like(saved_input, runtime)
    aclnn.tanh(
        _unwrap_storage(saved_input).data_ptr(),
        fwd_ptr,
        saved_input.shape,
        saved_input.stride,
        saved_input.dtype,
        runtime,
        stream=stream.stream,
    )

    # Backward: grad_input = aclnnTanhBackward(grad, tanh_output)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)
    aclnn.tanh_backward(
        _unwrap_storage(grad).data_ptr(),
        fwd_ptr,
        out_ptr,
        grad.shape,
        grad.stride,
        fwd_stride,
        out_stride,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    # Keep fwd_ptr alive
    _fwd_storage = npu_typed_storage_from_ptr(fwd_ptr, max(fwd_numel, 1), saved_input.dtype, device=saved_input.device)

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


# ---------------------------------------------------------------
# Softmax backward (need forward output + dim)
# ---------------------------------------------------------------

def npu_softmax_backward(grad, saved_input, dim):
    """Softmax backward: compute softmax(input, dim) first, then call
    aclnnSoftmaxBackward(grad, softmax_output, dim)."""
    runtime, stream = _get_runtime_stream(grad)

    # Compute softmax(saved_input, dim) -> fwd_output
    fwd_ptr, fwd_shape, fwd_stride, fwd_numel = _alloc_like(saved_input, runtime)
    aclnn.softmax(
        _unwrap_storage(saved_input).data_ptr(),
        fwd_ptr,
        saved_input.shape,
        saved_input.stride,
        saved_input.dtype,
        dim,
        runtime,
        stream=stream.stream,
    )

    # Backward
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)
    aclnn.softmax_backward(
        _unwrap_storage(grad).data_ptr(),
        fwd_ptr,
        out_ptr,
        grad.shape,
        grad.stride,
        fwd_stride,
        out_stride,
        grad.dtype,
        dim,
        runtime,
        stream=stream.stream,
    )

    # Keep fwd_ptr alive
    _fwd_storage = npu_typed_storage_from_ptr(fwd_ptr, max(fwd_numel, 1), saved_input.dtype, device=saved_input.device)

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_log_softmax_backward(grad, saved_input, dim):
    """Log-softmax backward: compute log_softmax(input, dim) first, then call
    aclnnLogSoftmaxBackward(grad, log_softmax_output, dim)."""
    runtime, stream = _get_runtime_stream(grad)

    # Compute log_softmax(saved_input, dim) -> fwd_output
    fwd_ptr, fwd_shape, fwd_stride, fwd_numel = _alloc_like(saved_input, runtime)
    aclnn.log_softmax(
        _unwrap_storage(saved_input).data_ptr(),
        fwd_ptr,
        saved_input.shape,
        saved_input.stride,
        saved_input.dtype,
        dim,
        runtime,
        stream=stream.stream,
    )

    # Backward
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)
    aclnn.log_softmax_backward(
        _unwrap_storage(grad).data_ptr(),
        fwd_ptr,
        out_ptr,
        grad.shape,
        grad.stride,
        fwd_stride,
        out_stride,
        grad.dtype,
        dim,
        runtime,
        stream=stream.stream,
    )

    # Keep fwd_ptr alive
    _fwd_storage = npu_typed_storage_from_ptr(fwd_ptr, max(fwd_numel, 1), saved_input.dtype, device=saved_input.device)

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


# ---------------------------------------------------------------
# Embedding backward
# ---------------------------------------------------------------

def npu_embedding_backward(grad, saved_weight, saved_indices, padding_idx=None,
                           scale_grad_by_freq=False):
    """Embedding backward via aclnnEmbeddingDenseBackward.

    Returns grad_weight with the same shape as *saved_weight*.
    """
    runtime, stream = _get_runtime_stream(grad)

    num_weights = saved_weight.shape[0]
    gw_shape = saved_weight.shape
    gw_stride = npu_runtime._contiguous_stride(gw_shape)
    gw_numel = _numel(gw_shape)
    gw_ptr = npu_runtime._alloc_device(
        max(gw_numel, 1) * _dtype_itemsize(grad.dtype), runtime=runtime
    )

    aclnn.embedding_dense_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_indices).data_ptr(),
        gw_ptr,
        grad.shape, grad.stride,
        saved_indices.shape, saved_indices.stride,
        gw_shape, gw_stride,
        grad.dtype,
        saved_indices.dtype,
        num_weights,
        padding_idx if padding_idx is not None else -1,
        scale_grad_by_freq,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(gw_ptr, gw_numel, gw_shape, gw_stride, grad.dtype, grad.device)


# ---------------------------------------------------------------
# Normalization backward
# ---------------------------------------------------------------

def npu_layer_norm_backward(grad, saved_input, backward_data, normalized_shape,
                            weight=None, bias=None, eps=1e-5):
    """Layer norm backward via aclnnLayerNormBackward.

    *backward_data* is the ``_backward_data`` dict stored on the forward
    output tensor.  It must contain ``mean_ptr``, ``rstd_ptr``,
    ``stats_shape``, ``stats_stride``.
    """
    runtime, stream = _get_runtime_stream(grad)

    mean_ptr = backward_data["mean_ptr"]
    rstd_ptr = backward_data["rstd_ptr"]
    stats_shape = backward_data["stats_shape"]
    stats_stride = backward_data["stats_stride"]

    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # grad_input (same shape as input)
    gi_ptr, gi_shape, gi_stride, gi_numel = _alloc_like(saved_input, runtime)

    # grad_weight / grad_bias (same shape as weight / bias if they exist)
    gw_ptr = None
    gw_shape = ()
    gw_stride = ()
    if weight is not None:
        gw_numel = _numel(weight.shape)
        gw_ptr = npu_runtime._alloc_device(
            max(gw_numel, 1) * _dtype_itemsize(weight.dtype), runtime=runtime
        )
        gw_shape = weight.shape
        gw_stride = npu_runtime._contiguous_stride(gw_shape)

    gb_ptr = None
    gb_shape = ()
    gb_stride = ()
    if bias is not None:
        gb_numel = _numel(bias.shape)
        gb_ptr = npu_runtime._alloc_device(
            max(gb_numel, 1) * _dtype_itemsize(bias.dtype), runtime=runtime
        )
        gb_shape = bias.shape
        gb_stride = npu_runtime._contiguous_stride(gb_shape)

    aclnn.layer_norm_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        mean_ptr,
        rstd_ptr,
        _unwrap_storage(weight).data_ptr() if weight is not None else None,
        _unwrap_storage(bias).data_ptr() if bias is not None else None,
        gi_ptr,
        gw_ptr,
        gb_ptr,
        saved_input.shape, saved_input.stride,
        stats_shape, stats_stride,
        weight.shape if weight is not None else (),
        weight.stride if weight is not None else (),
        bias.shape if bias is not None else (),
        bias.stride if bias is not None else (),
        tuple(normalized_shape),
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    grad_input = _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride, grad.dtype, grad.device)

    grad_weight = None
    if gw_ptr is not None:
        gw_numel_val = _numel(gw_shape)
        grad_weight = _wrap_output(gw_ptr, gw_numel_val, gw_shape, gw_stride, grad.dtype, grad.device)

    grad_bias = None
    if gb_ptr is not None:
        gb_numel_val = _numel(gb_shape)
        grad_bias = _wrap_output(gb_ptr, gb_numel_val, gb_shape, gb_stride, grad.dtype, grad.device)

    return grad_input, grad_weight, grad_bias


def npu_batch_norm_backward(grad, saved_input, backward_data, weight=None,
                            running_mean=None, running_var=None):
    """Batch norm backward via aclnnBatchNormBackward.

    *backward_data* must contain ``save_mean_ptr``, ``save_invstd_ptr``,
    ``C``, ``training``, ``eps``.
    """
    runtime, stream = _get_runtime_stream(grad)

    save_mean_ptr = backward_data["save_mean_ptr"]
    save_invstd_ptr = backward_data["save_invstd_ptr"]
    C = backward_data["C"]
    training = backward_data["training"]
    eps = backward_data["eps"]

    stat_shape = (C,)
    stat_stride = (1,)

    # grad_input (same shape as input)
    gi_ptr, gi_shape, gi_stride, gi_numel = _alloc_like(saved_input, runtime)

    # grad_weight / grad_bias (shape = (C,))
    gw_ptr = None
    gw_shape = ()
    gw_stride = ()
    gb_ptr = None
    gb_shape = ()
    gb_stride = ()
    if weight is not None:
        gw_ptr = npu_runtime._alloc_device(
            max(C, 1) * _dtype_itemsize(grad.dtype), runtime=runtime
        )
        gw_shape = weight.shape
        gw_stride = npu_runtime._contiguous_stride(gw_shape)
        gb_ptr = npu_runtime._alloc_device(
            max(C, 1) * _dtype_itemsize(grad.dtype), runtime=runtime
        )
        gb_shape = weight.shape  # bias shape == weight shape for BN
        gb_stride = npu_runtime._contiguous_stride(gb_shape)

    output_mask = [True, gw_ptr is not None, gb_ptr is not None]

    aclnn.batch_norm_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        _unwrap_storage(weight).data_ptr() if weight is not None else None,
        _unwrap_storage(running_mean).data_ptr() if running_mean is not None else None,
        _unwrap_storage(running_var).data_ptr() if running_var is not None else None,
        save_mean_ptr,
        save_invstd_ptr,
        gi_ptr,
        gw_ptr,
        gb_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        weight.shape if weight is not None else (),
        weight.stride if weight is not None else (),
        running_mean.shape if running_mean is not None else (),
        running_mean.stride if running_mean is not None else (),
        running_var.shape if running_var is not None else (),
        running_var.stride if running_var is not None else (),
        stat_shape, stat_stride,
        stat_shape, stat_stride,
        gi_shape, gi_stride,
        gw_shape, gw_stride,
        gb_shape, gb_stride,
        training, eps, output_mask,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    grad_input = _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride, grad.dtype, grad.device)

    grad_weight = None
    if gw_ptr is not None:
        grad_weight = _wrap_output(gw_ptr, _numel(gw_shape), gw_shape, gw_stride, grad.dtype, grad.device)

    grad_bias = None
    if gb_ptr is not None:
        grad_bias = _wrap_output(gb_ptr, _numel(gb_shape), gb_shape, gb_stride, grad.dtype, grad.device)

    return grad_input, grad_weight, grad_bias


def npu_rms_norm_backward(grad, saved_input, backward_data, weight=None):
    """RMS norm backward via aclnnRmsNormGrad.

    *backward_data* must contain ``rstd_ptr``, ``rstd_shape``, ``rstd_stride``,
    ``normalized_shape``.
    """
    runtime, stream = _get_runtime_stream(grad)

    rstd_ptr = backward_data["rstd_ptr"]
    rstd_shape = backward_data["rstd_shape"]
    rstd_stride = backward_data["rstd_stride"]
    normalized_shape = backward_data["normalized_shape"]

    # gamma: if weight is None, create ones (aclnnRmsNormGrad requires gamma)
    if weight is not None:
        gamma_ptr = _unwrap_storage(weight).data_ptr()
        gamma_shape = weight.shape
        gamma_stride = weight.stride
    else:
        from ..._creation import ones as _ones
        w = _ones(normalized_shape, dtype=saved_input.dtype, device=saved_input.device)
        gamma_ptr = _unwrap_storage(w).data_ptr()
        gamma_shape = w.shape
        gamma_stride = w.stride

    # Allocate dx (same shape as input)
    dx_ptr, dx_shape, dx_stride, dx_numel = _alloc_like(saved_input, runtime)

    # Allocate dgamma (same shape as gamma)
    dgamma_numel = _numel(gamma_shape)
    dgamma_ptr = npu_runtime._alloc_device(
        max(dgamma_numel, 1) * _dtype_itemsize(grad.dtype), runtime=runtime
    )
    dgamma_shape = gamma_shape
    dgamma_stride = npu_runtime._contiguous_stride(dgamma_shape)

    aclnn.rms_norm_grad(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        rstd_ptr,
        gamma_ptr,
        dx_ptr,
        dgamma_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        rstd_shape, rstd_stride,
        gamma_shape, gamma_stride,
        dx_shape, dx_stride,
        dgamma_shape, dgamma_stride,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    grad_input = _wrap_output(dx_ptr, dx_numel, dx_shape, dx_stride, grad.dtype, grad.device)

    grad_weight = None
    if weight is not None:
        grad_weight = _wrap_output(
            dgamma_ptr, dgamma_numel, dgamma_shape, dgamma_stride,
            grad.dtype, grad.device,
        )
    else:
        # No weight was provided, dgamma is unused — free it via storage ref
        _dgamma_storage = npu_typed_storage_from_ptr(
            dgamma_ptr, max(dgamma_numel, 1), grad.dtype, device=grad.device
        )

    return grad_input, grad_weight


# ---------------------------------------------------------------
# Conv backward
# ---------------------------------------------------------------

def npu_conv_backward(grad, saved_input, saved_weight, saved_bias, name,
                      stride, padding, dilation, groups):
    """Convolution backward via aclnnConvolutionBackward.

    *name* is 'conv1d', 'conv2d', 'conv_transpose2d', etc.  It determines
    whether ``transposed`` is True and provides ``output_padding``.

    Returns ``(grad_input, grad_weight, grad_bias)``.  Any gradient that is
    not needed (e.g. bias is None) is returned as ``None``.
    """
    runtime, stream = _get_runtime_stream(grad)

    transposed = "transpose" in name
    # Determine spatial dimensionality from name or weight shape
    if "1d" in name:
        output_padding = (0, 0)  # handled via unsqueeze in forward
    else:
        output_padding = (0, 0)

    # grad_input (same shape as saved_input)
    gi_ptr = None
    gi_shape = saved_input.shape
    gi_stride = npu_runtime._contiguous_stride(gi_shape)
    gi_numel = _numel(gi_shape)
    if saved_input.requires_grad:
        gi_ptr = npu_runtime._alloc_device(
            max(gi_numel, 1) * _dtype_itemsize(grad.dtype), runtime=runtime
        )

    # grad_weight (same shape as saved_weight)
    gw_ptr = None
    gw_shape = saved_weight.shape
    gw_stride = npu_runtime._contiguous_stride(gw_shape)
    gw_numel = _numel(gw_shape)
    if saved_weight.requires_grad:
        gw_ptr = npu_runtime._alloc_device(
            max(gw_numel, 1) * _dtype_itemsize(grad.dtype), runtime=runtime
        )

    # grad_bias
    gb_ptr = None
    gb_shape = ()
    gb_stride = ()
    gb_numel = 0
    if saved_bias is not None and saved_bias.requires_grad:
        gb_shape = saved_bias.shape
        gb_stride = npu_runtime._contiguous_stride(gb_shape)
        gb_numel = _numel(gb_shape)
        gb_ptr = npu_runtime._alloc_device(
            max(gb_numel, 1) * _dtype_itemsize(grad.dtype), runtime=runtime
        )

    bias_sizes = list(saved_bias.shape) if saved_bias is not None else []
    output_mask = [gi_ptr is not None, gw_ptr is not None, gb_ptr is not None]

    aclnn.convolution_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        _unwrap_storage(saved_weight).data_ptr(),
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        saved_weight.shape, saved_weight.stride,
        grad.dtype,
        bias_sizes,
        stride, padding, dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
        gi_ptr,
        gw_ptr,
        gb_ptr,
        gi_shape, gi_stride,
        gw_shape, gw_stride,
        gb_shape, gb_stride,
        runtime,
        stream=stream.stream,
    )

    grad_input = None
    if gi_ptr is not None:
        grad_input = _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride, grad.dtype, grad.device)

    grad_weight = None
    if gw_ptr is not None:
        grad_weight = _wrap_output(gw_ptr, gw_numel, gw_shape, gw_stride, grad.dtype, grad.device)

    grad_bias = None
    if gb_ptr is not None:
        grad_bias = _wrap_output(gb_ptr, gb_numel, gb_shape, gb_stride, grad.dtype, grad.device)

    return grad_input, grad_weight, grad_bias


# ---------------------------------------------------------------
# Pool backward
# ---------------------------------------------------------------

def npu_max_pool2d_backward(grad, saved_input, backward_data):
    """MaxPool2d backward via aclnnMaxPool2dWithMaskBackward.

    *backward_data* is the ``_backward_data`` dict from the forward output.
    It must contain ``mask_ptr``, ``mask_shape``, ``mask_stride``,
    ``kernel_size``, ``strides``, ``padding``, ``dilation``, ``ceil_mode``.
    """
    runtime, stream = _get_runtime_stream(grad)

    mask_ptr = backward_data["mask_ptr"]
    mask_shape = backward_data["mask_shape"]
    mask_stride = backward_data["mask_stride"]
    kernel_size = backward_data["kernel_size"]
    strides = backward_data["strides"]
    padding = backward_data["padding"]
    dilation = backward_data["dilation"]
    ceil_mode = backward_data["ceil_mode"]

    # grad_input (same shape as saved_input)
    gi_ptr, gi_shape, gi_stride, gi_numel = _alloc_like(saved_input, runtime)

    aclnn.max_pool2d_with_mask_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        mask_ptr,
        gi_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        mask_shape, mask_stride,
        gi_shape, gi_stride,
        list(kernel_size), list(strides), list(padding), list(dilation),
        ceil_mode,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride, grad.dtype, grad.device)


def npu_avg_pool2d_backward(grad, saved_input, kernel_size, stride, padding,
                            ceil_mode=False, count_include_pad=True,
                            divisor_override=None):
    """AvgPool2d backward via aclnnAvgPool2dBackward."""
    runtime, stream = _get_runtime_stream(grad)

    # grad_input (same shape as saved_input)
    gi_ptr, gi_shape, gi_stride, gi_numel = _alloc_like(saved_input, runtime)

    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)

    aclnn.avg_pool2d_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        gi_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        gi_shape, gi_stride,
        [kH, kW], [sH, sW], [pH, pW],
        ceil_mode, count_include_pad,
        divisor_override,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride, grad.dtype, grad.device)
