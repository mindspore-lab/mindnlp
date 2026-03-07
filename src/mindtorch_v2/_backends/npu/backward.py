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


def npu_group_norm_backward(grad, saved_input, num_groups, weight=None, eps=1e-5):
    """Group norm backward via aclnnGroupNormBackward.

    Computes mean and rstd from *saved_input* since the composite forward
    does not save them.  These are small [N, num_groups] tensors so the
    overhead of recomputing them is minimal.

    Returns ``(grad_input,)``.
    """
    from ..._dispatch.dispatcher import redispatch, current_dispatch_keyset
    from ..._dispatch.keys import DispatchKey
    from ..._creation import tensor as _create_tensor

    runtime, stream = _get_runtime_stream(grad)

    N = saved_input.shape[0]
    C = saved_input.shape[1]
    spatial = saved_input.shape[2:]
    HxW = 1
    for s in spatial:
        HxW *= s
    channels_per_group = C // num_groups

    # Compute mean and rstd per group from saved_input.
    # Reshape to [N, num_groups, channels_per_group * HxW] then reduce last dim.
    keyset = current_dispatch_keyset().without(
        (DispatchKey.Autograd, DispatchKey.AutogradCPU, DispatchKey.AutogradNPU)
    )

    reshaped = redispatch("reshape", keyset, saved_input,
                          (N, num_groups, channels_per_group * HxW))
    mean = redispatch("mean", keyset, reshaped, dim=2, keepdim=False)  # [N, num_groups]

    # var = E[(x - mean)^2]
    diff = redispatch("add", keyset, reshaped,
                      redispatch("neg", keyset, redispatch("unsqueeze", keyset, mean, dim=-1)))
    var = redispatch("mean", keyset, redispatch("mul", keyset, diff, diff),
                     dim=2, keepdim=False)  # [N, num_groups]

    eps_t = _create_tensor(eps, dtype=saved_input.dtype, device=saved_input.device)
    rstd = redispatch("rsqrt", keyset,
                      redispatch("add", keyset, var, eps_t))  # [N, num_groups]

    # Cast mean/rstd to float32 if needed (ACLNN expects float32 stats)
    stats_dtype = mean.dtype
    if str(stats_dtype) != "float32":
        from ..._dtype import float32 as f32_dtype
        mean_f32_ptr = npu_runtime._alloc_device(
            max(_numel(mean.shape), 1) * 4, runtime=runtime)
        aclnn.cast(
            _unwrap_storage(mean).data_ptr(), mean_f32_ptr,
            mean.shape, mean.stride, stats_dtype, f32_dtype,
            runtime, stream=stream.stream,
        )
        mean_storage = npu_typed_storage_from_ptr(
            mean_f32_ptr, max(_numel(mean.shape), 1), f32_dtype, device=mean.device)
        from ..._tensor import Tensor as _Tensor
        mean = _Tensor(mean_storage, mean.shape, npu_runtime._contiguous_stride(mean.shape))

        rstd_f32_ptr = npu_runtime._alloc_device(
            max(_numel(rstd.shape), 1) * 4, runtime=runtime)
        aclnn.cast(
            _unwrap_storage(rstd).data_ptr(), rstd_f32_ptr,
            rstd.shape, rstd.stride, stats_dtype, f32_dtype,
            runtime, stream=stream.stream,
        )
        rstd_storage = npu_typed_storage_from_ptr(
            rstd_f32_ptr, max(_numel(rstd.shape), 1), f32_dtype, device=rstd.device)
        rstd = _Tensor(rstd_storage, rstd.shape, npu_runtime._contiguous_stride(rstd.shape))

    # Allocate output grad_input (same shape as input)
    gi_ptr, gi_shape, gi_stride, gi_numel = _alloc_like(saved_input, runtime)

    # Allocate grad_gamma, grad_beta (shape = [C]) only if weight exists
    has_weight = weight is not None
    dgamma_ptr = None
    dgamma_shape = (C,)
    dgamma_stride = (1,)
    dgamma_numel = C
    dbeta_ptr = None
    dbeta_shape = (C,)
    dbeta_stride = (1,)
    dbeta_numel = C
    if has_weight:
        dgamma_ptr, dgamma_shape, dgamma_stride, dgamma_numel = _alloc_like(
            weight, runtime, shape=weight.shape)
        dbeta_ptr, dbeta_shape, dbeta_stride, dbeta_numel = _alloc_like(
            weight, runtime, shape=weight.shape)

    output_mask = (True, has_weight, has_weight)

    aclnn.group_norm_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        _unwrap_storage(mean).data_ptr(),
        _unwrap_storage(rstd).data_ptr(),
        _unwrap_storage(weight).data_ptr() if has_weight else 0,
        gi_ptr,
        dgamma_ptr,
        dbeta_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        mean.shape, mean.stride,
        rstd.shape, rstd.stride,
        weight.shape if has_weight else (C,),
        weight.stride if has_weight else (1,),
        gi_shape, gi_stride,
        dgamma_shape, dgamma_stride,
        dbeta_shape, dbeta_stride,
        N, C, HxW, num_groups,
        output_mask,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    grad_input = _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride,
                              grad.dtype, grad.device)

    return (grad_input,)


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


# ---------------------------------------------------------------
# Activation backward — hardswish, hardsigmoid, mish (simple)
# ---------------------------------------------------------------

def npu_hardswish_backward(grad, saved_input):
    """Hardswish backward via aclnnHardswishBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.hardswish_backward(
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


def npu_hardsigmoid_backward(grad, saved_input):
    """Hardsigmoid backward via aclnnHardsigmoidBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.hardsigmoid_backward(
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


def npu_mish_backward(grad, saved_input):
    """Mish backward via aclnnMishBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.mish_backward(
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
# Activation backward — softplus, hardtanh (with scalar params)
# ---------------------------------------------------------------

def npu_softplus_backward(grad, saved_input, beta=1.0, threshold=20.0):
    """Softplus backward via aclnnSoftplusBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.softplus_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        out_stride,
        grad.dtype,
        beta,
        threshold,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_hardtanh_backward(grad, saved_input, min_val=-1.0, max_val=1.0):
    """Hardtanh backward via aclnnHardtanhBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.hardtanh_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        out_stride,
        grad.dtype,
        min_val,
        max_val,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


# ---------------------------------------------------------------
# Parameterized activation backward (leaky_relu, elu, prelu)
# ---------------------------------------------------------------

def npu_leaky_relu_backward(grad, saved_input, negative_slope=0.01):
    """Leaky ReLU backward via aclnnLeakyReluBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.leaky_relu_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        out_stride,
        grad.dtype,
        negative_slope,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_elu_backward(grad, saved_input, alpha=1.0, scale=1.0, input_scale=1.0):
    """ELU backward via aclnnEluBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(grad, runtime)

    aclnn.elu_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        out_stride,
        grad.dtype,
        alpha,
        scale,
        input_scale,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_prelu_backward(grad, saved_input, saved_weight):
    """PReLU backward via aclnnPreluBackward -- returns (grad_input, grad_weight)."""
    runtime, stream = _get_runtime_stream(grad)
    gi_ptr, gi_shape, gi_stride, gi_numel = _alloc_like(grad, runtime)
    gw_ptr, gw_shape, gw_stride, gw_numel = _alloc_like(saved_weight, runtime, shape=saved_weight.shape)

    aclnn.prelu_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        _unwrap_storage(saved_weight).data_ptr(),
        gi_ptr,
        gw_ptr,
        grad.shape,
        grad.stride,
        saved_input.stride,
        saved_weight.shape,
        saved_weight.stride,
        gi_stride,
        gw_stride,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    grad_input = _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride, grad.dtype, grad.device)
    grad_weight = _wrap_output(gw_ptr, gw_numel, gw_shape, gw_stride, grad.dtype, grad.device)
    return grad_input, grad_weight


# ---------------------------------------------------------------
# Upsample backward (5 ops)
# ---------------------------------------------------------------

def npu_upsample_nearest2d_backward(grad, saved_input, output_size):
    """Upsample nearest 2d backward via aclnnUpsampleNearest2dBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    H_out, W_out = output_size
    N, C, H_in, W_in = saved_input.shape
    scales_h = float(H_out) / float(H_in)
    scales_w = float(W_out) / float(W_in)
    aclnn.upsample_nearest2d_backward(
        _unwrap_storage(grad).data_ptr(), out_ptr,
        grad.shape, grad.stride, saved_input.shape, out_stride,
        (H_out, W_out), (N, C, H_in, W_in),
        scales_h, scales_w,
        grad.dtype, runtime, stream=stream.stream,
    )
    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_upsample_bilinear2d_backward(grad, saved_input, output_size, align_corners=False):
    """Upsample bilinear 2d backward via aclnnUpsampleBilinear2dBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    H_out, W_out = output_size
    N, C, H_in, W_in = saved_input.shape
    scales_h = float(H_out) / float(H_in)
    scales_w = float(W_out) / float(W_in)
    aclnn.upsample_bilinear2d_backward(
        _unwrap_storage(grad).data_ptr(), out_ptr,
        grad.shape, grad.stride, saved_input.shape, out_stride,
        (H_out, W_out), (N, C, H_in, W_in),
        align_corners, scales_h, scales_w,
        grad.dtype, runtime, stream=stream.stream,
    )
    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_upsample_bicubic2d_backward(grad, saved_input, output_size, align_corners=False):
    """Upsample bicubic 2d backward via aclnnUpsampleBicubic2dBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    H_out, W_out = output_size
    N, C, H_in, W_in = saved_input.shape
    scales_h = float(H_out) / float(H_in)
    scales_w = float(W_out) / float(W_in)
    aclnn.upsample_bicubic2d_backward(
        _unwrap_storage(grad).data_ptr(), out_ptr,
        grad.shape, grad.stride, saved_input.shape, out_stride,
        (H_out, W_out), (N, C, H_in, W_in),
        align_corners, scales_h, scales_w,
        grad.dtype, runtime, stream=stream.stream,
    )
    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_upsample_nearest1d_backward(grad, saved_input, output_size):
    """Upsample nearest 1d backward via aclnnUpsampleNearest1dBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    (L_out,) = output_size
    N, C, L_in = saved_input.shape
    scales = float(L_out) / float(L_in)
    aclnn.upsample_nearest1d_backward(
        _unwrap_storage(grad).data_ptr(), out_ptr,
        grad.shape, grad.stride, saved_input.shape, out_stride,
        (L_out,), (N, C, L_in), scales,
        grad.dtype, runtime, stream=stream.stream,
    )
    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_upsample_linear1d_backward(grad, saved_input, output_size, align_corners=False):
    """Upsample linear 1d backward via aclnnUpsampleLinear1dBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    (L_out,) = output_size
    N, C, L_in = saved_input.shape
    scales = float(L_out) / float(L_in)
    aclnn.upsample_linear1d_backward(
        _unwrap_storage(grad).data_ptr(), out_ptr,
        grad.shape, grad.stride, saved_input.shape, out_stride,
        (L_out,), (N, C, L_in),
        align_corners, scales,
        grad.dtype, runtime, stream=stream.stream,
    )
    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


# ---------------------------------------------------------------
# Pool backward
# ---------------------------------------------------------------

def npu_adaptive_avg_pool2d_backward(grad, saved_input):
    """Adaptive avg pool 2d backward via aclnnAdaptiveAvgPool2dBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    aclnn.adaptive_avg_pool2d_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        saved_input.shape, out_stride,
        grad.dtype, runtime, stream=stream.stream,
    )
    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_adaptive_avg_pool3d_backward(grad, saved_input):
    """Adaptive avg pool 3d backward via aclnnAdaptiveAvgPool3dBackward."""
    runtime, stream = _get_runtime_stream(grad)
    out_ptr, out_shape, out_stride, out_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    aclnn.adaptive_avg_pool3d_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        out_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        saved_input.shape, out_stride,
        grad.dtype, runtime, stream=stream.stream,
    )
    return _wrap_output(out_ptr, out_numel, out_shape, out_stride, grad.dtype, grad.device)


def npu_avg_pool3d_backward(grad, saved_input, kernel_size, stride, padding,
                             ceil_mode=False, count_include_pad=True,
                             divisor_override=None):
    """Avg pool 3d backward via aclnnAvgPool3dBackward."""
    runtime, stream = _get_runtime_stream(grad)

    # grad_input (same shape as saved_input)
    gi_ptr, gi_shape, gi_stride, gi_numel = _alloc_like(saved_input, runtime)

    kD, kH, kW = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sD, sH, sW = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
    pD, pH, pW = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)

    aclnn.avg_pool3d_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        gi_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        gi_shape, gi_stride,
        [kD, kH, kW], [sD, sH, sW], [pD, pH, pW],
        ceil_mode, count_include_pad,
        divisor_override,
        grad.dtype,
        runtime,
        stream=stream.stream,
    )

    return _wrap_output(gi_ptr, gi_numel, gi_shape, gi_stride, grad.dtype, grad.device)


# ---------------------------------------------------------------
# Grid sample backward
# ---------------------------------------------------------------

def npu_grid_sample_backward(grad, saved_input, saved_grid,
                              interpolation_mode=0, padding_mode=0, align_corners=False):
    """Grid sample backward via aclnnGridSampler2DBackward."""
    runtime, stream = _get_runtime_stream(grad)

    # Allocate inputGrad (same shape as input)
    ig_ptr, ig_shape, ig_stride, ig_numel = _alloc_like(saved_input, runtime, shape=saved_input.shape)
    # Allocate gridGrad (same shape as grid)
    gg_ptr, gg_shape, gg_stride, gg_numel = _alloc_like(saved_grid, runtime, shape=saved_grid.shape)

    compute_input_grad = getattr(saved_input, "requires_grad", True)
    compute_grid_grad = getattr(saved_grid, "requires_grad", True)

    aclnn.grid_sampler2d_backward(
        _unwrap_storage(grad).data_ptr(),
        _unwrap_storage(saved_input).data_ptr(),
        _unwrap_storage(saved_grid).data_ptr(),
        ig_ptr, gg_ptr,
        grad.shape, grad.stride,
        saved_input.shape, saved_input.stride,
        saved_grid.shape, saved_grid.stride,
        ig_shape, ig_stride,
        gg_shape, gg_stride,
        interpolation_mode, padding_mode, align_corners,
        compute_input_grad, compute_grid_grad,
        grad.dtype, runtime, stream=stream.stream,
    )

    grad_input = _wrap_output(ig_ptr, ig_numel, ig_shape, ig_stride, grad.dtype, grad.device)
    grad_grid = _wrap_output(gg_ptr, gg_numel, gg_shape, gg_stride, grad.dtype, grad.device)
    return grad_input, grad_grid
