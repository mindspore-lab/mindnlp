from ..._dtype import bool as bool_dtype
from ..._dtype import int32 as int32_dtype
from ..._dtype import int64 as int64_dtype
from ..._dtype import float32 as float_dtype
from ..._storage import npu_typed_storage_from_ptr
from . import aclnn
from . import runtime as npu_runtime
from . import state as npu_state
from ..common import convert as convert_backend


def _unwrap_storage(tensor):
    if tensor.storage().device.type != "npu":
        raise ValueError("Expected NPU storage for NPU op")
    return tensor.storage()


def _wrap_tensor(storage, shape, stride):
    from ..._tensor import Tensor

    return Tensor(storage, shape, stride)


def _dtype_itemsize(dtype):
    size = getattr(dtype, "itemsize", None)
    if size is not None:
        return int(size)
    name = getattr(dtype, "name", None) or str(dtype).split(".")[-1]
    return {"float16": 2, "float32": 4, "float64": 8, "bfloat16": 2, "int8": 1, "int16": 2,
            "int32": 4, "int64": 8, "uint8": 1, "bool": 1}.get(name, 4)


def _broadcast_shape(a_shape, b_shape):
    max_len = max(len(a_shape), len(b_shape))
    result = []
    for i in range(1, max_len + 1):
        a_dim = a_shape[-i] if i <= len(a_shape) else 1
        b_dim = b_shape[-i] if i <= len(b_shape) else 1
        if a_dim == 1:
            result.append(b_dim)
        elif b_dim == 1:
            result.append(a_dim)
        elif a_dim == b_dim:
            result.append(a_dim)
        else:
            raise ValueError("matmul shape mismatch")
    return tuple(reversed(result))


def _numel(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size


def _matmul_out_shape(a_shape, b_shape):
    a_dim = len(a_shape)
    b_dim = len(b_shape)

    if a_dim == 1 and b_dim == 1:
        if a_shape[0] != b_shape[0]:
            raise ValueError("matmul shape mismatch")
        return ()
    if a_dim == 1:
        k = a_shape[0]
        if b_dim < 2 or b_shape[-2] != k:
            raise ValueError("matmul shape mismatch")
        return b_shape[:-2] + (b_shape[-1],)
    if b_dim == 1:
        k = b_shape[0]
        if a_shape[-1] != k:
            raise ValueError("matmul shape mismatch")
        return a_shape[:-2] + (a_shape[-2],)
    if a_shape[-1] != b_shape[-2]:
        raise ValueError("matmul shape mismatch")
    batch = _broadcast_shape(a_shape[:-2], b_shape[:-2])
    return batch + (a_shape[-2], b_shape[-1])


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    total = 1
    for dim in shape:
        total *= dim
    for flat in range(total):
        idx = []
        rem = flat
        for dim in reversed(shape):
            idx.append(rem % dim)
            rem //= dim
        yield tuple(reversed(idx))


def _broadcast_index(index, shape, out_shape):
    if not shape:
        return ()
    offset = len(out_shape) - len(shape)
    sliced = index[offset:]
    result = []
    for i, dim in enumerate(shape):
        result.append(0 if dim == 1 else sliced[i])
    return tuple(result)


def _batch_offset(index, stride):
    return sum(i * s for i, s in zip(index, stride))


def _unary_op(a, fn, name, out_dtype=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError(f"NPU {name} expects NPU tensors")
    if out_dtype is None:
        out_dtype = a.dtype
    out_size = _numel(a.shape) * _dtype_itemsize(out_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    fn(storage.data_ptr(), out_ptr, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), out_dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def _binary_op(a, b, fn, name):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError(f"NPU {name} expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError(f"NPU {name} requires matching dtypes")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    fn(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        out_shape,
        out_stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def add(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU add expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU add requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)

    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.add(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, a.stride)


def mul(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU mul expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU mul requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.mul(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, a.stride)


def matmul(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU matmul expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU matmul requires matching dtypes")

    itemsize = _dtype_itemsize(a.dtype)
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    a_ptr = int(a_storage.data_ptr()) + int(a.offset * itemsize)
    b_ptr = int(b_storage.data_ptr()) + int(b.offset * itemsize)

    orig_a_shape = tuple(a.shape)
    orig_b_shape = tuple(b.shape)
    out_shape = _matmul_out_shape(orig_a_shape, orig_b_shape)

    a_shape = orig_a_shape
    b_shape = orig_b_shape
    a_stride = a.stride
    b_stride = b.stride

    a_dim = len(orig_a_shape)
    b_dim = len(orig_b_shape)
    if a_dim == 1:
        a_shape = (1, orig_a_shape[0])
        a_stride = (0, a_stride[0])
    if b_dim == 1:
        b_shape = (orig_b_shape[0], 1)
        b_stride = (b_stride[0], 0)

    if a_dim == 1 and b_dim == 1:
        out_shape_comp = (1, 1)
    elif a_dim == 1:
        out_shape_comp = orig_b_shape[:-2] + (1, orig_b_shape[-1])
    elif b_dim == 1:
        out_shape_comp = orig_a_shape[:-2] + (orig_a_shape[-2], 1)
    else:
        batch = _broadcast_shape(orig_a_shape[:-2], orig_b_shape[:-2])
        out_shape_comp = batch + (orig_a_shape[-2], orig_b_shape[-1])

    out_stride = npu_runtime._contiguous_stride(out_shape_comp)
    out_size = _numel(out_shape_comp) * itemsize
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)

    try:
        aclnn.matmul(
            a_ptr,
            b_ptr,
            out_ptr,
            a_shape,
            a_stride,
            b_shape,
            b_stride,
            out_shape_comp,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    except RuntimeError:
        a_batch = a_shape[:-2]
        b_batch = b_shape[:-2]
        batch_shape = _broadcast_shape(a_batch, b_batch)
        if not batch_shape:
            raise
        a_batch_stride = a_stride[:len(a_batch)]
        b_batch_stride = b_stride[:len(b_batch)]
        out_batch_stride = out_stride[:len(batch_shape)]
        for idx in _iter_indices(batch_shape):
            a_idx = _broadcast_index(idx, a_batch, batch_shape)
            b_idx = _broadcast_index(idx, b_batch, batch_shape)
            a_off = _batch_offset(a_idx, a_batch_stride)
            b_off = _batch_offset(b_idx, b_batch_stride)
            out_off = _batch_offset(idx, out_batch_stride)
            aclnn.matmul(
                a_ptr + int(a_off * itemsize),
                b_ptr + int(b_off * itemsize),
                out_ptr + int(out_off * itemsize),
                a_shape[-2:],
                a_stride[-2:],
                b_shape[-2:],
                b_stride[-2:],
                out_shape_comp[-2:],
                out_stride[-2:],
                a.dtype,
                runtime,
                stream=stream.stream,
            )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape_comp), a.dtype, device=a.device)
    out = _wrap_tensor(storage, out_shape_comp, out_stride)
    if out_shape_comp != out_shape:
        from ..common import view as view_backend

        out = view_backend.reshape(out, out_shape)
    return out


def relu(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU relu expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.relu(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, a.stride)


def abs(a):
    return _unary_op(a, aclnn.abs, "abs")


def neg(a):
    return _unary_op(a, aclnn.neg, "neg")


def sign(a):
    return _unary_op(a, aclnn.sign, "sign")


def signbit(a):
    return _unary_op(a, aclnn.signbit, "signbit", out_dtype=bool_dtype)


def isfinite(a):
    return _unary_op(a, aclnn.isfinite, "isfinite", out_dtype=bool_dtype)


def isinf(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU isinf expects NPU tensors")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(bool_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    if not a.dtype.is_floating_point:
        aclnn.logical_not(
            _unwrap_storage(isfinite(a)).data_ptr(),
            out_ptr,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
        return _wrap_tensor(out_storage, out_shape, out_stride)
    try:
        aclnn.isinf(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    except RuntimeError as exc:
        if "161001" not in str(exc):
            raise
        if not (aclnn.logical_not_symbols_ok() and aclnn.logical_and_symbols_ok()):
            raise RuntimeError("aclnnIsInf unavailable and logical ops missing")
        finite = isfinite(a)
        recip = pow(a, -1.0)
        recip_finite = isfinite(recip)
        tmp_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.logical_not(
            _unwrap_storage(finite).data_ptr(),
            tmp_ptr,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        tmp_bool_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.logical_and(
            tmp_ptr,
            _unwrap_storage(recip_finite).data_ptr(),
            tmp_bool_ptr,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        runtime.defer_free(tmp_ptr)
        out_ptr = tmp_bool_ptr
        runtime.defer_free(tmp_bool_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def isnan(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU isnan expects NPU tensors")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(bool_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    if not a.dtype.is_floating_point:
        aclnn.logical_not(
            _unwrap_storage(isfinite(a)).data_ptr(),
            out_ptr,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
        return _wrap_tensor(out_storage, out_shape, out_stride)
    if not (aclnn.logical_not_symbols_ok() and aclnn.logical_and_symbols_ok()):
        raise RuntimeError("aclnn logical ops missing for isnan")
    finite = isfinite(a)
    recip = pow(a, -1.0)
    recip_finite = isfinite(recip)
    tmp_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.logical_not(
        _unwrap_storage(finite).data_ptr(),
        tmp_ptr,
        out_shape,
        out_stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_not(
        _unwrap_storage(recip_finite).data_ptr(),
        out_ptr,
        out_shape,
        out_stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_and(
        tmp_ptr,
        out_ptr,
        out_ptr,
        out_shape,
        out_stride,
        out_shape,
        out_stride,
        out_shape,
        out_stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(tmp_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def _normalize_reduction_dims(dim, ndim):
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        return [dim]
    return list(dim)


def _reduce_out_shape(shape, dims, keepdim):
    out_shape = list(shape)
    for d in sorted(dims):
        out_shape[d] = 1
    if not keepdim:
        out_shape = [s for i, s in enumerate(out_shape) if i not in dims]
    return tuple(out_shape)


def _reduce_dim_sizes(shape, dims, keepdim):
    dims = sorted(dims)
    sizes = []
    for d in dims:
        sizes.append(shape[d])
    if keepdim:
        out_sizes = [1] * len(shape)
        for d, size in zip(dims, sizes):
            out_sizes[d] = size
        return tuple(out_sizes)
    return tuple(sizes)


def _broadcast_dims_to_out(dims, out_shape, keepdim):
    if keepdim:
        return dims
    offset = len(out_shape) - len(dims)
    return tuple(range(offset, offset + len(dims)))


def argmax(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU argmax expects NPU tensors")
    if not aclnn.max_dim_symbols_ok():
        raise RuntimeError("aclnnMaxDim not available")
    if dim is None:
        from ..common import view as view_backend

        flat = view_backend.reshape(a, (_numel(a.shape),))
        return argmax(flat, dim=0, keepdim=False)
    dims = _normalize_reduction_dims(dim, len(a.shape))
    if len(dims) != 1:
        raise ValueError("NPU argmax only supports single dimension")
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int64_dtype), runtime=runtime)
    val_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(a.dtype), runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.max_dim(
        storage.data_ptr(),
        val_ptr,
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims[0],
        keepdim,
        out_shape,
        out_stride,
        out_stride,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(val_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), int64_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)

def argmin(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU argmin expects NPU tensors")
    if not aclnn.min_dim_symbols_ok():
        raise RuntimeError("aclnnMinDim not available")
    if dim is None:
        from ..common import view as view_backend

        flat = view_backend.reshape(a, (_numel(a.shape),))
        return argmin(flat, dim=0, keepdim=False)
    dims = _normalize_reduction_dims(dim, len(a.shape))
    if len(dims) != 1:
        raise ValueError("NPU argmin only supports single dimension")
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int64_dtype), runtime=runtime)
    val_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(a.dtype), runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.min_dim(
        storage.data_ptr(),
        val_ptr,
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims[0],
        keepdim,
        out_shape,
        out_stride,
        out_stride,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(val_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), int64_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)

def amax(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU amax expects NPU tensors")
    if not aclnn.max_dim_symbols_ok():
        raise RuntimeError("aclnnMaxDim not available")
    if dim is None:
        from ..common import view as view_backend

        flat = view_backend.reshape(a, (_numel(a.shape),))
        return amax(flat, dim=0, keepdim=False)
    dims = _normalize_reduction_dims(dim, len(a.shape))
    if len(dims) != 1:
        raise ValueError("NPU amax only supports single dimension")
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    idx_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int64_dtype), runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.max_dim(
        storage.data_ptr(),
        out_ptr,
        idx_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims[0],
        keepdim,
        out_shape,
        out_stride,
        out_stride,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(idx_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def amin(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU amin expects NPU tensors")
    if not aclnn.min_dim_symbols_ok():
        raise RuntimeError("aclnnMinDim not available")
    if dim is None:
        from ..common import view as view_backend

        flat = view_backend.reshape(a, (_numel(a.shape),))
        return amin(flat, dim=0, keepdim=False)
    dims = _normalize_reduction_dims(dim, len(a.shape))
    if len(dims) != 1:
        raise ValueError("NPU amin only supports single dimension")
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    idx_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int64_dtype), runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.min_dim(
        storage.data_ptr(),
        out_ptr,
        idx_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims[0],
        keepdim,
        out_shape,
        out_stride,
        out_stride,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(idx_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def count_nonzero(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU count_nonzero expects NPU tensors")
    if not (aclnn.eq_scalar_symbols_ok() and aclnn.logical_not_symbols_ok() and aclnn.cast_symbols_ok()):
        raise RuntimeError("aclnn eq_scalar/logical_not/cast not available")
    dims = _normalize_reduction_dims(dim, len(a.shape))
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int64_dtype), runtime=runtime)
    mask_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.eq_scalar(
        _unwrap_storage(a).data_ptr(),
        0,
        mask_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_not(
        mask_ptr,
        mask_ptr,
        a.shape,
        a.stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    cast_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(int32_dtype), runtime=runtime)
    aclnn.cast(
        mask_ptr,
        cast_ptr,
        a.shape,
        a.stride,
        bool_dtype,
        int32_dtype,
        runtime,
        stream=stream.stream,
    )
    count_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int32_dtype), runtime=runtime)
    dims_payload = {
        "dims": dims if dim is not None else None,
        "out_shape": out_shape,
        "out_stride": out_stride,
    }
    aclnn.reduce_sum(
        cast_ptr,
        count_ptr,
        a.shape,
        a.stride,
        int32_dtype,
        dims_payload,
        keepdim,
        runtime,
        stream=stream.stream,
    )
    aclnn.cast(
        count_ptr,
        out_ptr,
        out_shape,
        out_stride,
        int32_dtype,
        int64_dtype,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(mask_ptr)
    runtime.defer_free(cast_ptr)
    runtime.defer_free(count_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), int64_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)

def all_(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU all expects NPU tensors")
    if not (aclnn.eq_scalar_symbols_ok() and aclnn.logical_not_symbols_ok() and aclnn.cast_symbols_ok()):
        raise RuntimeError("aclnn eq_scalar/logical_not/cast not available")
    dims = _normalize_reduction_dims(dim, len(a.shape))
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    mask_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.eq_scalar(
        _unwrap_storage(a).data_ptr(),
        0,
        mask_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_not(
        mask_ptr,
        mask_ptr,
        a.shape,
        a.stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    cast_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(int32_dtype), runtime=runtime)
    aclnn.cast(
        mask_ptr,
        cast_ptr,
        a.shape,
        a.stride,
        bool_dtype,
        int32_dtype,
        runtime,
        stream=stream.stream,
    )
    count_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int32_dtype), runtime=runtime)
    dims_payload = {
        "dims": dims if dim is not None else None,
        "out_shape": out_shape,
        "out_stride": out_stride,
    }
    aclnn.reduce_sum(
        cast_ptr,
        count_ptr,
        a.shape,
        a.stride,
        int32_dtype,
        dims_payload,
        keepdim,
        runtime,
        stream=stream.stream,
    )
    total = 1
    for d in dims:
        total *= a.shape[d]
    aclnn.eq_scalar(
        count_ptr,
        total,
        out_ptr,
        out_shape,
        out_stride,
        int32_dtype,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(mask_ptr)
    runtime.defer_free(cast_ptr)
    runtime.defer_free(count_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)

def any_(a, dim=None, keepdim=False):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU any expects NPU tensors")
    if not (aclnn.eq_scalar_symbols_ok() and aclnn.logical_not_symbols_ok() and aclnn.cast_symbols_ok()):
        raise RuntimeError("aclnn eq_scalar/logical_not/cast not available")
    dims = _normalize_reduction_dims(dim, len(a.shape))
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    mask_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.eq_scalar(
        _unwrap_storage(a).data_ptr(),
        0,
        mask_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_not(
        mask_ptr,
        mask_ptr,
        a.shape,
        a.stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    cast_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(int32_dtype), runtime=runtime)
    aclnn.cast(
        mask_ptr,
        cast_ptr,
        a.shape,
        a.stride,
        bool_dtype,
        int32_dtype,
        runtime,
        stream=stream.stream,
    )
    count_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(int32_dtype), runtime=runtime)
    dims_payload = {
        "dims": dims if dim is not None else None,
        "out_shape": out_shape,
        "out_stride": out_stride,
    }
    aclnn.reduce_sum(
        cast_ptr,
        count_ptr,
        a.shape,
        a.stride,
        int32_dtype,
        dims_payload,
        keepdim,
        runtime,
        stream=stream.stream,
    )
    aclnn.eq_scalar(
        count_ptr,
        0,
        out_ptr,
        out_shape,
        out_stride,
        int32_dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_not(
        out_ptr,
        out_ptr,
        out_shape,
        out_stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(mask_ptr)
    runtime.defer_free(cast_ptr)
    runtime.defer_free(count_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)

def exp(a):
    return _unary_op(a, aclnn.exp, "exp")


def log(a):
    return _unary_op(a, aclnn.log, "log")


def sqrt(a):
    return _unary_op(a, aclnn.sqrt, "sqrt")


def rsqrt(a):
    return _unary_op(a, aclnn.rsqrt, "rsqrt")


def sin(a):
    return _unary_op(a, aclnn.sin, "sin")


def cos(a):
    return _unary_op(a, aclnn.cos, "cos")


def tan(a):
    return _unary_op(a, aclnn.tan, "tan")


def tanh(a):
    return _unary_op(a, aclnn.tanh, "tanh")


def sigmoid(a):
    return _unary_op(a, aclnn.sigmoid, "sigmoid")


def sinh(a):
    return _unary_op(a, aclnn.sinh, "sinh")


def cosh(a):
    return _unary_op(a, aclnn.cosh, "cosh")


def erf(a):
    return _unary_op(a, aclnn.erf, "erf")


def erfc(a):
    return _unary_op(a, aclnn.erfc, "erfc")


def floor(a):
    return _unary_op(a, aclnn.floor, "floor")


def ceil(a):
    return _unary_op(a, aclnn.ceil, "ceil")


def round(a):
    return _unary_op(a, aclnn.round, "round")


def trunc(a):
    try:
        return _unary_op(a, aclnn.trunc, "trunc")
    except RuntimeError as exc:
        if "561103" not in str(exc):
            raise
    if not a.dtype.is_floating_point:
        return a
    if not aclnn.sign_symbols_ok():
        raise RuntimeError("aclnnTrunc not available and aclnnSign unavailable")
    return mul(sign(a), floor(abs(a)))


def frac(a):
    try:
        return _unary_op(a, aclnn.frac, "frac")
    except RuntimeError as exc:
        if "561103" not in str(exc):
            raise
    out = trunc(a)
    return add(a, neg(out))


def log2(a):
    return _unary_op(a, aclnn.log2, "log2")


def log10(a):
    return _unary_op(a, aclnn.log10, "log10")


def exp2(a):
    return _unary_op(a, aclnn.exp2, "exp2")


def softplus(a, beta=1.0, threshold=20.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU softplus expects NPU tensors")
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.softplus(
        storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        beta,
        threshold,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def clamp(a, min_val=None, max_val=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp expects NPU tensors")
    if min_val is None and max_val is None:
        raise ValueError("clamp requires min or max")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    if hasattr(min_val, "shape") and hasattr(max_val, "shape"):
        out_shape = _broadcast_shape(a.shape, min_val.shape)
        out_shape = _broadcast_shape(out_shape, max_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_tensor(
            storage.data_ptr(),
            _unwrap_storage(min_val).data_ptr(),
            _unwrap_storage(max_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            min_val.shape,
            min_val.stride,
            max_val.shape,
            max_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    elif hasattr(min_val, "shape"):
        out_shape = _broadcast_shape(a.shape, min_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_min_tensor(
            storage.data_ptr(),
            _unwrap_storage(min_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            min_val.shape,
            min_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
        if max_val is not None:
            temp_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
            temp_tensor = _wrap_tensor(temp_storage, out_shape, out_stride)
            return clamp_max(temp_tensor, max_val)
    elif hasattr(max_val, "shape"):
        out_shape = _broadcast_shape(a.shape, max_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_max_tensor(
            storage.data_ptr(),
            _unwrap_storage(max_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            max_val.shape,
            max_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
        if min_val is not None:
            temp_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
            temp_tensor = _wrap_tensor(temp_storage, out_shape, out_stride)
            return clamp_min(temp_tensor, min_val)
    else:
        aclnn.clamp_scalar(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            min_val,
            max_val,
            runtime,
            stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def clamp_min(a, min_val):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp_min expects NPU tensors")
    storage = _unwrap_storage(a)
    if hasattr(min_val, "shape"):
        out_shape = _broadcast_shape(a.shape, min_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_min_tensor(
            storage.data_ptr(),
            _unwrap_storage(min_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            min_val.shape,
            min_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    else:
        out_shape = a.shape
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_min_scalar(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            min_val,
            runtime,
            stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def clamp_max(a, max_val):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp_max expects NPU tensors")
    storage = _unwrap_storage(a)
    if hasattr(max_val, "shape"):
        out_shape = _broadcast_shape(a.shape, max_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_max_tensor(
            storage.data_ptr(),
            _unwrap_storage(max_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            max_val.shape,
            max_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    else:
        out_shape = a.shape
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_max_scalar(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            max_val,
            runtime,
            stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def relu6(a):
    return clamp(a, 0.0, 6.0)


def hardtanh(a, min_val=-1.0, max_val=1.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU hardtanh expects NPU tensors")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    try:
        aclnn.hardtanh(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            min_val,
            max_val,
            runtime,
            stream=stream.stream,
        )
    except RuntimeError as exc:
        if "561103" not in str(exc):
            raise
        # Fallback to clamp when hardtanh kernel is unsupported.
        return clamp(a, min_val, max_val)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)
def _pow_tensor_scalar_op(a, exponent):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU pow expects NPU tensors")
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.pow_tensor_scalar(
        storage.data_ptr(),
        exponent,
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def pow(a, b):
    if hasattr(b, "shape"):
        return _binary_op(a, b, aclnn.pow_tensor_tensor, "pow")
    return _pow_tensor_scalar_op(a, b)


def sum_(a, dim=None, keepdim=False, dtype=None):
    if dtype is not None:
        raise NotImplementedError("sum dtype not supported yet")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU sum expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_shape = list(a.shape)
    if dim is None:
        dims = list(range(len(out_shape)))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    for d in sorted(dims):
        out_shape[d] = 1
    if not keepdim:
        out_shape = [s for i, s in enumerate(out_shape) if i not in dims]
    out_shape = tuple(out_shape)

    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    dims_payload = {
        "dims": dims if dim is not None else None,
        "out_shape": out_shape,
        "out_stride": npu_runtime._contiguous_stride(out_shape),
    }
    aclnn.reduce_sum(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims_payload,
        keepdim,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, out_shape, npu_runtime._contiguous_stride(out_shape))


def add_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU add_ expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU add_ requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.add(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def _scalar_to_npu_tensor(scalar, ref_tensor):
    """Convert scalar to NPU tensor matching ref_tensor's shape/dtype/device."""
    import mindtorch_v2 as torch
    import numpy as np
    arr = np.full(ref_tensor.shape, scalar, dtype=np.float32)
    return torch.tensor(arr, dtype=ref_tensor.dtype).to(ref_tensor.device)


def mul_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU mul_ expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU mul_ requires matching dtypes")

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.mul(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def relu_(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU relu_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.relu(
        a_storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    runtime.activate()
    ret = npu_runtime.acl.rt.memcpy(
        a_storage.data_ptr(),
        out_size,
        out_ptr,
        out_size,
        3,
    )
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
    npu_runtime.get_runtime((a.device.index or 0)).defer_free(out_ptr)
    return a


def zero_(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU zero_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    aclnn.inplace_zero(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a

def contiguous(a):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU contiguous expects NPU tensors")

    a_storage = _unwrap_storage(a)
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    ret = npu_runtime.acl.rt.memcpy(
        out_ptr,
        out_size,
        a_storage.data_ptr(),
        out_size,
        3,
    )
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(storage, a.shape, npu_runtime._contiguous_stride(a.shape))


def getitem(tensor, key):
    """NPU tensor indexing via D2D memcpy of contiguous sub-regions."""
    itemsize = _dtype_itemsize(tensor.dtype)

    if isinstance(key, int):
        if key < 0:
            key += tensor.shape[0]
        if tensor.dim() == 1:
            # Return 0-dim tensor (matches PyTorch behavior)
            runtime = npu_runtime.get_runtime((tensor.device.index or 0))
            src_ptr = int(_unwrap_storage(tensor).data_ptr()) + key * itemsize
            out_ptr = npu_runtime._alloc_device(itemsize, runtime=runtime)
            ret = npu_runtime.acl.rt.memcpy(out_ptr, itemsize, src_ptr, itemsize, 3)  # D2D
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
            storage = npu_typed_storage_from_ptr(out_ptr, 1, tensor.dtype, device=tensor.device)
            return _wrap_tensor(storage, (), ())
        # Multi-dim: slice [key:key+1] then reshape to drop dim 0
        sliced = getitem(tensor, slice(key, key + 1))
        from ..common import view as view_backend
        return view_backend.reshape(sliced, tensor.shape[1:])

    if isinstance(key, slice):
        start, stop, step = key.indices(tensor.shape[0])
        if step != 1:
            raise NotImplementedError("NPU getitem with step != 1 not supported")
        length = max(0, stop - start)
        out_shape = (length,) + tensor.shape[1:]
        out_numel = _numel(out_shape)
        if out_numel == 0:
            out_shape_list = list(out_shape)
            out_stride = npu_runtime._contiguous_stride(out_shape) if out_numel else tuple([0] * len(out_shape))
            runtime = npu_runtime.get_runtime((tensor.device.index or 0))
            out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
            storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), tensor.dtype, device=tensor.device)
            return _wrap_tensor(storage, out_shape, out_stride)

        runtime = npu_runtime.get_runtime((tensor.device.index or 0))
        src_storage = _unwrap_storage(tensor)
        # Compute byte offset for start along dim 0
        stride0 = tensor.stride[0] if tensor.stride else 1
        byte_offset = int(start * stride0 * itemsize)
        src_ptr = int(src_storage.data_ptr()) + byte_offset
        out_size = int(out_numel * itemsize)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        ret = npu_runtime.acl.rt.memcpy(out_ptr, out_size, src_ptr, out_size, 3)  # D2D
        if ret != npu_runtime.ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
        out_stride = npu_runtime._contiguous_stride(out_shape)
        storage = npu_typed_storage_from_ptr(out_ptr, out_numel, tensor.dtype, device=tensor.device)
        return _wrap_tensor(storage, out_shape, out_stride)

    raise NotImplementedError(f"NPU getitem not implemented for key type: {type(key)}")


def setitem(tensor, key, value):
    """NPU tensor index assignment via D2D memcpy."""
    itemsize = _dtype_itemsize(tensor.dtype)

    if isinstance(key, int):
        if key < 0:
            key += tensor.shape[0]
        key = slice(key, key + 1)

    if isinstance(key, slice):
        start, stop, step = key.indices(tensor.shape[0])
        if step != 1:
            raise NotImplementedError("NPU setitem with step != 1 not supported")
        length = max(0, stop - start)
        if length == 0:
            return tensor

        stride0 = tensor.stride[0] if tensor.stride else 1
        byte_offset = int(start * stride0 * itemsize)
        dst_ptr = int(_unwrap_storage(tensor).data_ptr()) + byte_offset
        slice_numel = length
        for d in tensor.shape[1:]:
            slice_numel *= d
        copy_size = int(slice_numel * itemsize)

        runtime = npu_runtime.get_runtime((tensor.device.index or 0))

        if isinstance(value, (int, float)):
            # Scalar assignment: create a filled buffer on host, then H2D
            import numpy as np
            dtype_name = getattr(tensor.dtype, 'name', None) or str(tensor.dtype).split('.')[-1]
            np_dtype = {'float32': np.float32, 'float16': np.float16, 'float64': np.float64,
                        'int32': np.int32, 'int64': np.int64, 'int8': np.int8, 'int16': np.int16,
                        'uint8': np.uint8}.get(dtype_name, np.float32)
            arr = np.full(slice_numel, value, dtype=np_dtype)
            host_ptr = arr.ctypes.data
            ret = npu_runtime.acl.rt.memcpy(dst_ptr, copy_size, host_ptr, copy_size, 1)  # H2D
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy H2D failed: {ret}")
        elif hasattr(value, 'storage'):
            src_ptr = value.storage().data_ptr()
            if value.device.type != "npu":
                # CPU tensor -> H2D
                ret = npu_runtime.acl.rt.memcpy(dst_ptr, copy_size, src_ptr, copy_size, 1)
            else:
                # NPU tensor -> D2D
                ret = npu_runtime.acl.rt.memcpy(dst_ptr, copy_size, src_ptr, copy_size, 3)
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy failed: {ret}")
        return tensor

    raise NotImplementedError(f"NPU setitem not implemented for key type: {type(key)}")
