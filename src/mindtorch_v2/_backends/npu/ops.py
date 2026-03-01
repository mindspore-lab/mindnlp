import ctypes
from ..._dtype import bool as bool_dtype
from ..._dtype import int32 as int32_dtype
from ..._dtype import int64 as int64_dtype
from ..._dtype import float32 as float_dtype
from ..._storage import npu_typed_storage_from_ptr
from ..common import view as view_backend
reshape = view_backend.reshape
from . import aclnn
from . import runtime as npu_runtime
from . import state as npu_state


def _unwrap_storage(tensor):
    if tensor.storage().device.type != "npu":
        raise ValueError("Expected NPU storage for NPU op")
    return tensor.storage()


def _wrap_tensor(storage, shape, stride):
    from ..._tensor import Tensor

    return Tensor(storage, shape, stride)


def _broadcast_shape_checked(a_shape, b_shape, name):
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
            raise ValueError(f"NPU {name} shape mismatch")
    return tuple(reversed(result))


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


def _npu_broadcast_to(tensor, shape):
    from .creation import zeros_create

    shape = tuple(shape)
    if tensor.shape == shape:
        return tensor
    zeros = zeros_create(shape, dtype=tensor.dtype, device=tensor.device)
    return _binary_op(tensor, zeros, aclnn.add, "add")


def _npu_arange_1d(size, device):
    runtime = npu_runtime.get_runtime((device.index or 0))
    stream = npu_state.current_stream((device.index or 0))
    if not aclnn.arange_symbols_ok():
        raise RuntimeError("aclnnArange symbols not available")
    shape = (int(size),)
    stride = npu_runtime._contiguous_stride(shape)
    out_size = _numel(shape) * _dtype_itemsize(int64_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.arange(0, int(size), 1, out_ptr, shape, stride, int64_dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(out_ptr, _numel(shape), int64_dtype, device=device)
    return _wrap_tensor(storage, shape, stride)


def _npu_add_scalar_(tensor, scalar):
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    if not aclnn.add_scalar_symbols_ok():
        raise RuntimeError("aclnnAdds symbols not available")
    storage = _unwrap_storage(tensor)
    aclnn.add_scalar(
        storage.data_ptr(),
        scalar,
        storage.data_ptr(),
        tensor.shape,
        tensor.stride,
        tensor.dtype,
        runtime,
        stream=stream.stream,
    )
    return tensor


def _npu_linear_index(view_shape, view_stride, view_offset, device):
    ndim = len(view_shape)
    if ndim == 0:
        runtime = npu_runtime.get_runtime((device.index or 0))
        stream = npu_state.current_stream((device.index or 0))
        out_ptr = npu_runtime._alloc_device(_dtype_itemsize(int64_dtype), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, 1, int64_dtype, device=device)
        out = _wrap_tensor(storage, (), ())
        return _npu_add_scalar_(out, view_offset)
    linear = None
    for dim, size in enumerate(view_shape):
        idx = _npu_arange_1d(size, device)
        shape = [1] * ndim
        shape[dim] = int(size)
        idx = idx.reshape(shape)
        target_shape = _broadcast_shape_checked(tuple(shape), tuple(view_shape), "view-index")
        if target_shape != tuple(view_shape):
            raise RuntimeError("NPU view index broadcast mismatch")
        if view_stride[dim] != 1:
            idx = idx * _scalar_to_npu_tensor(view_stride[dim], idx)
        if linear is None:
            linear = idx
        else:
            linear = _binary_op(linear, idx, aclnn.add, "add")
    return _npu_add_scalar_(linear, view_offset)


def npu_index_put_impl(self_tensor, index_tensor, values, accumulate=False, unsafe=False):
    runtime = npu_runtime.get_runtime((self_tensor.device.index or 0))
    stream = npu_state.current_stream((self_tensor.device.index or 0))
    if not aclnn.index_put_impl_symbols_ok():
        raise RuntimeError("aclnnIndexPutImpl symbols not available")
    self_storage = _unwrap_storage(self_tensor)
    index_storage = _unwrap_storage(index_tensor)
    values_storage = _unwrap_storage(values)
    aclnn.index_put_impl(
        self_storage.data_ptr(),
        self_tensor.shape,
        self_tensor.stride,
        self_tensor.dtype,
        [index_storage.data_ptr()],
        [index_tensor.shape],
        [index_tensor.stride],
        [index_tensor.dtype],
        values_storage.data_ptr(),
        values.shape,
        values.stride,
        values.dtype,
        bool(accumulate),
        bool(unsafe),
        runtime,
        stream=stream.stream,
    )


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


def _normalize_tensor_sequence_args(tensors):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        return tuple(tensors[0])
    return tuple(tensors)


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
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.add, "add")


def mul(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.mul, "mul")




def sub(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.sub, "sub")


def div(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.div, "div")


def eq(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if not aclnn.eq_tensor_symbols_ok():
        raise RuntimeError("aclnnEqTensor symbols not available")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.eq_tensor(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
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
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def ne(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if not aclnn.ne_tensor_symbols_ok():
        raise RuntimeError("aclnnNeTensor symbols not available")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.ne_tensor(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
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
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def logical_and(a, b):
    return _binary_op(a, b, aclnn.logical_and, "logical_and")


def logical_or(a, b):
    return _binary_op(a, b, aclnn.logical_or, "logical_or")


def logical_not(a):
    return _unary_op(a, aclnn.logical_not, "logical_not", out_dtype=bool_dtype)


def le(a, b):
    return logical_or(signbit(sub(a, b)), eq(a, b))


def lt(a, b):
    return signbit(sub(a, b))


def gt(a, b):
    return signbit(sub(b, a))


def ge(a, b):
    return logical_or(gt(a, b), eq(a, b))


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

def asinh(a):
    return _unary_op(a, aclnn.asinh, "asinh")


def acosh(a):
    out = _unary_op(a, aclnn.acosh, "acosh")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = lt(a, one)
    return where(mask, _nan_like(a), out)


def atanh(a):
    out = _unary_op(a, aclnn.atanh, "atanh")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = ge(abs(a), one)
    return where(mask, _nan_like(a), out)


def atan(a):
    return _unary_op(a, aclnn.atan, "atan")


def asin(a):
    out = _unary_op(a, aclnn.asin, "asin")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = gt(abs(a), one)
    return where(mask, _nan_like(a), out)


def acos(a):
    out = _unary_op(a, aclnn.acos, "acos")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = gt(abs(a), one)
    return where(mask, _nan_like(a), out)


def atan2(a, b):
    return _binary_op(a, b, aclnn.atan2, "atan2")


def min_(a, b):
    result = _binary_op(a, b, aclnn.minimum, "min")
    nan_mask = logical_or(isnan(a), isnan(b))
    return where(nan_mask, add(a, b), result)


def max_(a, b):
    result = _binary_op(a, b, aclnn.maximum, "max")
    nan_mask = logical_or(isnan(a), isnan(b))
    return where(nan_mask, add(a, b), result)


def fmin(a, b):
    nan_a = isnan(a)
    nan_b = isnan(b)
    return where(nan_a, b, where(nan_b, a, min_(a, b)))


def fmax(a, b):
    nan_a = isnan(a)
    nan_b = isnan(b)
    return where(nan_a, b, where(nan_b, a, max_(a, b)))


def where(cond, x, y):
    if x.device.type != "npu":
        raise ValueError("NPU where expects NPU tensors")
    if isinstance(cond, (int, float)):
        cond = _scalar_to_npu_tensor(cond, x)
    if isinstance(y, (int, float)):
        y = _scalar_to_npu_tensor(y, x)
    if cond.device.type != "npu" or y.device.type != "npu":
        raise ValueError("NPU where expects NPU tensors")
    if x.dtype != y.dtype:
        raise ValueError("NPU where requires matching dtypes")
    if cond.dtype != bool_dtype:
        cond = ne(cond, _scalar_to_npu_tensor(0, cond))
    if not aclnn.s_where_symbols_ok():
        raise RuntimeError("aclnnSWhere symbols not available")

    out_shape = _broadcast_shape(cond.shape, x.shape)
    out_shape = _broadcast_shape(out_shape, y.shape)
    if out_shape != x.shape:
        x = _npu_broadcast_to(x, out_shape)
    if out_shape != y.shape:
        y = _npu_broadcast_to(y, out_shape)
    if out_shape != cond.shape:
        cond = _npu_broadcast_to(cond, out_shape)

    runtime = npu_runtime.get_runtime((x.device.index or 0))
    stream = npu_state.current_stream((x.device.index or 0))
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(x.dtype), runtime=runtime)
    aclnn.s_where(
        _unwrap_storage(cond).data_ptr(),
        _unwrap_storage(x).data_ptr(),
        _unwrap_storage(y).data_ptr(),
        out_ptr,
        cond.shape,
        cond.stride,
        cond.dtype,
        x.shape,
        x.stride,
        x.dtype,
        y.shape,
        y.stride,
        y.dtype,
        out_shape,
        out_stride,
        x.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), x.dtype, device=x.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def _normalize_dims_tuple(dims, ndim, name):
    if isinstance(dims, int):
        raise TypeError(f"{name} dims must be tuple/list of ints")
    if not isinstance(dims, (tuple, list)):
        raise TypeError(f"{name} dims must be tuple/list of ints")
    norm = []
    seen = set()
    for d in dims:
        if not isinstance(d, int):
            raise TypeError(f"{name} dims must contain ints")
        d = _normalize_dim(d, ndim)
        if d in seen:
            raise RuntimeError(f"dim {d} appears multiple times in the list of dims")
        seen.add(d)
        norm.append(d)
    return tuple(norm)


def _normalize_roll_args(shifts, dims, ndim):
    if isinstance(shifts, int):
        shifts_tuple = (int(shifts),)
    elif isinstance(shifts, (tuple, list)):
        shifts_tuple = tuple(int(s) for s in shifts)
    else:
        raise TypeError("roll shifts must be int/tuple/list")
    if len(shifts_tuple) == 0:
        raise RuntimeError("`shifts` required")

    if isinstance(dims, int):
        dims_tuple = (int(dims),)
    elif isinstance(dims, (tuple, list)):
        dims_tuple = tuple(int(d) for d in dims)
    else:
        raise TypeError("roll dims must be int/tuple/list/None")

    if len(shifts_tuple) != len(dims_tuple):
        raise RuntimeError(f"shifts and dimensions must align. shifts: {len(shifts_tuple)}, dims:{len(dims_tuple)}")
    return shifts_tuple, tuple(_normalize_dim(d, ndim) for d in dims_tuple)


def _cumulative_out_dtype(dtype):
    # torch promotes bool/int cumulative ops to int64 by default.
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return int64_dtype


def _normalize_repeats_tuple(repeats, ndim, name):
    if isinstance(repeats, int):
        repeats = (int(repeats),)
    elif isinstance(repeats, (tuple, list)):
        repeats = tuple(int(r) for r in repeats)
    else:
        raise TypeError(f"{name} repeats must be int/tuple/list")
    if len(repeats) < ndim:
        repeats = (1,) * (ndim - len(repeats)) + repeats
    if len(repeats) < ndim:
        raise RuntimeError("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    return repeats


def _build_repeat_interleave_indices(dim_size, repeats, device):
    from .creation import zeros_create

    if isinstance(repeats, int):
        if repeats < 0:
            raise ValueError("repeats must be non-negative")
        output_size = int(dim_size) * int(repeats)
        if output_size == 0:
            return zeros_create((0,), dtype=int64_dtype, device=device), output_size
        base = _npu_arange_1d(int(dim_size), device)
        idx = repeat(base, (int(repeats),))
        return idx, output_size

    if not hasattr(repeats, "shape"):
        raise TypeError("repeats must be int or Tensor")
    if repeats.device.type != "npu":
        raise ValueError("repeats tensor must be on NPU")
    if repeats.dtype != int64_dtype:
        raise TypeError("repeats tensor must be int64")
    if repeats.dim() != 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
    if repeats.shape[0] not in (1, int(dim_size)):
        raise RuntimeError(
            f"repeats must have the same size as input along dim, but got repeats.size(0) = {repeats.shape[0]} and input.size(0) = {dim_size}"
        )

    rep_list = repeats.to("cpu").tolist()
    if any(int(v) < 0 for v in rep_list):
        raise ValueError("repeats must be non-negative")

    if repeats.shape[0] == 1:
        rep = int(rep_list[0])
        return _build_repeat_interleave_indices(dim_size, rep, device)

    output_size = int(sum(int(v) for v in rep_list))
    if output_size == 0:
        return zeros_create((0,), dtype=int64_dtype, device=device), output_size

    idx_chunks = []
    for i, rep in enumerate(rep_list):
        rep = int(rep)
        if rep == 0:
            continue
        scalar = _npu_arange_1d(int(rep), device)
        scalar = add(scalar, _scalar_to_npu_tensor(int(i), scalar))
        idx_chunks.append(scalar)

    if not idx_chunks:
        return zeros_create((0,), dtype=int64_dtype, device=device), 0
    if len(idx_chunks) == 1:
        return idx_chunks[0], output_size
    return cat(idx_chunks, dim=0), output_size


def flip(a, dims):
    if a.device.type != "npu":
        raise ValueError("NPU flip expects NPU tensors")
    dims = _normalize_dims_tuple(dims, a.dim(), "flip")
    if len(dims) == 0:
        return a
    if not aclnn.flip_symbols_ok():
        raise RuntimeError("aclnnFlip symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.flip(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dims,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def roll(a, shifts, dims=None):
    if a.device.type != "npu":
        raise ValueError("NPU roll expects NPU tensors")
    if dims is None:
        flat = view_backend.reshape(a, (a.numel(),))
        rolled = roll(flat, shifts, dims=0)
        return view_backend.reshape(rolled, a.shape)
    shifts_tuple, dims_tuple = _normalize_roll_args(shifts, dims, a.dim())
    if not aclnn.roll_symbols_ok():
        raise RuntimeError("aclnnRoll symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.roll(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        shifts_tuple,
        dims_tuple,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def cumsum(a, dim=0):
    if a.device.type != "npu":
        raise ValueError("NPU cumsum expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())
    if not aclnn.cumsum_symbols_ok():
        raise RuntimeError("aclnnCumsum symbols not available")
    out_dtype = _cumulative_out_dtype(a.dtype)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(out_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.cumsum(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dim,
        out_dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), out_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def cumprod(a, dim=0):
    if a.device.type != "npu":
        raise ValueError("NPU cumprod expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())
    if not aclnn.cumprod_symbols_ok():
        raise RuntimeError("aclnnCumprod symbols not available")
    out_dtype = _cumulative_out_dtype(a.dtype)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(out_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.cumprod(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dim,
        out_dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), out_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def cummax(a, dim=0):
    if a.device.type != "npu":
        raise ValueError("NPU cummax expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())
    if not aclnn.cummax_symbols_ok():
        raise RuntimeError("aclnnCummax symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    numel = max(_numel(out_shape), 1)
    values_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
    indices_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(int64_dtype), runtime=runtime)
    aclnn.cummax(
        _unwrap_storage(a).data_ptr(),
        values_ptr,
        indices_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dim,
        runtime,
        stream=stream.stream,
    )
    values_storage = npu_typed_storage_from_ptr(values_ptr, numel, a.dtype, device=a.device)
    indices_storage = npu_typed_storage_from_ptr(indices_ptr, numel, int64_dtype, device=a.device)
    return _wrap_tensor(values_storage, out_shape, out_stride), _wrap_tensor(indices_storage, out_shape, out_stride)


def argsort(a, dim=-1, descending=False, stable=False):
    if a.device.type != "npu":
        raise ValueError("NPU argsort expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())

    # aclnnArgsort/aclnnSort can poison subsequent topk in current runtime.
    # Use topk(k=full_dim) for stable=False to keep behavior and runtime stability.
    if not stable:
        _, indices = topk(a, k=a.shape[dim], dim=dim, largest=bool(descending), sorted=True)
        return indices

    if not aclnn.argsort_symbols_ok():
        raise RuntimeError("aclnnArgsort symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    numel = max(_numel(out_shape), 1)
    out_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(int64_dtype), runtime=runtime)
    aclnn.argsort(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        dim,
        bool(descending),
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, numel, int64_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def sort(a, dim=-1, descending=False, stable=False):
    if a.device.type != "npu":
        raise ValueError("NPU sort expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())

    # Keep runtime stable for default unstable sort path.
    if not stable:
        return topk(a, k=a.shape[dim], dim=dim, largest=bool(descending), sorted=True)

    if not aclnn.sort_symbols_ok():
        raise RuntimeError("aclnnSort symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    numel = max(_numel(out_shape), 1)
    values_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
    indices_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(int64_dtype), runtime=runtime)
    aclnn.sort(
        _unwrap_storage(a).data_ptr(),
        values_ptr,
        indices_ptr,
        a.shape,
        a.stride,
        dim,
        bool(descending),
        bool(stable),
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    values_storage = npu_typed_storage_from_ptr(values_ptr, numel, a.dtype, device=a.device)
    indices_storage = npu_typed_storage_from_ptr(indices_ptr, numel, int64_dtype, device=a.device)
    return _wrap_tensor(values_storage, out_shape, out_stride), _wrap_tensor(indices_storage, out_shape, out_stride)


def topk(a, k, dim=-1, largest=True, sorted=True):
    if a.device.type != "npu":
        raise ValueError("NPU topk expects NPU tensors")
    k = int(k)
    dim = _normalize_dim(dim, a.dim())
    dim_size = a.shape[dim]
    if k < 0 or k > dim_size:
        raise RuntimeError("selected index k out of range")
    if not aclnn.topk_symbols_ok():
        raise RuntimeError("aclnnTopk symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = list(a.shape)
    out_shape[dim] = int(k)
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    numel = max(_numel(out_shape), 1)
    values_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
    indices_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(int64_dtype), runtime=runtime)
    aclnn.topk(
        _unwrap_storage(a).data_ptr(),
        values_ptr,
        indices_ptr,
        a.shape,
        a.stride,
        int(k),
        dim,
        bool(largest),
        bool(sorted),
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    values_storage = npu_typed_storage_from_ptr(values_ptr, numel, a.dtype, device=a.device)
    indices_storage = npu_typed_storage_from_ptr(indices_ptr, numel, int64_dtype, device=a.device)
    return _wrap_tensor(values_storage, out_shape, out_stride), _wrap_tensor(indices_storage, out_shape, out_stride)


def tril(a, diagonal=0):
    if a.device.type != "npu":
        raise ValueError("NPU tril expects NPU tensors")
    if not aclnn.tril_symbols_ok():
        raise RuntimeError("aclnnTril symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.tril(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        int(diagonal),
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def triu(a, diagonal=0):
    if a.device.type != "npu":
        raise ValueError("NPU triu expects NPU tensors")
    if not aclnn.triu_symbols_ok():
        raise RuntimeError("aclnnTriu symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.triu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        int(diagonal),
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def rot90(a, k=1, dims=(0, 1)):
    if a.device.type != "npu":
        raise ValueError("NPU rot90 expects NPU tensors")
    if a.dim() < 2:
        raise RuntimeError(f"expected total dims >= 2, but got total dims = {a.dim()}")
    if not isinstance(dims, (tuple, list)) or len(dims) != 2:
        raise RuntimeError("rot90 expects dims to be a tuple of length 2")

    dim0 = _normalize_dim(int(dims[0]), a.dim())
    dim1 = _normalize_dim(int(dims[1]), a.dim())
    if dim0 == dim1:
        raise RuntimeError(f"expected rotation dims to be different, but got dim0 = {dim0} and dim1 = {dim1}")

    k = int(k) % 4
    if k == 0:
        return a
    if k == 1:
        return view_backend.transpose(flip(a, dims=(dim1,)), dim0, dim1)
    if k == 2:
        return flip(flip(a, dims=(dim0,)), dims=(dim1,))
    return view_backend.transpose(flip(a, dims=(dim0,)), dim0, dim1)


def repeat(a, repeats):
    if a.device.type != "npu":
        raise ValueError("NPU repeat expects NPU tensors")
    repeats = _normalize_repeats_tuple(repeats, a.dim(), "repeat")
    if any(int(r) < 0 for r in repeats):
        raise RuntimeError(f"Trying to create tensor with negative dimension {tuple(int(s) * int(r) for s, r in zip(a.shape, repeats))}")
    if not aclnn.repeat_symbols_ok():
        raise RuntimeError("aclnnRepeat symbols not available")

    out_shape = tuple(int(s) * int(r) for s, r in zip(a.shape, repeats))
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = max(_numel(out_shape), 1)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(out_numel * _dtype_itemsize(a.dtype), runtime=runtime)

    aclnn.repeat(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        repeats,
        out_shape,
        out_stride,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def tile(a, dims):
    if isinstance(dims, int):
        raise TypeError("tile(): argument 'dims' (position 2) must be tuple of ints, not int")
    return repeat(a, dims)


def repeat_interleave(a, repeats, dim=None):
    if a.device.type != "npu":
        raise ValueError("NPU repeat_interleave expects NPU tensors")

    from .creation import zeros_create

    if isinstance(repeats, int) and aclnn.repeat_interleave_int_symbols_ok():
        rep = int(repeats)
        if rep < 0:
            raise ValueError("repeats must be non-negative")
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        stream = npu_state.current_stream((a.device.index or 0))
        if dim is None:
            in_tensor = view_backend.reshape(a, (a.numel(),))
            output_size = in_tensor.numel() * rep
            out_shape = (output_size,)
        else:
            dim = _normalize_dim(dim, a.dim())
            in_tensor = a
            output_size = in_tensor.shape[dim] * rep
            out_shape = list(in_tensor.shape)
            out_shape[dim] = output_size
            out_shape = tuple(out_shape)

        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype), runtime=runtime)
        aclnn.repeat_interleave_int(
            _unwrap_storage(in_tensor).data_ptr(),
            out_ptr,
            in_tensor.shape,
            in_tensor.stride,
            in_tensor.dtype,
            rep,
            None if dim is None else int(dim),
            int(output_size),
            out_shape,
            out_stride,
            runtime,
            stream=stream.stream,
        )
        out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
        return _wrap_tensor(out_storage, out_shape, out_stride)

    if dim is None:
        flat = view_backend.reshape(a, (a.numel(),))
        idx, out_size = _build_repeat_interleave_indices(flat.shape[0], repeats, a.device)
        if out_size == 0:
            return zeros_create((0,), dtype=a.dtype, device=a.device)
        return index_select(flat, 0, idx)

    dim = _normalize_dim(dim, a.dim())
    idx, out_size = _build_repeat_interleave_indices(a.shape[dim], repeats, a.device)
    out_shape = list(a.shape)
    out_shape[dim] = out_size
    out_shape = tuple(out_shape)
    if out_size == 0:
        return zeros_create(out_shape, dtype=a.dtype, device=a.device)
    return index_select(a, dim, idx)


def scatter(a, dim, index, src):
    if a.device.type != "npu":
        raise ValueError("NPU scatter expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(index, "scatter")
    if index.dim() != a.dim():
        raise ValueError("index shape mismatch")
    for i, size in enumerate(index.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("index shape mismatch")
    _validate_index_bounds(index, a.shape[dim], allow_negative=False, name="scatter")

    if hasattr(src, "shape"):
        if src.device.type != "npu":
            raise ValueError("scatter src tensor must be on NPU")
        src_tensor = src
    else:
        src_tensor = _scalar_to_npu_tensor(src, a)

    if src_tensor.shape != index.shape:
        src_tensor = _npu_broadcast_to(src_tensor, index.shape)

    if not aclnn.scatter_symbols_ok():
        raise RuntimeError("aclnnScatter symbols not available")

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = tuple(a.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype), runtime=runtime)

    aclnn.scatter(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(index).data_ptr(),
        _unwrap_storage(src_tensor).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        index.shape,
        index.stride,
        index.dtype,
        src_tensor.shape,
        src_tensor.stride,
        src_tensor.dtype,
        dim,
        0,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def tril_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    if layout is not None and layout != "strided":
        raise ValueError("layout must be strided")
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype

    from .creation import tensor_create
    from ..._device import device as Device

    dev = Device("cpu") if device is None else (Device(device) if isinstance(device, str) else device)
    row = int(row)
    col = int(col)
    offset = int(offset)

    rows = []
    cols = []
    for r in range(row):
        upper = min(col - 1, r + offset)
        if upper < 0:
            continue
        for c in range(upper + 1):
            rows.append(r)
            cols.append(c)

    return tensor_create([rows, cols], dtype=dtype, device=dev)


def triu_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    if layout is not None and layout != "strided":
        raise ValueError("layout must be strided")
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype

    from .creation import tensor_create
    from ..._device import device as Device

    dev = Device("cpu") if device is None else (Device(device) if isinstance(device, str) else device)
    row = int(row)
    col = int(col)
    offset = int(offset)

    rows = []
    cols = []
    for r in range(row):
        start = max(0, r + offset)
        if start >= col:
            continue
        for c in range(start, col):
            rows.append(r)
            cols.append(c)

    return tensor_create([rows, cols], dtype=dtype, device=dev)


def diag(a, diagonal=0):
    if a.device.type != "npu":
        raise ValueError("NPU diag expects NPU tensors")
    if a.dim() not in (1, 2):
        raise ValueError("diag expects 1D or 2D tensor")
    if not aclnn.diag_symbols_ok():
        raise RuntimeError("aclnnDiag symbols not available")

    from ..meta import infer as meta_infer
    spec = meta_infer.infer_diag(a, diagonal=diagonal)
    out_shape = spec.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(a.dtype), runtime=runtime)

    aclnn.diag(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        int(diagonal),
        out_shape,
        out_stride,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def cartesian_prod(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)
    if len(tensors) == 0:
        raise RuntimeError("cartesian_prod expects at least one tensor")
    first = tensors[0]
    for t in tensors:
        if t.device.type != "npu":
            raise ValueError("NPU cartesian_prod expects NPU tensors")
        if t.dim() != 1:
            raise ValueError("cartesian_prod expects 1D tensors")
        if t.dtype != first.dtype:
            raise RuntimeError("meshgrid expects all tensors to have the same dtype")

    from .creation import tensor_create

    cols = [t.to("cpu").tolist() for t in tensors]
    if any(len(c) == 0 for c in cols):
        return tensor_create([], dtype=first.dtype, device=first.device).reshape((0, len(tensors)))

    rows = [[]]
    for c in cols:
        rows = [prefix + [v] for prefix in rows for v in c]
    return tensor_create(rows, dtype=first.dtype, device=first.device)


def block_diag(*tensors):
    tensors = _normalize_tensor_sequence_args(tensors)
    from .creation import tensor_create

    if len(tensors) == 0:
        return tensor_create([[]], dtype=float_dtype, device="cpu")

    first = tensors[0]
    for t in tensors:
        if t.device.type != "npu":
            raise ValueError("NPU block_diag expects NPU tensors")
        if t.dim() != 2:
            raise ValueError("block_diag expects 2D tensors")
        if t.dtype != first.dtype:
            raise ValueError("block_diag expects tensors with the same dtype")

    rows = sum(int(t.shape[0]) for t in tensors)
    cols = sum(int(t.shape[1]) for t in tensors)
    out = [[0 for _ in range(cols)] for _ in range(rows)]

    r0 = 0
    c0 = 0
    for t in tensors:
        data = t.to("cpu").tolist()
        h = int(t.shape[0])
        w = int(t.shape[1])
        for i in range(h):
            row_vals = data[i]
            for j in range(w):
                out[r0 + i][c0 + j] = row_vals[j]
        r0 += h
        c0 += w

    return tensor_create(out, dtype=first.dtype, device=first.device)


def nonzero(a, as_tuple=False):
    if a.device.type != "npu":
        raise ValueError("NPU nonzero expects NPU tensors")
    if not aclnn.nonzero_symbols_ok():
        raise RuntimeError("aclnnNonzero symbols not available")

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    out_shape = (max(a.numel(), 1), a.dim())
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(int64_dtype), runtime=runtime)

    aclnn.nonzero(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        out_shape,
        out_stride,
        runtime,
        stream=stream.stream,
    )

    full_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), int64_dtype, device=a.device)
    full = _wrap_tensor(full_storage, out_shape, out_stride)

    nonzero_count = count_nonzero(a, dim=None, keepdim=False)
    rows = _read_int64_scalar(nonzero_count)

    if rows < out_shape[0]:
        full = _slice_along_dim(full, 0, rows, 0)

    if not as_tuple:
        return full

    from .creation import zeros_create
    from ..common import view as view_backend

    if a.dim() == 0:
        if rows == 0:
            return (zeros_create((0,), dtype=int64_dtype, device=a.device),)
        return (zeros_create((1,), dtype=int64_dtype, device=a.device),)

    outputs = []
    for dim_idx in range(a.dim()):
        col = _slice_along_dim(full, dim_idx, dim_idx + 1, 1)
        outputs.append(view_backend.reshape(col, (rows,)))
    return tuple(outputs)


def lerp(a, b, weight):
    if not hasattr(weight, "shape"):
        weight = _scalar_to_npu_tensor(weight, a)
    return add(a, mul(weight, sub(b, a)))


def addcmul(a, b, c, value=1.0):
    if not hasattr(value, "shape"):
        value = _scalar_to_npu_tensor(value, a)
    return add(a, mul(value, mul(b, c)))


def addcdiv(a, b, c, value=1.0):
    if not hasattr(value, "shape"):
        value = _scalar_to_npu_tensor(value, a)
    return add(a, mul(value, div(b, c)))


def logaddexp(a, b):
    m = max_(a, b)
    return add(m, log(add(exp(sub(a, m)), exp(sub(b, m)))))


def logaddexp2(a, b):
    m = max_(a, b)
    return add(m, log2(add(exp2(sub(a, m)), exp2(sub(b, m)))))


def hypot(a, b):
    return sqrt(add(mul(a, a), mul(b, b)))


def remainder(a, b):
    return sub(a, mul(floor(div(a, b)), b))


def fmod(a, b):
    return sub(a, mul(trunc(div(a, b)), b))


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    from ..._tensor import Tensor
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise ValueError("NPU allclose expects tensors")
    diff = abs(sub(a, b))
    tol = add(_scalar_to_npu_tensor(atol, diff), mul(_scalar_to_npu_tensor(rtol, diff), abs(b)))
    close = le(diff, tol)
    if equal_nan:
        nan_match = logical_and(isnan(a), isnan(b))
        close = logical_or(close, nan_match)
    return all_(close).item()


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    diff = abs(sub(a, b))
    tol = add(_scalar_to_npu_tensor(atol, diff), mul(_scalar_to_npu_tensor(rtol, diff), abs(b)))
    close = le(diff, tol)
    if equal_nan:
        nan_match = logical_and(isnan(a), isnan(b))
        close = logical_or(close, nan_match)
    return close


def equal(a, b):
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU equal expects NPU tensors")
    neq = ne(a, b)
    return logical_not(any_(neq)).item()



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
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU add_ expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU add_ requires matching dtypes")
    out_shape = _broadcast_shape(a.shape, b.shape)
    if out_shape != a.shape:
        raise ValueError("NPU add_ requires broadcastable to self shape")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.add(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        out_shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def _scalar_to_npu_tensor(scalar, ref_tensor):
    """Convert scalar to NPU tensor matching ref_tensor's shape/dtype/device."""
    runtime = npu_runtime.get_runtime((ref_tensor.device.index or 0))
    stream = npu_state.current_stream((ref_tensor.device.index or 0))
    out_shape = ref_tensor.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(ref_tensor.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    # Fill on host then memcpy H2D to avoid aclnn scalar ops.
    from . import acl_loader
    import ctypes
    import struct
    acl = acl_loader.ensure_acl()
    host_ptr, ret = acl.rt.malloc_host(int(out_size))
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc_host failed: {ret}")
    try:
        host_buf = (ctypes.c_uint8 * int(out_size)).from_address(int(host_ptr))
        itemsize = _dtype_itemsize(ref_tensor.dtype)
        dtype_name = getattr(ref_tensor.dtype, "name", None) or str(ref_tensor.dtype).split(".")[-1]
        if dtype_name == "float16":
            from .aclnn import _float_to_float16_bits
            bits = _float_to_float16_bits(float(scalar))
            pattern = int(bits).to_bytes(2, byteorder="little", signed=False)
        elif dtype_name == "bfloat16":
            from .aclnn import _float_to_bfloat16_bits
            bits = _float_to_bfloat16_bits(float(scalar))
            pattern = int(bits).to_bytes(2, byteorder="little", signed=False)
        elif dtype_name == "float32":
            pattern = struct.pack("<f", float(scalar))
        elif dtype_name == "float64":
            pattern = struct.pack("<d", float(scalar))
        elif dtype_name == "int8":
            pattern = int(scalar).to_bytes(1, byteorder="little", signed=True)
        elif dtype_name == "uint8":
            pattern = int(scalar).to_bytes(1, byteorder="little", signed=False)
        elif dtype_name == "int16":
            pattern = int(scalar).to_bytes(2, byteorder="little", signed=True)
        elif dtype_name == "int32":
            pattern = int(scalar).to_bytes(4, byteorder="little", signed=True)
        elif dtype_name == "int64":
            pattern = int(scalar).to_bytes(8, byteorder="little", signed=True)
        elif dtype_name == "bool":
            pattern = (1 if bool(scalar) else 0).to_bytes(1, byteorder="little", signed=False)
        else:
            raise ValueError(f"Unsupported scalar dtype: {dtype_name}")
        for offset in range(0, int(out_size), itemsize):
            host_buf[offset:offset + itemsize] = pattern
        runtime.activate()
        ret = acl.rt.memcpy(out_ptr, int(out_size), host_ptr, int(out_size), npu_runtime.ACL_MEMCPY_HOST_TO_DEVICE)
        if ret != npu_runtime.ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.memcpy H2D failed: {ret}")
    finally:
        acl.rt.free_host(host_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), ref_tensor.dtype, device=ref_tensor.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)










def _scalar_to_npu_tensor_no_add(scalar, ref_tensor):
    """Helper to avoid recursion: create scalar using add_scalar without add()."""
    runtime = npu_runtime.get_runtime((ref_tensor.device.index or 0))
    stream = npu_state.current_stream((ref_tensor.device.index or 0))
    out_shape = ref_tensor.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(ref_tensor.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.inplace_zero(
        out_ptr,
        out_shape,
        out_stride,
        ref_tensor.dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.add_scalar(
        out_ptr,
        scalar,
        out_ptr,
        out_shape,
        out_stride,
        ref_tensor.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), ref_tensor.dtype, device=ref_tensor.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)






def _nan_like(a):
    return _scalar_to_npu_tensor(float("nan"), a)



def mul_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU mul_ expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU mul_ requires matching dtypes")
    out_shape = _broadcast_shape(a.shape, b.shape)
    if out_shape != a.shape:
        raise ValueError("NPU mul_ requires broadcastable to self shape")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.mul(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        out_shape,
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


def uniform_(a, low=0.0, high=1.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU uniform_ expects NPU tensors")

    from ... import npu as npu_mod
    seed = npu_mod._get_seed()
    offset = npu_mod._get_and_advance_offset(advance=_numel(a.shape))

    a_storage = _unwrap_storage(a)
    aclnn.inplace_uniform(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        float(low),
        float(high),
        seed,
        offset,
        runtime,
        stream=stream.stream,
    )
    return a


def normal_(a, mean=0.0, std=1.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU normal_ expects NPU tensors")

    from ... import npu as npu_mod
    seed = npu_mod._get_seed()
    offset = npu_mod._get_and_advance_offset(advance=_numel(a.shape))

    a_storage = _unwrap_storage(a)
    aclnn.inplace_normal(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        float(mean),
        float(std),
        seed,
        offset,
        runtime,
        stream=stream.stream,
    )
    return a


def fill_(a, value):
    """In-place fill using aclnnInplaceFillScalar."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU fill_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    aclnn.inplace_fill_scalar(
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        float(value),
        runtime,
        stream=stream.stream,
    )
    return a


def clamp_(a, min_val=None, max_val=None):
    """In-place clamp: output written back to a's storage."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    # Use clamp_scalar with output == input for in-place
    aclnn.clamp_scalar(
        a_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        min_val,
        max_val,
        runtime,
        stream=stream.stream,
    )
    return a


def copy_(a, src):
    """In-place copy from src into a."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU copy_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    src_storage = _unwrap_storage(src)
    aclnn.inplace_copy(
        a_storage.data_ptr(),
        src_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        src.shape,
        src.stride,
        src.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def erfinv_(a):
    """In-place erfinv using aclnnErfinv."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU erfinv_ expects NPU tensors")

    a_storage = _unwrap_storage(a)
    # erfinv: output to same storage for in-place
    aclnn.erfinv(
        a_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def sub_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU sub_ expects NPU tensors")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.sub(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        b.shape,
        b.stride,
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
            slice_shape = (length,) + tuple(tensor.shape[1:])
            if tensor.stride:
                slice_stride = (tensor.stride[0],) + tuple(tensor.stride[1:])
            else:
                slice_stride = npu_runtime._contiguous_stride(slice_shape)
            stream = npu_state.current_stream((tensor.device.index or 0))
            temp_ptr = npu_runtime._alloc_device(copy_size, runtime=runtime)
            aclnn.inplace_zero(
                temp_ptr,
                slice_shape,
                slice_stride,
                tensor.dtype,
                runtime,
                stream=stream.stream,
            )
            # Fill via host buffer then copy into destination
            temp_storage = npu_typed_storage_from_ptr(temp_ptr, slice_numel, tensor.dtype, device=tensor.device)
            temp_tensor = _wrap_tensor(temp_storage, slice_shape, slice_stride)
            filled = _scalar_to_npu_tensor(value, temp_tensor)
            ret = npu_runtime.acl.rt.memcpy(dst_ptr, copy_size, filled.storage().data_ptr(), copy_size, 3)
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy failed: {ret}")
            runtime.defer_free(temp_ptr)
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


def cat(tensors, dim=0):
    """Concatenate tensors along an existing dimension using aclnnCat."""
    if not tensors:
        raise RuntimeError("cat requires at least one tensor")
    if len(tensors) == 1:
        return contiguous(tensors[0])

    first = tensors[0]
    runtime = npu_runtime.get_runtime((first.device.index or 0))
    stream = npu_state.current_stream((first.device.index or 0))

    if not aclnn.cat_symbols_ok():
        raise RuntimeError("aclnnCat not available")

    ndim = len(first.shape)
    if dim < 0:
        dim += ndim

    # Validate shapes and compute output shape
    out_shape = list(first.shape)
    for t in tensors[1:]:
        if len(t.shape) != ndim:
            raise RuntimeError("cat: tensors must have the same number of dimensions")
        for d in range(ndim):
            if d != dim and t.shape[d] != first.shape[d]:
                raise RuntimeError(f"cat: dimension {d} size mismatch")
        out_shape[dim] += t.shape[dim]
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(first.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Prepare inputs for aclnn
    tensor_ptrs = [_unwrap_storage(t).data_ptr() for t in tensors]
    shapes = [t.shape for t in tensors]
    strides = [t.stride for t in tensors]
    dtypes = [t.dtype for t in tensors]

    aclnn.cat(
        tensor_ptrs, shapes, strides, dtypes,
        dim, out_ptr, out_shape, out_stride, first.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, first.dtype, device=first.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def concatenate(tensors, dim=0):
    return cat(tensors, dim=dim)


def stack(tensors, dim=0):
    """Stack tensors along a new dimension using aclnnStack."""
    if not tensors:
        raise RuntimeError("stack requires at least one tensor")

    first = tensors[0]
    runtime = npu_runtime.get_runtime((first.device.index or 0))
    stream = npu_state.current_stream((first.device.index or 0))

    if not aclnn.stack_symbols_ok():
        raise RuntimeError("aclnnStack not available")

    ndim = len(first.shape)
    if dim < 0:
        dim += ndim + 1

    # Validate shapes
    for t in tensors[1:]:
        if t.shape != first.shape:
            raise RuntimeError("stack: all tensors must have the same shape")

    # Compute output shape: insert new dimension with size = len(tensors)
    out_shape = list(first.shape[:dim]) + [len(tensors)] + list(first.shape[dim:])
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(first.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Prepare inputs for aclnn
    tensor_ptrs = [_unwrap_storage(t).data_ptr() for t in tensors]
    shapes = [t.shape for t in tensors]
    strides = [t.stride for t in tensors]
    dtypes = [t.dtype for t in tensors]

    aclnn.stack(
        tensor_ptrs, shapes, strides, dtypes,
        dim, out_ptr, out_shape, out_stride, first.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, first.dtype, device=first.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def mean(a, dim=None, keepdim=False):
    """Compute mean along dimensions using aclnnMean."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.mean_symbols_ok():
        raise RuntimeError("aclnnMean not available")

    # Compute output shape
    if dim is None:
        dims = list(range(len(a.shape)))
    elif isinstance(dim, int):
        dims = [dim if dim >= 0 else dim + len(a.shape)]
    else:
        dims = [d if d >= 0 else d + len(a.shape) for d in dim]

    out_shape = list(a.shape)
    for d in sorted(dims, reverse=True):
        if keepdim:
            out_shape[d] = 1
        else:
            out_shape.pop(d)
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.mean(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        dims, keepdim,
        out_shape, out_stride,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def softmax(a, dim=-1):
    """Compute softmax along a dimension using aclnnSoftmax."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.softmax_symbols_ok():
        raise RuntimeError("aclnnSoftmax not available")

    # Normalize dim
    if dim < 0:
        dim += len(a.shape)

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.softmax(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        dim,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def log_softmax(a, dim=-1):
    """Compute log_softmax along a dimension using aclnnLogSoftmax."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.log_softmax_symbols_ok():
        raise RuntimeError("aclnnLogSoftmax not available")

    # Normalize dim
    if dim < 0:
        dim += len(a.shape)

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.log_softmax(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        dim,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def gelu(a):
    """Compute GELU activation using aclnnGelu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.gelu_symbols_ok():
        raise RuntimeError("aclnnGelu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.gelu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Compute layer normalization using aclnnLayerNorm."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    if not aclnn.layer_norm_symbols_ok():
        raise RuntimeError("aclnnLayerNorm not available")

    # Compute stats shape (all dims except normalized dims)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    num_normalized_dims = len(normalized_shape)
    # Stats (mean/rstd) must have same rank as input, with normalized dims replaced by 1
    if num_normalized_dims > 0:
        stats_shape = tuple(
            s if i < len(input.shape) - num_normalized_dims else 1
            for i, s in enumerate(input.shape)
        )
    else:
        stats_shape = input.shape
    stats_stride = npu_runtime._contiguous_stride(stats_shape)
    stats_numel = _numel(stats_shape)

    out_shape = input.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)

    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)
    # Mean and rstd are always float32
    mean_ptr = npu_runtime._alloc_device(stats_numel * 4, runtime=runtime)  # float32 = 4 bytes
    rstd_ptr = npu_runtime._alloc_device(stats_numel * 4, runtime=runtime)  # float32 = 4 bytes

    weight_ptr = _unwrap_storage(weight).data_ptr() if weight is not None else None
    bias_ptr = _unwrap_storage(bias).data_ptr() if bias is not None else None

    aclnn.layer_norm(
        _unwrap_storage(input).data_ptr(),
        weight_ptr,
        bias_ptr,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        input.shape, input.stride,
        weight.shape if weight is not None else (),
        weight.stride if weight is not None else (),
        bias.shape if bias is not None else (),
        bias.stride if bias is not None else (),
        out_shape, out_stride,
        stats_shape, stats_stride,
        normalized_shape,
        eps,
        input.dtype,
        runtime, stream=stream.stream
    )

    runtime.defer_free(mean_ptr)
    runtime.defer_free(rstd_ptr)

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def embedding(weight, indices, padding_idx=None, scale_grad_by_freq=False, sparse=False):
    """Compute embedding lookup using aclnnEmbedding."""
    runtime = npu_runtime.get_runtime((weight.device.index or 0))
    stream = npu_state.current_stream((weight.device.index or 0))

    if not aclnn.embedding_symbols_ok():
        raise RuntimeError("aclnnEmbedding not available")

    # Output shape: indices.shape + (embedding_dim,)
    embedding_dim = weight.shape[1] if len(weight.shape) > 1 else weight.shape[0]
    out_shape = indices.shape + (embedding_dim,)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(weight.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Note: aclnnEmbedding doesn't support padding_idx, scale_grad_by_freq, sparse parameters
    # These are ignored for now
    aclnn.embedding(
        _unwrap_storage(weight).data_ptr(),
        _unwrap_storage(indices).data_ptr(),
        out_ptr,
        weight.shape, weight.stride,
        indices.shape, indices.stride,
        out_shape, out_stride,
        weight.dtype,
        indices.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, weight.dtype, device=weight.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)



def silu(a):
    """Compute SiLU (Swish) activation using aclnnSilu."""
    if not aclnn.silu_symbols_ok():
        raise RuntimeError("aclnnSilu not available")
    return _unary_op(a, aclnn.silu, "silu")


def leaky_relu(a, negative_slope=0.01):
    """Compute Leaky ReLU activation using aclnnLeakyRelu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.leaky_relu_symbols_ok():
        raise RuntimeError("aclnnLeakyRelu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.leaky_relu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        negative_slope,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def elu(a, alpha=1.0):
    """Compute ELU activation using aclnnElu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.elu_symbols_ok():
        raise RuntimeError("aclnnElu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.elu(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape, a.stride, a.dtype,
        alpha,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def mish(a):
    """Compute Mish activation using aclnnMish."""
    if not aclnn.mish_symbols_ok():
        raise RuntimeError("aclnnMish not available")
    return _unary_op(a, aclnn.mish, "mish")


def prelu(a, weight):
    """Compute PReLU activation using aclnnPrelu."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.prelu_symbols_ok():
        raise RuntimeError("aclnnPrelu not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    aclnn.prelu(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(weight).data_ptr(),
        out_ptr,
        a.shape, a.stride,
        weight.shape, weight.stride,
        a.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """Compute batch normalization using aclnnBatchNorm."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    if not aclnn.batch_norm_symbols_ok():
        raise RuntimeError("aclnnBatchNorm not available")

    out_shape = input.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    weight_ptr = _unwrap_storage(weight).data_ptr() if weight is not None else None
    bias_ptr = _unwrap_storage(bias).data_ptr() if bias is not None else None
    running_mean_ptr = _unwrap_storage(running_mean).data_ptr() if running_mean is not None else None
    running_var_ptr = _unwrap_storage(running_var).data_ptr() if running_var is not None else None

    aclnn.batch_norm(
        _unwrap_storage(input).data_ptr(),
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        out_ptr,
        input.shape, input.stride,
        weight.shape if weight is not None else (),
        weight.stride if weight is not None else (),
        bias.shape if bias is not None else (),
        bias.stride if bias is not None else (),
        running_mean.shape if running_mean is not None else (),
        running_mean.stride if running_mean is not None else (),
        running_var.shape if running_var is not None else (),
        running_var.stride if running_var is not None else (),
        out_shape, out_stride,
        training, momentum, eps,
        input.dtype,
        runtime, stream=stream.stream
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """Compute group normalization using aclnnLayerNorm (composite implementation).

    This avoids the aclnnGroupNorm state contamination bug in CANN 8.3.RC2.
    Algorithm:
    1. Reshape input from (N, C, H, W) to (N*num_groups, C//num_groups * H * W)
    2. Apply layer_norm over the last dimension (normalizes each group independently)
    3. Reshape back to (N, C, H, W)
    4. Apply affine transform: result * weight + bias
    """
    if not aclnn.layer_norm_symbols_ok():
        raise RuntimeError("aclnnLayerNorm not available (required for group_norm)")

    # Extract dimensions
    N = input.shape[0]
    C = input.shape[1]
    spatial_dims = input.shape[2:]
    spatial_size = 1
    for dim in spatial_dims:
        spatial_size *= dim

    if C % num_groups != 0:
        raise ValueError(f"num_channels ({C}) must be divisible by num_groups ({num_groups})")

    channels_per_group = C // num_groups

    # Step 1: Reshape to (N*num_groups, channels_per_group * spatial_size)
    reshaped_shape = (N * num_groups, channels_per_group * spatial_size)
    reshaped = reshape(input, reshaped_shape)

    # Step 2: Apply layer_norm over the last dimension (no weight/bias yet)
    normalized_shape = (channels_per_group * spatial_size,)
    normalized = layer_norm(reshaped, normalized_shape, weight=None, bias=None, eps=eps)

    # Step 3: Reshape back to original shape
    result = reshape(normalized, input.shape)

    # Step 4: Apply affine transform if weight/bias provided
    if weight is not None:
        # Reshape weight from (C,) to (1, C, 1, 1, ...) for broadcasting
        weight_shape = (1, C) + (1,) * len(spatial_dims)
        weight_reshaped = reshape(weight, weight_shape)
        result = mul(result, weight_reshaped)

    if bias is not None:
        # Reshape bias from (C,) to (1, C, 1, 1, ...) for broadcasting
        bias_shape = (1, C) + (1,) * len(spatial_dims)
        bias_reshaped = reshape(bias, bias_shape)
        result = add(result, bias_reshaped)

    return result


def dropout(a, p=0.5, training=True):
    """Compute dropout using aclnnDropoutGenMask + aclnnDropoutDoMask."""
    if not training or p == 0:
        return a

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if not aclnn.dropout_symbols_ok():
        raise RuntimeError("aclnnDropout symbols not available")

    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Allocate mask (bit-packed: align(numel, 128) / 8 bytes)
    mask_numel = (out_numel + 127) // 128 * 128 // 8
    mask_ptr = npu_runtime._alloc_device(mask_numel, runtime=runtime)

    # Get seed and offset from npu module
    from ... import npu as npu_mod
    seed = npu_mod._get_seed()
    offset = npu_mod._get_and_advance_offset(advance=out_numel)

    # Step 1: Generate mask
    aclnn.dropout_gen_mask(
        a.shape, p, seed, offset,
        mask_ptr, mask_numel,
        runtime, stream=stream.stream
    )

    # Step 2: Apply mask
    aclnn.dropout_do_mask(
        _unwrap_storage(a).data_ptr(),
        mask_ptr,
        out_ptr,
        a.shape, a.stride, a.dtype,
        mask_numel, p,
        runtime, stream=stream.stream
    )

    # Free mask - we don't need it
    runtime.defer_free(mask_ptr)

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def pad(input, pad, mode='constant', value=0):
    if input.device.type != "npu":
        raise ValueError("NPU pad expects NPU tensors")
    if mode != "constant":
        raise NotImplementedError("NPU pad currently supports constant mode only")
    if not isinstance(pad, (tuple, list)):
        raise TypeError("pad must be a tuple/list of ints")
    if len(pad) % 2 != 0:
        raise ValueError("pad length must be even")
    if len(pad) > 2 * input.dim():
        raise ValueError("padding length too large")
    pad_vals = tuple(int(v) for v in pad)

    out_shape = list(input.shape)
    n_pairs = len(pad_vals) // 2
    for i in range(n_pairs):
        dim = input.dim() - 1 - i
        left = pad_vals[2 * i]
        right = pad_vals[2 * i + 1]
        out_shape[dim] = out_shape[dim] + left + right
        if out_shape[dim] < 0:
            raise RuntimeError("negative output size is not supported")
    out_shape = tuple(out_shape)

    out_stride = npu_runtime._contiguous_stride(out_shape)
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))
    out_ptr = npu_runtime._alloc_device(max(_numel(out_shape), 1) * _dtype_itemsize(input.dtype), runtime=runtime)

    if not aclnn.constant_pad_nd_symbols_ok():
        raise RuntimeError("aclnnConstantPadNd symbols not available")

    aclnn.constant_pad_nd(
        _unwrap_storage(input).data_ptr(),
        out_ptr,
        input.shape,
        input.stride,
        input.dtype,
        pad_vals,
        value,
        out_shape,
        out_stride,
        input.dtype,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(_numel(out_shape), 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def pad_sequence(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    if not seqs:
        raise ValueError("pad_sequence expects a non-empty list of tensors")

    first = seqs[0]
    if first.device.type != "npu":
        raise ValueError("NPU pad_sequence expects NPU tensors")
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be 'left' or 'right'")

    from .creation import full_create

    max_len = max(int(t.shape[0]) for t in seqs)
    batch = len(seqs)
    trailing = tuple(first.shape[1:])
    trailing_numel = 1
    for d in trailing:
        trailing_numel *= int(d)

    if batch_first:
        out_shape = (batch, max_len) + trailing
    else:
        out_shape = (max_len, batch) + trailing
    out = full_create(out_shape, padding_value, dtype=first.dtype, device=first.device)

    itemsize = _dtype_itemsize(first.dtype)
    dst_base = int(_unwrap_storage(out).data_ptr())
    out_stride = out.stride

    for i, t in enumerate(seqs):
        if t.device.type != "npu":
            raise ValueError("all tensors must be NPU tensors")
        if t.dtype != first.dtype:
            raise ValueError("all tensors must have the same dtype")
        if tuple(t.shape[1:]) != trailing:
            raise ValueError("all tensors must have the same trailing dimensions")

        src = t if t.is_contiguous() else contiguous(t)
        length = int(src.shape[0])
        start_idx = max_len - length if padding_side == "left" else 0

        src_base = int(_unwrap_storage(src).data_ptr())
        if batch_first:
            dst_elem_offset = int(i) * int(out_stride[0]) + int(start_idx) * int(out_stride[1])
            copy_bytes = int(length * trailing_numel * itemsize)
            ret = npu_runtime.acl.rt.memcpy(
                dst_base + dst_elem_offset * itemsize,
                copy_bytes,
                src_base,
                copy_bytes,
                3,
            )
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy failed: {ret}")
        else:
            block_bytes = int(trailing_numel * itemsize)
            for step in range(length):
                dst_elem_offset = (int(start_idx + step) * int(out_stride[0])) + (int(i) * int(out_stride[1]))
                src_elem_offset = int(step) * int(src.stride[0])
                ret = npu_runtime.acl.rt.memcpy(
                    dst_base + dst_elem_offset * itemsize,
                    block_bytes,
                    src_base + src_elem_offset * itemsize,
                    block_bytes,
                    3,
                )
                if ret != npu_runtime.ACL_ERROR_CODE:
                    raise RuntimeError(f"acl.rt.memcpy failed: {ret}")
    return out


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("dim out of range")
    return dim


def _split_sections_from_count(dim_size, sections):
    if sections <= 0:
        raise ValueError("sections must be > 0")
    size, extra = divmod(dim_size, sections)
    return [size + 1] * extra + [size] * (sections - extra)


def _slice_along_dim(a, start, end, dim):
    if a.device.type != "npu":
        raise ValueError("NPU slice expects NPU tensors")
    dim = _normalize_dim(dim, a.dim())
    if not a.is_contiguous():
        raise NotImplementedError("NPU split only supports contiguous input")
    dim_size = a.shape[dim]
    length = max(0, end - start)
    out_shape = list(a.shape)
    out_shape[dim] = length
    out_shape = tuple(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    runtime = npu_runtime.get_runtime((a.device.index or 0))

    if out_numel == 0:
        out_stride = npu_runtime._contiguous_stride(out_shape) if out_shape else ()
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
        return _wrap_tensor(storage, out_shape, out_stride)

    inner = 1
    for d in a.shape[dim + 1:]:
        inner *= d
    outer = 1
    for d in a.shape[:dim]:
        outer *= d

    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    src_base = int(_unwrap_storage(a).data_ptr()) + a.offset * itemsize
    dst_base = int(out_ptr)

    if inner == 1:
        block_bytes = length * itemsize
        for outer_idx in range(outer):
            src_ptr = src_base + (outer_idx * dim_size + start) * itemsize
            dst_ptr = dst_base + outer_idx * length * itemsize
            ret = npu_runtime.acl.rt.memcpy(dst_ptr, block_bytes, src_ptr, block_bytes, 3)
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy failed: {ret}")
    else:
        block_bytes = inner * itemsize
        for outer_idx in range(outer):
            src_outer = src_base + outer_idx * dim_size * inner * itemsize
            dst_outer = dst_base + outer_idx * length * inner * itemsize
            for i in range(length):
                src_ptr = src_outer + (start + i) * inner * itemsize
                dst_ptr = dst_outer + i * inner * itemsize
                ret = npu_runtime.acl.rt.memcpy(dst_ptr, block_bytes, src_ptr, block_bytes, 3)
                if ret != npu_runtime.ACL_ERROR_CODE:
                    raise RuntimeError(f"acl.rt.memcpy failed: {ret}")

    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(storage, out_shape, out_stride)


def chunk(a, chunks, dim=0):
    dim = _normalize_dim(dim, a.dim())
    dim_size = a.shape[dim]
    if chunks <= 0:
        raise ValueError("chunks must be > 0")
    actual_chunks = min(chunks, dim_size) if dim_size > 0 else chunks
    if actual_chunks == 0:
        return tuple()
    chunk_size = (dim_size + actual_chunks - 1) // actual_chunks
    outputs = []
    for i in range(actual_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, dim_size)
        if start >= end:
            break
        outputs.append(_slice_along_dim(a, start, end, dim))
    return tuple(outputs)


def split(a, split_size_or_sections, dim=0):
    dim = _normalize_dim(dim, a.dim())
    dim_size = a.shape[dim]
    outputs = []
    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            raise ValueError("split_size must be > 0")
        step = split_size_or_sections
        for start in range(0, dim_size, step):
            end = min(start + step, dim_size)
            outputs.append(_slice_along_dim(a, start, end, dim))
    else:
        sizes = list(split_size_or_sections)
        if sum(sizes) != dim_size:
            raise ValueError("split sections must sum to dim size")
        start = 0
        for size in sizes:
            end = start + size
            outputs.append(_slice_along_dim(a, start, end, dim))
            start = end
    return tuple(outputs)


def vsplit(a, split_size_or_sections):
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[0], split_size_or_sections)
        return split(a, sizes, dim=0)
    return split(a, split_size_or_sections, dim=0)


def hsplit(a, split_size_or_sections):
    dim = 0 if a.dim() == 1 else 1
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[dim], split_size_or_sections)
        return split(a, sizes, dim=dim)
    return split(a, split_size_or_sections, dim=dim)


def dsplit(a, split_size_or_sections):
    if a.dim() < 3:
        raise ValueError("dsplit expects input with at least 3 dimensions")
    if isinstance(split_size_or_sections, int):
        sizes = _split_sections_from_count(a.shape[2], split_size_or_sections)
        return split(a, sizes, dim=2)
    return split(a, split_size_or_sections, dim=2)


def unbind(a, dim=0):
    dim = _normalize_dim(dim, a.dim())
    dim_size = a.shape[dim]
    outputs = []
    from ..common import view as view_backend
    for i in range(dim_size):
        sliced = _slice_along_dim(a, i, i + 1, dim)
        out_shape = a.shape[:dim] + a.shape[dim + 1:]
        outputs.append(view_backend.reshape(sliced, out_shape))
    return tuple(outputs)


def hstack(tensors):
    if tensors[0].dim() == 1:
        return cat(tensors, dim=0)
    return cat(tensors, dim=1)


def vstack(tensors):
    from ..common import view as view_backend
    if tensors[0].dim() == 1:
        expanded = [view_backend.reshape(t, (1, t.shape[0])) for t in tensors]
        return cat(expanded, dim=0)
    return cat(tensors, dim=0)


def row_stack(tensors):
    return vstack(tensors)


def dstack(tensors):
    from ..common import view as view_backend
    expanded = []
    for t in tensors:
        if t.dim() == 1:
            expanded.append(view_backend.reshape(t, (1, t.shape[0], 1)))
        elif t.dim() == 2:
            expanded.append(view_backend.reshape(t, (t.shape[0], t.shape[1], 1)))
        else:
            expanded.append(t)
    return cat(expanded, dim=2)






def column_stack(tensors):
    from ..common import view as view_backend
    if tensors[0].dim() == 1:
        expanded = [view_backend.reshape(t, (t.shape[0], 1)) for t in tensors]
        return cat(expanded, dim=1)
    return cat(tensors, dim=1)


def _read_bool_scalar(tensor):
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    runtime.activate()
    if hasattr(runtime, "synchronize_stream"):
        runtime.synchronize_stream(stream.stream)
    buf = (ctypes.c_uint8 * 1)()
    ret = npu_runtime.acl.rt.memcpy(
        ctypes.addressof(buf),
        1,
        _unwrap_storage(tensor).data_ptr(),
        1,
        npu_runtime.ACL_MEMCPY_DEVICE_TO_HOST,
    )
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2H failed: {ret}")
    return bool(buf[0])


def _read_int64_scalar(tensor):
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    runtime.activate()
    if hasattr(runtime, "synchronize_stream"):
        runtime.synchronize_stream(stream.stream)
    buf = ctypes.c_int64()
    size = ctypes.sizeof(buf)
    ret = npu_runtime.acl.rt.memcpy(
        ctypes.addressof(buf),
        size,
        _unwrap_storage(tensor).data_ptr(),
        size,
        npu_runtime.ACL_MEMCPY_DEVICE_TO_HOST,
    )
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2H failed: {ret}")
    return int(buf.value)


def _require_int64_indices(indices, name):
    if indices.dtype != int64_dtype:
        raise ValueError(f"{name} indices must be int64")
    if indices.device.type != "npu":
        raise ValueError(f"{name} indices must be on NPU")
    return indices


def _validate_index_bounds(indices, dim_size, allow_negative, name):
    if indices.numel() == 0:
        return
    if allow_negative:
        min_ok = _scalar_to_npu_tensor(-int(dim_size), indices)
        max_ok = _scalar_to_npu_tensor(int(dim_size - 1), indices)
    else:
        min_ok = _scalar_to_npu_tensor(0, indices)
        max_ok = _scalar_to_npu_tensor(int(dim_size - 1), indices)
    below_min = lt(indices, min_ok)
    above_max = gt(indices, max_ok)
    if _read_bool_scalar(any_(below_min)) or _read_bool_scalar(any_(above_max)):
        raise IndexError(f"{name} indices out of range")

def _normalize_negative_indices(indices, dim_size):
    neg_mask = lt(indices, _scalar_to_npu_tensor(0, indices))
    if not _read_bool_scalar(any_(neg_mask)):
        return indices
    return where(neg_mask, add(indices, _scalar_to_npu_tensor(int(dim_size), indices)), indices)

def gather(a, dim, index):
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(index, "gather")
    if index.dim() != a.dim():
        raise ValueError("index shape mismatch")
    for i, size in enumerate(index.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("index shape mismatch")
    _validate_index_bounds(index, a.shape[dim], allow_negative=False, name="gather")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = index.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.gather(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(index).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        index.shape,
        index.stride,
        index.dtype,
        out_shape,
        out_stride,
        a.dtype,
        dim,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def index_select(a, dim, index):
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(index, "index_select")
    if index.dim() != 1:
        raise ValueError("index must be 1D")
    dim_size = a.shape[dim]
    _validate_index_bounds(index, dim_size, allow_negative=True, name="index_select")
    norm_index = _normalize_negative_indices(index, dim_size)
    index_shape = list(a.shape)
    index_shape[dim] = norm_index.shape[0]
    index_shape = tuple(index_shape)
    expand_shape = [1] * a.dim()
    expand_shape[dim] = norm_index.shape[0]
    expanded = view_backend.reshape(norm_index, tuple(expand_shape))
    expanded = _npu_broadcast_to(expanded, index_shape)
    return gather(a, dim, expanded)

def take(a, index):
    _require_int64_indices(index, "take")
    flat = view_backend.reshape(a, (a.numel(),))
    dim_size = flat.shape[0]
    _validate_index_bounds(index, dim_size, allow_negative=True, name="take")
    norm_index = _normalize_negative_indices(index, dim_size)
    index_shape = norm_index.shape
    gather_index = norm_index
    if gather_index.dim() == 0:
        gather_index = gather_index.reshape((1,))
    if gather_index.dim() != 1:
        gather_index = gather_index.reshape((gather_index.numel(),))
    out = gather(flat, 0, gather_index)
    return out.reshape(index_shape)

def take_along_dim(a, indices, dim):
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(indices, "take_along_dim")
    if indices.dim() != a.dim():
        raise ValueError("indices shape mismatch")
    for i, size in enumerate(indices.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("indices shape mismatch")
    dim_size = a.shape[dim]
    _validate_index_bounds(indices, dim_size, allow_negative=True, name="take_along_dim")
    norm_indices = _normalize_negative_indices(indices, dim_size)
    return gather(a, dim, norm_indices)


def masked_select(a, mask):
    if mask.dtype != bool_dtype:
        mask = ne(mask, _scalar_to_npu_tensor(0, mask))
    if mask.shape != a.shape:
        raise ValueError("mask shape mismatch")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    mask_count = count_nonzero(mask, dim=None, keepdim=False)
    out_numel = _read_int64_scalar(mask_count)
    out_shape = (a.numel(),)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(max(a.numel(), 1) * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.masked_select(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(mask).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        mask.shape,
        mask.stride,
        mask.dtype,
        out_shape,
        out_stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    full_storage = npu_typed_storage_from_ptr(out_ptr, max(a.numel(), 1), a.dtype, device=a.device)
    full = _wrap_tensor(full_storage, out_shape, out_stride)
    if out_numel == out_shape[0]:
        return full
    return _slice_along_dim(full, 0, out_numel, 0)


def linalg_qr(a, mode='reduced'):
    """QR decomposition on NPU via aclnnLinalgQr."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    a_storage = _unwrap_storage(a)
    m, n = a.shape[-2], a.shape[-1]
    k = min(m, n)

    # mode: 0 = reduced, 1 = complete
    mode_int = 1 if mode == 'complete' else 0

    if mode_int == 0:
        q_shape = a.shape[:-2] + (m, k)
        r_shape = a.shape[:-2] + (k, n)
    else:
        q_shape = a.shape[:-2] + (m, m)
        r_shape = a.shape[:-2] + (m, n)

    q_stride = npu_runtime._contiguous_stride(q_shape)
    r_stride = npu_runtime._contiguous_stride(r_shape)

    q_size = 1
    for s in q_shape:
        q_size *= s
    r_size = 1
    for s in r_shape:
        r_size *= s

    itemsize = _dtype_itemsize(a.dtype)
    q_ptr = npu_runtime._alloc_device(max(q_size, 1) * itemsize, runtime=runtime)
    r_ptr = npu_runtime._alloc_device(max(r_size, 1) * itemsize, runtime=runtime)

    aclnn.linalg_qr(
        a_storage.data_ptr(),
        q_ptr,
        r_ptr,
        a.shape,
        a.stride,
        q_shape,
        q_stride,
        r_shape,
        r_stride,
        a.dtype,
        mode_int,
        runtime,
        stream=stream.stream,
    )

    q_storage = npu_typed_storage_from_ptr(q_ptr, max(q_size, 1), a.dtype, device=a.device)
    r_storage = npu_typed_storage_from_ptr(r_ptr, max(r_size, 1), a.dtype, device=a.device)
    Q = _wrap_tensor(q_storage, q_shape, q_stride)
    R = _wrap_tensor(r_storage, r_shape, r_stride)
    return Q, R
