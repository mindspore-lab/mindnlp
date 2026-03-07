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


def _cast_tensor_dtype(a, dst_dtype):
    if a.dtype == dst_dtype:
        return a
    if not aclnn.cast_symbols_ok():
        raise RuntimeError("aclnnCast symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(dst_dtype), runtime=runtime)
    aclnn.cast(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dst_dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), dst_dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)



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
    size = int(size)
    shape = (size,)

    if npu_runtime.soc_profile() == "310b":
        from .creation import empty_create, ones_create

        if size == 0:
            return empty_create(shape, dtype=int64_dtype, device=device)
        ones = ones_create(shape, dtype=int64_dtype, device=device)
        return sub(cumsum(ones, dim=0), 1)

    runtime = npu_runtime.get_runtime((device.index or 0))
    stream = npu_state.current_stream((device.index or 0))
    if not aclnn.arange_symbols_ok():
        raise RuntimeError("aclnnArange symbols not available")
    stride = npu_runtime._contiguous_stride(shape)
    out_size = _numel(shape) * _dtype_itemsize(int64_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.arange(0, size, 1, out_ptr, shape, stride, int64_dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(out_ptr, _numel(shape), int64_dtype, device=device)
    return _wrap_tensor(storage, shape, stride)


def _is_310b_profile():
    return npu_runtime.soc_profile() == "310b"


_310B_FALLBACK_OPS = {
    "atan2",
    "where",
    "flip",
    "argsort",
    "sort",
    "topk",
    "diag",
    "lerp",
    "remainder",
    "isclose",
    "softplus",
    "uniform_",
    "normal_",
    "layer_norm",
    "mish",
    "batch_norm",
    "dropout",
    "take_along_dim",
    "gather",
}


def _use_310b_fallback(op_name):
    return _is_310b_profile() and op_name in _310B_FALLBACK_OPS


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


# Bitwise operations
def bitwise_not(a):
    if not aclnn.bitwise_not_symbols_ok():
        raise RuntimeError("aclnnBitwiseNot symbols not available")
    return _unary_op(a, aclnn.bitwise_not, "bitwise_not")


def bitwise_and(a, b):
    if not aclnn.bitwise_and_symbols_ok():
        raise RuntimeError("aclnnBitwiseAndTensor symbols not available")
    return _binary_op(a, b, aclnn.bitwise_and, "bitwise_and")


def bitwise_or(a, b):
    if not aclnn.bitwise_or_symbols_ok():
        raise RuntimeError("aclnnBitwiseOrTensor symbols not available")
    return _binary_op(a, b, aclnn.bitwise_or, "bitwise_or")


def bitwise_xor(a, b):
    if not aclnn.bitwise_xor_symbols_ok():
        raise RuntimeError("aclnnBitwiseXorTensor symbols not available")
    return _binary_op(a, b, aclnn.bitwise_xor, "bitwise_xor")


def le(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.le_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def lt(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.lt_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def gt(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.gt_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def ge(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.ge_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


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


def dot(a, b):
    """Dot product of two 1D tensors."""
    if not aclnn.dot_symbols_ok():
        raise RuntimeError("aclnnDot symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU dot expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU dot requires matching dtypes")
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("NPU dot expects 1D tensors")
    if a.shape[0] != b.shape[0]:
        raise ValueError("NPU dot requires tensors of same length")

    itemsize = _dtype_itemsize(a.dtype)
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    a_ptr = int(a_storage.data_ptr()) + int(a.offset * itemsize)
    b_ptr = int(b_storage.data_ptr()) + int(b.offset * itemsize)

    # Output is a 0-dim scalar tensor
    out_shape = ()
    out_stride = ()
    out_ptr = npu_runtime._alloc_device(itemsize, runtime=runtime)

    aclnn.dot(
        a_ptr, b_ptr, out_ptr,
        a.shape, a.stride,
        b.shape, b.stride,
        out_shape, out_stride,
        a.dtype, runtime, stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, 1, a.dtype, device=a.device)
    return _wrap_tensor(storage, out_shape, out_stride)


def mv(a, b):
    """Matrix-vector multiplication."""
    if not aclnn.mv_symbols_ok():
        raise RuntimeError("aclnnMv symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU mv expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU mv requires matching dtypes")
    if len(a.shape) != 2:
        raise ValueError("NPU mv expects 2D matrix as first argument")
    if len(b.shape) != 1:
        raise ValueError("NPU mv expects 1D vector as second argument")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"NPU mv: matrix columns ({a.shape[1]}) != vector length ({b.shape[0]})")

    itemsize = _dtype_itemsize(a.dtype)
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    a_ptr = int(a_storage.data_ptr()) + int(a.offset * itemsize)
    b_ptr = int(b_storage.data_ptr()) + int(b.offset * itemsize)

    out_shape = (a.shape[0],)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(out_shape[0] * itemsize, runtime=runtime)

    # cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) for Ascend910B
    aclnn.mv(
        a_ptr, b_ptr, out_ptr,
        a.shape, a.stride,
        b.shape, b.stride,
        out_shape, out_stride,
        a.dtype, 1, runtime, stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, out_shape[0], a.dtype, device=a.device)
    return _wrap_tensor(storage, out_shape, out_stride)


def outer(a, b):
    """Outer product of two 1D tensors (ger)."""
    if not aclnn.ger_symbols_ok():
        raise RuntimeError("aclnnGer symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU outer expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU outer requires matching dtypes")
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("NPU outer expects 1D tensors")

    itemsize = _dtype_itemsize(a.dtype)
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    a_ptr = int(a_storage.data_ptr()) + int(a.offset * itemsize)
    b_ptr = int(b_storage.data_ptr()) + int(b.offset * itemsize)

    out_shape = (a.shape[0], b.shape[0])
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(out_shape[0] * out_shape[1] * itemsize, runtime=runtime)

    aclnn.ger(
        a_ptr, b_ptr, out_ptr,
        a.shape, a.stride,
        b.shape, b.stride,
        out_shape, out_stride,
        a.dtype, runtime, stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, out_shape[0] * out_shape[1], a.dtype, device=a.device)
    return _wrap_tensor(storage, out_shape, out_stride)


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


def square(a):
    if aclnn.square_symbols_ok():
        try:
            return _unary_op(a, aclnn.square, "square")
        except RuntimeError:
            pass
    return mul(a, a)


def isposinf(a):
    if aclnn.isposinf_symbols_ok():
        try:
            return _unary_op(a, aclnn.isposinf, "isposinf", out_dtype=bool_dtype)
        except RuntimeError:
            pass
    return logical_and(isinf(a), gt(a, _scalar_to_npu_tensor(0, a)))


def isneginf(a):
    if aclnn.isneginf_symbols_ok():
        try:
            return _unary_op(a, aclnn.isneginf, "isneginf", out_dtype=bool_dtype)
        except RuntimeError:
            pass
    return logical_and(isinf(a), lt(a, _scalar_to_npu_tensor(0, a)))


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


def median(a, dim=None, keepdim=False):
    """Median along a dimension or global median."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU median expects NPU tensors")

    storage = _unwrap_storage(a)
    itemsize = _dtype_itemsize(a.dtype)

    if dim is None:
        # Global median - returns scalar
        if not aclnn.median_symbols_ok():
            raise RuntimeError("aclnnMedian symbols not available")
        out_shape = (1,)
        out_stride = (1,)
        out_ptr = npu_runtime._alloc_device(itemsize, runtime=runtime)

        aclnn.median(
            storage.data_ptr(),
            out_ptr,
            a.shape, a.stride, a.dtype,
            out_shape, out_stride,
            runtime, stream=stream.stream,
        )

        out_storage = npu_typed_storage_from_ptr(out_ptr, 1, a.dtype, device=a.device)
        # Return as scalar (reshape from (1,) to ())
        from ..common import view as view_backend
        result = _wrap_tensor(out_storage, out_shape, out_stride)
        return view_backend.reshape(result, ())

    # Median along a dimension
    if not aclnn.median_dim_symbols_ok():
        raise RuntimeError("aclnnMedianDim symbols not available")

    if dim < 0:
        dim += len(a.shape)

    out_shape = _reduce_out_shape(a.shape, [dim], keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = max(_numel(out_shape), 1)

    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)
    idx_ptr = npu_runtime._alloc_device(out_numel * _dtype_itemsize(int64_dtype), runtime=runtime)

    aclnn.median_dim(
        storage.data_ptr(),
        out_ptr,
        idx_ptr,
        a.shape, a.stride, a.dtype,
        out_shape, out_stride,
        dim, keepdim,
        runtime, stream=stream.stream,
    )

    runtime.defer_free(idx_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def kthvalue(a, k, dim=None, keepdim=False):
    """K-th smallest element along a dimension."""
    if not aclnn.kthvalue_symbols_ok():
        raise RuntimeError("aclnnKthvalue symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU kthvalue expects NPU tensors")

    storage = _unwrap_storage(a)
    itemsize = _dtype_itemsize(a.dtype)

    if dim is None:
        dim = 0
        from ..common import view as view_backend
        flat = view_backend.reshape(a, (_numel(a.shape),))
        if a.shape != flat.shape:
            return kthvalue(flat, k, dim=0, keepdim=False)

    if dim < 0:
        dim += len(a.shape)

    if k < 1 or k > a.shape[dim]:
        raise ValueError(f"k ({k}) out of range for dimension {dim} with size {a.shape[dim]}")

    out_shape = _reduce_out_shape(a.shape, [dim], keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = max(_numel(out_shape), 1)

    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)
    idx_ptr = npu_runtime._alloc_device(out_numel * _dtype_itemsize(int64_dtype), runtime=runtime)

    aclnn.kthvalue(
        storage.data_ptr(),
        out_ptr,
        idx_ptr,
        a.shape, a.stride, a.dtype,
        out_shape, out_stride,
        k, dim, keepdim,
        runtime, stream=stream.stream,
    )

    runtime.defer_free(idx_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def searchsorted(sorted_sequence, values, out_int32=False, right=False, side=None, sorter=None):
    """Find indices where elements should be inserted to maintain order."""
    if side is not None:
        right = (side == "right")
    if not aclnn.search_sorted_symbols_ok():
        raise RuntimeError("aclnnSearchSorted symbols not available")
    runtime = npu_runtime.get_runtime((sorted_sequence.device.index or 0))
    stream = npu_state.current_stream((sorted_sequence.device.index or 0))
    if sorted_sequence.device.type != "npu" or values.device.type != "npu":
        raise ValueError("NPU searchsorted expects NPU tensors")
    if sorted_sequence.dtype != values.dtype:
        raise ValueError("NPU searchsorted requires matching dtypes")

    sorted_storage = _unwrap_storage(sorted_sequence)
    values_storage = _unwrap_storage(values)

    out_dtype = "int32" if out_int32 else "int64"
    out_shape = tuple(values.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = max(_numel(out_shape), 1)
    out_ptr = npu_runtime._alloc_device(out_numel * _dtype_itemsize(out_dtype), runtime=runtime)

    aclnn.search_sorted(
        sorted_storage.data_ptr(),
        values_storage.data_ptr(),
        out_ptr,
        sorted_sequence.shape, sorted_sequence.stride,
        values.shape, values.stride,
        out_shape, out_stride,
        sorted_sequence.dtype,
        out_int32,
        right,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, out_dtype, device=sorted_sequence.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def unique(a, sorted=True, return_inverse=False, return_counts=False, dim=None):
    """Unique elements of a tensor."""
    if not aclnn.unique_symbols_ok():
        raise RuntimeError("aclnnUnique symbols not available")
    if dim is not None:
        raise NotImplementedError("NPU unique with dim argument not supported")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU unique expects NPU tensors")

    storage = _unwrap_storage(a)
    itemsize = _dtype_itemsize(a.dtype)
    numel = _numel(a.shape)

    # Output tensors - allocate same size as input (ACLNN will fill up to actual unique count)
    out_shape = (numel,)
    out_stride = (1,)
    out_ptr = npu_runtime._alloc_device(numel * itemsize, runtime=runtime)
    # inverse_indices always needed by ACLNN (even if not returned to user)
    inverse_shape = (numel,)
    inverse_stride = (1,)
    inverse_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize("int64"), runtime=runtime)

    aclnn.unique(
        storage.data_ptr(),
        out_ptr,
        inverse_ptr,
        a.shape, a.stride, a.dtype,
        out_shape, out_stride,
        inverse_shape, inverse_stride,
        sorted, return_inverse,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, numel, a.dtype, device=a.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)

    if return_inverse:
        inv_storage = npu_typed_storage_from_ptr(inverse_ptr, numel, "int64", device=a.device)
        inv = _wrap_tensor(inv_storage, inverse_shape, inverse_stride)
        if return_counts:
            return out, inv, None
        return out, inv
    else:
        runtime.defer_free(inverse_ptr)
        if return_counts:
            return out, None
        return out


def randperm(n, dtype=None, device=None, generator=None):
    """Random permutation of integers from 0 to n-1."""
    if not aclnn.randperm_symbols_ok():
        raise RuntimeError("aclnnRandperm symbols not available")
    # Import device handling
    from ..._device import device as Device
    if device is None:
        device = Device("npu:0")
    elif isinstance(device, str):
        device = Device(device)
    if device.type != "npu":
        raise ValueError("NPU randperm only supports NPU device")

    if dtype is None:
        dtype = "int64"
    runtime = npu_runtime.get_runtime((device.index or 0))
    stream = npu_state.current_stream((device.index or 0))

    # Get deterministic seed
    if generator is not None and hasattr(generator, 'philox_engine_inputs'):
        seed, offset = generator.philox_engine_inputs(10)
    else:
        from ... import npu as npu_mod
        seed, offset = npu_mod._get_and_advance_offset(device_index=(device.index or 0), increment=10)

    itemsize = _dtype_itemsize(dtype)
    out_ptr = npu_runtime._alloc_device(n * itemsize, runtime=runtime)

    aclnn.randperm(n, out_ptr, dtype, runtime, stream=stream.stream, seed=seed, offset=offset)

    out_storage = npu_typed_storage_from_ptr(out_ptr, n, dtype, device=device)
    return _wrap_tensor(out_storage, (n,), (1,))


def flatten_op(a, start_dim=0, end_dim=-1):
    """Flatten tensor dimensions using reshape."""
    from ..common import view as view_backend
    ndim = len(a.shape)
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim
    if start_dim == end_dim:
        return a
    if start_dim > end_dim:
        raise ValueError(f"flatten: start_dim ({start_dim}) > end_dim ({end_dim})")
    flat_size = 1
    for i in range(start_dim, end_dim + 1):
        flat_size *= a.shape[i]
    new_shape = a.shape[:start_dim] + (flat_size,) + a.shape[end_dim+1:]
    return view_backend.reshape(a, new_shape)


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


def expm1(a):
    if not aclnn.expm1_symbols_ok():
        raise RuntimeError("aclnnExpm1 symbols not available")
    return _unary_op(a, aclnn.expm1, "expm1")


def log1p(a):
    if not aclnn.log1p_symbols_ok():
        raise RuntimeError("aclnnLog1p symbols not available")
    return _unary_op(a, aclnn.log1p, "log1p")


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
    if _use_310b_fallback("atan2"):
        z = div(a, b)
        out = atan(z)

        zero = _scalar_to_npu_tensor(0, out)
        pi = _scalar_to_npu_tensor(3.141592653589793, out)
        pi_half = _scalar_to_npu_tensor(1.5707963267948966, out)

        x_lt0 = lt(b, zero)
        x_eq0 = eq(b, zero)
        y_ge0 = ge(a, zero)
        y_gt0 = gt(a, zero)
        y_lt0 = lt(a, zero)
        y_eq0 = eq(a, zero)

        out = where(logical_and(x_lt0, y_ge0), add(out, pi), out)
        out = where(logical_and(x_lt0, y_lt0), sub(out, pi), out)
        out = where(logical_and(x_eq0, y_gt0), pi_half, out)
        out = where(logical_and(x_eq0, y_lt0), neg(pi_half), out)
        out = where(logical_and(x_eq0, y_eq0), zero, out)
        return out

    return _binary_op(a, b, aclnn.atan2, "atan2")


def min_(a, b):
    result = _binary_op(a, b, aclnn.minimum, "min")
    nan_mask = logical_or(isnan(a), isnan(b))
    return where(nan_mask, add(a, b), result)


def max_(a, b):
    result = _binary_op(a, b, aclnn.maximum, "max")
    nan_mask = logical_or(isnan(a), isnan(b))
    return where(nan_mask, add(a, b), result)


def maximum(a, b):
    """Element-wise maximum of two tensors."""
    return _binary_op(a, b, aclnn.maximum, "maximum")


def minimum(a, b):
    """Element-wise minimum of two tensors."""
    return _binary_op(a, b, aclnn.minimum, "minimum")


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

    out_shape = _broadcast_shape(cond.shape, x.shape)
    out_shape = _broadcast_shape(out_shape, y.shape)
    if out_shape != x.shape:
        x = _npu_broadcast_to(x, out_shape)
    if out_shape != y.shape:
        y = _npu_broadcast_to(y, out_shape)
    if out_shape != cond.shape:
        cond = _npu_broadcast_to(cond, out_shape)

    if _use_310b_fallback("where"):
        out = contiguous(y)
        idx = nonzero(cond, as_tuple=True)
        if len(idx) == 0 or idx[0].numel() == 0:
            return out
        vals = masked_select(x, cond)
        return index_put_(out, idx, vals, accumulate=False)

    if not aclnn.s_where_symbols_ok():
        raise RuntimeError("aclnnSWhere symbols not available")

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


def _flip_310b_fallback(a, dims):
    out = a
    for d in dims:
        size = int(out.shape[d])
        if size <= 1:
            continue
        parts = split(out, 1, dim=d)
        out = cat(tuple(reversed(parts)), dim=d)
    return out


def _topk_310b_fill_value(dtype, largest):
    name = getattr(dtype, "name", None) or str(dtype).split(".")[-1]
    if name in ("float16", "float32", "float64", "bfloat16"):
        return -float("inf") if largest else float("inf")
    if name == "int8":
        return -128 if largest else 127
    if name == "uint8":
        return 0 if largest else 255
    if name == "int16":
        return -32768 if largest else 32767
    if name == "int32":
        return -2147483648 if largest else 2147483647
    if name == "int64":
        return -9223372036854775808 if largest else 9223372036854775807
    raise RuntimeError(f"NPU topk 310B fallback does not support dtype {dtype}")


def _topk_310b_fallback(a, k, dim, largest, sorted_flag):
    from .creation import empty_create

    out_shape = list(a.shape)
    out_shape[dim] = int(k)
    out_shape = tuple(out_shape)

    if int(k) == 0:
        values = empty_create(out_shape, dtype=a.dtype, device=a.device)
        indices = empty_create(out_shape, dtype=int64_dtype, device=a.device)
        return values, indices

    work = a
    values_parts = []
    indices_parts = []
    fill_value = _topk_310b_fill_value(a.dtype, largest)

    for _ in range(int(k)):
        if largest:
            idx = argmax(work, dim=dim, keepdim=True)
            val = amax(work, dim=dim, keepdim=True)
        else:
            idx = argmin(work, dim=dim, keepdim=True)
            val = amin(work, dim=dim, keepdim=True)
        values_parts.append(val)
        indices_parts.append(idx)
        work = scatter(work, dim, idx, fill_value)

    if len(values_parts) == 1:
        values = values_parts[0]
        indices = indices_parts[0]
    else:
        values = cat(values_parts, dim=dim)
        indices = cat(indices_parts, dim=dim)

    if not bool(sorted_flag):
        return values, indices
    return values, indices


def flip(a, dims):
    if a.device.type != "npu":
        raise ValueError("NPU flip expects NPU tensors")
    dims = _normalize_dims_tuple(dims, a.dim(), "flip")
    if len(dims) == 0:
        return a

    if _use_310b_fallback("flip"):
        return _flip_310b_fallback(a, dims)

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

    if _use_310b_fallback("argsort"):
        _, indices = _topk_310b_fallback(a, k=a.shape[dim], dim=dim, largest=bool(descending), sorted_flag=True)
        return indices

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

    if _use_310b_fallback("sort"):
        return _topk_310b_fallback(a, k=a.shape[dim], dim=dim, largest=bool(descending), sorted_flag=True)

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

    if _use_310b_fallback("topk"):
        return _topk_310b_fallback(a, k=k, dim=dim, largest=bool(largest), sorted_flag=bool(sorted))

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


def _diag_310b_fallback(a, diagonal=0):
    from .creation import empty_create, zeros_create

    diagonal = int(diagonal)

    if a.dim() == 1:
        n = int(a.shape[0])
        size = n + (diagonal if diagonal >= 0 else -diagonal)
        out = zeros_create((size, size), dtype=a.dtype, device=a.device)
        if n == 0:
            return out

        idx = _npu_arange_1d(n, a.device)
        if diagonal >= 0:
            rows = idx
            cols = idx if diagonal == 0 else add(idx, diagonal)
        else:
            rows = add(idx, -diagonal)
            cols = idx
        return index_put_(out, (rows, cols), a, accumulate=False)

    m = int(a.shape[0])
    n = int(a.shape[1])
    if diagonal >= 0:
        length = max(0, min(m, n - diagonal))
    else:
        length = max(0, min(m + diagonal, n))

    if length == 0:
        return empty_create((0,), dtype=a.dtype, device=a.device)

    idx = _npu_arange_1d(length, a.device)
    if diagonal >= 0:
        rows = idx
        cols = idx if diagonal == 0 else add(idx, diagonal)
    else:
        rows = add(idx, -diagonal)
        cols = idx

    linear = add(mul(rows, n), cols)
    flat = view_backend.reshape(a, (a.numel(),))
    return take(flat, linear)


def diag(a, diagonal=0):
    if a.device.type != "npu":
        raise ValueError("NPU diag expects NPU tensors")
    if a.dim() not in (1, 2):
        raise ValueError("diag expects 1D or 2D tensor")

    if _use_310b_fallback("diag"):
        return _diag_310b_fallback(a, diagonal=diagonal)

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
    if _use_310b_fallback("lerp"):
        # Static small-op fallback on 310B to avoid aclnnLerp 561103.
        delta = sub(b, a)
        scaled = mul(delta, weight)
        return add(a, scaled)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    if hasattr(weight, "shape"):
        # Tensor weight path
        w_storage = _unwrap_storage(weight)
        out_shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), weight.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.lerp_tensor(
            a_storage.data_ptr(), b_storage.data_ptr(), w_storage.data_ptr(), out_ptr,
            a.shape, a.stride, b.shape, b.stride,
            weight.shape, weight.stride, out_shape, out_stride,
            a.dtype, runtime, stream=stream.stream,
        )
    else:
        # Scalar weight path
        out_shape = _broadcast_shape(a.shape, b.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.lerp_scalar(
            a_storage.data_ptr(), b_storage.data_ptr(), out_ptr,
            a.shape, a.stride, b.shape, b.stride,
            out_shape, out_stride, a.dtype, float(weight),
            runtime, stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def addcmul(a, b, c, value=1.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    c_storage = _unwrap_storage(c)
    out_shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), c.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    if hasattr(value, "shape"):
        value = float(_to_numpy(value))
    aclnn.addcmul(
        a_storage.data_ptr(), b_storage.data_ptr(), c_storage.data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        c.shape, c.stride, out_shape, out_stride,
        a.dtype, float(value), runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def addcdiv(a, b, c, value=1.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    c_storage = _unwrap_storage(c)
    out_shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), c.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    if hasattr(value, "shape"):
        value = float(_to_numpy(value))
    aclnn.addcdiv(
        a_storage.data_ptr(), b_storage.data_ptr(), c_storage.data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        c.shape, c.stride, out_shape, out_stride,
        a.dtype, float(value), runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def logaddexp(a, b):
    return _binary_op(a, b, aclnn.slogaddexp, "logaddexp")


def logaddexp2(a, b):
    return _binary_op(a, b, aclnn.slogaddexp2, "logaddexp2")


def hypot(a, b):
    return sqrt(add(mul(a, a), mul(b, b)))


def _remainder_310b_fallback(a, b):
    # torch-style remainder keeps the sign of divisor b.
    r = fmod(a, b)
    zero = _scalar_to_npu_tensor(0, r)
    nz = ne(r, zero)
    r_neg = lt(r, zero)
    b_neg = lt(b, zero)
    mismatch = ne(r_neg, b_neg)
    fix = logical_and(nz, mismatch)
    return where(fix, add(r, b), r)


def remainder(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _use_310b_fallback("remainder"):
        return _remainder_310b_fallback(a, b)
    return _binary_op(a, b, aclnn.sremainder, "remainder")


def fmod(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.sfmod, "fmod")


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
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)

    if _use_310b_fallback("isclose"):
        diff = abs(sub(a, b))
        tol = add(_scalar_to_npu_tensor(float(atol), diff), mul(_scalar_to_npu_tensor(float(rtol), diff), abs(b)))
        close = le(diff, tol)
        if equal_nan:
            nan_both = logical_and(isnan(a), isnan(b))
            close = logical_or(close, nan_both)
        else:
            nan_any = logical_or(isnan(a), isnan(b))
            close = logical_and(close, logical_not(nan_any))
        return close

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.sisclose(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
        out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype,
        float(rtol), float(atol), True,  # ACLNN ignores equal_nan, always pass True
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    result = _wrap_tensor(out_storage, out_shape, out_stride)
    if not equal_nan:
        # ACLNN always treats NaN==NaN as True; mask out when equal_nan=False
        nan_both = logical_and(isnan(a), isnan(b))
        result = logical_and(result, logical_not(nan_both))
    return result


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

    if _use_310b_fallback("softplus"):
        beta = float(beta)
        threshold = float(threshold)
        bx = mul(a, beta)
        base = add(relu(bx), log(add(exp(neg(abs(bx))), 1)))
        out = div(base, beta)
        if threshold > 0:
            thr = _scalar_to_npu_tensor(threshold, bx)
            mask = gt(bx, thr)
            out = where(mask, a, out)
        return out

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

    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        dim = None

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


def uniform_(a, low=0.0, high=1.0, generator=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU uniform_ expects NPU tensors")

    if _use_310b_fallback("uniform_"):
        from ... import npu as npu_mod

        if generator is not None and hasattr(generator, 'philox_engine_inputs'):
            seed, offset = generator.philox_engine_inputs(10)
        else:
            seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

        idx = _cast_tensor_dtype(_npu_arange_1d(_numel(a.shape), a.device), float_dtype)
        idx = add(idx, float(seed + offset))
        u = sin(add(mul(idx, 12.9898), float(seed) * 78.233))
        u = frac(abs(mul(u, 43758.5453)))
        u = reshape(u, a.shape)

        scale = float(high) - float(low)
        if scale != 1.0:
            u = mul(u, scale)
        if float(low) != 0.0:
            u = add(u, float(low))

        if a.dtype != float_dtype:
            u = _cast_tensor_dtype(u, a.dtype)
        return copy_(a, u)

    if generator is not None and hasattr(generator, 'philox_engine_inputs'):
        seed, offset = generator.philox_engine_inputs(10)
    else:
        from ... import npu as npu_mod
        seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

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


def normal_(a, mean=0.0, std=1.0, generator=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU normal_ expects NPU tensors")

    if _use_310b_fallback("normal_"):
        # Deterministic hash-based normal approximation on NPU to avoid unstable ACLNN random ops.
        from ... import npu as npu_mod

        if generator is not None and hasattr(generator, 'philox_engine_inputs'):
            seed, offset = generator.philox_engine_inputs(10)
        else:
            seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

        idx = _cast_tensor_dtype(_npu_arange_1d(_numel(a.shape), a.device), float_dtype)
        idx = add(idx, float(seed + offset))

        # u = frac(abs(sin(idx * 12.9898 + seed * 78.233) * 43758.5453))
        u = sin(add(mul(idx, 12.9898), float(seed) * 78.233))
        u = frac(abs(mul(u, 43758.5453)))

        # map to roughly standard normal via centered scaling
        z = mul(sub(u, 0.5), 3.4641016151377544)
        z = reshape(z, a.shape)

        if float(std) != 1.0:
            z = mul(z, float(std))
        if float(mean) != 0.0:
            z = add(z, float(mean))
        if a.dtype != float_dtype:
            z = _cast_tensor_dtype(z, a.dtype)
        return copy_(a, z)

    if generator is not None and hasattr(generator, 'philox_engine_inputs'):
        seed, offset = generator.philox_engine_inputs(10)
    else:
        from ... import npu as npu_mod
        seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

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


def randint_(a, low, high=None, generator=None):
    """In-place randint — fills tensor with random integers from [low, high)."""
    if high is None:
        low, high = 0, low
    # Fill with uniform [low, high), then floor to get integers
    uniform_(a, float(low), float(high), generator=generator)
    # In-place floor
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.floor(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def random_(a, from_=0, to=None, generator=None):
    """In-place random — fills tensor with random values from [from_, to)."""
    import numpy as np
    from ..._dtype import to_numpy_dtype
    np_dtype = to_numpy_dtype(a.dtype)
    if to is None:
        if np.issubdtype(np_dtype, np.floating):
            to = 2**24 if np_dtype == np.float32 else 2**53
        else:
            to = int(np.iinfo(np_dtype).max) + 1
    # Fill with uniform [from_, to), then floor
    uniform_(a, float(from_), float(to), generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.floor(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def bernoulli_(a, p=0.5, generator=None):
    """In-place Bernoulli — fills tensor with 0/1 from Bernoulli(p)."""
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    numel = _numel(a.shape)
    if hasattr(p, 'storage'):
        p_storage = _unwrap_storage(p)
        p_shape, p_stride = p.shape, p.stride
    else:
        p_tensor = _scalar_to_npu_tensor(float(p), a)
        p_storage = _unwrap_storage(p_tensor)
        p_shape, p_stride = p_tensor.shape, p_tensor.stride
    bool_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize("bool"), runtime=runtime)
    aclnn.lt(a_storage.data_ptr(), p_storage.data_ptr(), bool_ptr,
             a.shape, a.stride, p_shape, p_stride, a.shape, a.stride,
             a.dtype, runtime, stream=stream.stream)
    aclnn.cast(bool_ptr, a_storage.data_ptr(), a.shape, a.stride, "bool", a.dtype, runtime, stream=stream.stream)
    runtime.defer_free(bool_ptr)
    return a


def exponential_(a, lambd=1.0, generator=None):
    """In-place exponential — fills with samples from Exp(lambd)."""
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.log(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    aclnn.neg(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    if lambd != 1.0:
        scale = _scalar_to_npu_tensor(1.0 / lambd, a)
        scale_storage = _unwrap_storage(scale)
        numel = _numel(a.shape)
        tmp_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
        aclnn.mul(a_storage.data_ptr(), scale_storage.data_ptr(), tmp_ptr,
                  a.shape, a.stride, scale.shape, scale.stride, a.shape, a.stride,
                  a.dtype, runtime, stream=stream.stream)
        aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
        runtime.defer_free(tmp_ptr)
    return a


def log_normal_(a, mean=1.0, std=2.0, generator=None):
    """In-place log-normal — fills with exp(N(mean, std))."""
    normal_(a, mean, std, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.exp(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def cauchy_(a, median=0.0, sigma=1.0, generator=None):
    """In-place Cauchy — fills with median + sigma * tan(pi * (U - 0.5))."""
    import math
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    numel = _numel(a.shape)
    # sub 0.5
    aclnn.sub_scalar(a_storage.data_ptr(), 0.5, a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    # mul pi
    pi_tensor = _scalar_to_npu_tensor(math.pi, a)
    pi_storage = _unwrap_storage(pi_tensor)
    tmp_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.mul(a_storage.data_ptr(), pi_storage.data_ptr(), tmp_ptr,
              a.shape, a.stride, pi_tensor.shape, pi_tensor.stride, a.shape, a.stride,
              a.dtype, runtime, stream=stream.stream)
    aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    runtime.defer_free(tmp_ptr)
    # tan in-place
    aclnn.tan(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    # mul sigma
    if sigma != 1.0:
        sigma_tensor = _scalar_to_npu_tensor(sigma, a)
        sigma_storage = _unwrap_storage(sigma_tensor)
        tmp_ptr2 = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
        aclnn.mul(a_storage.data_ptr(), sigma_storage.data_ptr(), tmp_ptr2,
                  a.shape, a.stride, sigma_tensor.shape, sigma_tensor.stride, a.shape, a.stride,
                  a.dtype, runtime, stream=stream.stream)
        aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr2, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
        runtime.defer_free(tmp_ptr2)
    # add median
    if median != 0.0:
        aclnn.add_scalar(a_storage.data_ptr(), median, a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    return a


def geometric_(a, p, generator=None):
    """In-place geometric — fills with ceil(ln(U) / ln(1-p))."""
    import math
    uniform_(a, 0.0, 1.0, generator=generator)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    aclnn.log(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    # divide by log(1-p)
    log_1_minus_p = math.log(1.0 - float(p))
    divisor = _scalar_to_npu_tensor(log_1_minus_p, a)
    divisor_storage = _unwrap_storage(divisor)
    numel = _numel(a.shape)
    tmp_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.div(a_storage.data_ptr(), divisor_storage.data_ptr(), tmp_ptr,
              a.shape, a.stride, divisor.shape, divisor.stride, a.shape, a.stride,
              a.dtype, runtime, stream=stream.stream)
    aclnn.inplace_copy(a_storage.data_ptr(), tmp_ptr, a.shape, a.stride, a.dtype, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    runtime.defer_free(tmp_ptr)
    # ceil in-place
    aclnn.ceil(a_storage.data_ptr(), a_storage.data_ptr(), a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
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


def div_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU div_ expects NPU tensors")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.div(
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
    """NPU tensor indexing — full support for basic and advanced indexing."""
    if not isinstance(key, tuple):
        key = (key,)

    if _is_basic_index_key(key):
        view = _npu_basic_getitem_view(tensor, key)
        if view is not None:
            return view

    return _npu_advanced_getitem(tensor, key)


def setitem(tensor, key, value):
    """NPU tensor index assignment — full support for basic and advanced indexing."""
    if not isinstance(key, tuple):
        key = (key,)

    if _is_basic_index_key(key):
        view = _npu_basic_getitem_view(tensor, key)
        if view is not None:
            _npu_assign_to_view(view, value)
            return tensor

    _npu_advanced_setitem(tensor, key, value)
    return tensor


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

def _is_int_index(key):
    """True for integer indices (not bool). Handles numpy.integer too."""
    import numpy as np
    return isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))


def _is_basic_index_key(keys):
    """True when *keys* (a tuple) contains only int/slice/None/Ellipsis/bool."""
    import numpy as np
    for item in keys:
        if item is Ellipsis or item is None:
            continue
        if isinstance(item, slice):
            continue
        if _is_int_index(item):
            continue
        # Python bool / numpy.bool_ treated as basic index
        # (True → unsqueeze+keep, False → unsqueeze+empty)
        if isinstance(item, (bool, np.bool_)):
            continue
        return False
    return True


def _expand_ellipsis(keys, ndim):
    """Expand Ellipsis into the right number of ``slice(None)``."""
    import numpy as np
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 0:
        return list(keys)

    # Count dims that consume real tensor dimensions
    # None, Ellipsis, and bool don't consume tensor dims
    specified_dims = 0
    for item in keys:
        if item is None or item is Ellipsis:
            continue
        if isinstance(item, (bool, np.bool_)):
            continue
        specified_dims += 1
    fill = ndim - specified_dims
    if fill < 0:
        raise IndexError("too many indices for tensor")

    expanded = []
    for item in keys:
        if item is Ellipsis:
            expanded.extend([slice(None)] * fill)
        else:
            expanded.append(item)
    return expanded


def _npu_basic_getitem_view(tensor, key):
    """Create a view for basic indexing (int, slice, None, Ellipsis, bool).

    Returns a Tensor sharing the same storage, or None if we need to fall back
    to a copy (e.g. negative-step slices that require aclnnSlice).
    """
    from ..._tensor import Tensor
    import numpy as np

    keys = list(key) if isinstance(key, tuple) else [key]
    keys = _expand_ellipsis(keys, tensor.dim())

    in_dim = 0
    out_shape = []
    out_stride = []
    out_offset = tensor.offset

    needs_aclnn_slice = False

    for item in keys:
        if item is None:
            out_shape.append(1)
            if in_dim < tensor.dim():
                out_stride.append(tensor.stride[in_dim] * tensor.shape[in_dim])
            else:
                out_stride.append(1)
            continue

        # Python bool / np.bool_: True → unsqueeze (size 1), False → empty dim (size 0)
        # Does NOT consume a tensor dimension (same as None).
        if isinstance(item, (bool, np.bool_)):
            if item:
                out_shape.append(1)
            else:
                out_shape.append(0)
            if in_dim < tensor.dim():
                out_stride.append(tensor.stride[in_dim] * tensor.shape[in_dim])
            else:
                out_stride.append(1)
            continue

        if in_dim >= tensor.dim():
            raise IndexError("too many indices for tensor")

        dim_size = tensor.shape[in_dim]
        dim_stride = tensor.stride[in_dim]

        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += dim_size
            if idx < 0 or idx >= dim_size:
                raise IndexError(
                    f"index {item} is out of bounds for dimension {in_dim} with size {dim_size}"
                )
            out_offset += idx * dim_stride
            in_dim += 1
            continue

        if isinstance(item, slice):
            start, stop, step = item.indices(dim_size)
            if step < 0:
                # Negative step requires data reversal — fall back to aclnnSlice
                needs_aclnn_slice = True
                break
            length = len(range(start, stop, step))
            out_offset += start * dim_stride
            out_shape.append(length)
            out_stride.append(dim_stride * step)
            in_dim += 1
            continue

        # Non-basic element — shouldn't reach here due to _is_basic_index_key check
        return None

    if needs_aclnn_slice:
        return _npu_basic_getitem_with_strided_slices(tensor, keys)

    # Append remaining dims
    while in_dim < tensor.dim():
        out_shape.append(tensor.shape[in_dim])
        out_stride.append(tensor.stride[in_dim])
        in_dim += 1

    out_shape = tuple(out_shape)
    out_stride = tuple(out_stride)

    return Tensor(tensor.storage(), out_shape, out_stride, out_offset)


def _npu_basic_getitem_with_strided_slices(tensor, keys):
    """Handle basic indexing when one or more slices have step != 1.

    Process left-to-right: step==1 slices and ints become view ops;
    step!=1 slices use aclnnSlice which produces a contiguous copy.
    """
    from ..._tensor import Tensor

    cur = tensor
    in_dim = 0
    pending_none_count = 0

    for item in keys:
        if item is None:
            pending_none_count += 1
            continue

        if in_dim >= cur.dim():
            raise IndexError("too many indices for tensor")

        # Insert pending None (unsqueeze) dimensions before this real dim
        for _ in range(pending_none_count):
            cur = _npu_unsqueeze_view(cur, in_dim)
            in_dim += 1
        pending_none_count = 0

        dim_size = cur.shape[in_dim]

        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += dim_size
            cur = _npu_select_view(cur, in_dim, idx)
            # select removes the dim, so in_dim stays the same
            continue

        if isinstance(item, slice):
            start, stop, step = item.indices(dim_size)
            if step > 0:
                # Positive step: strided view (no copy)
                cur = _npu_strided_slice_view(cur, in_dim, start, stop, step)
            else:
                # Negative step: requires data reversal, use aclnnSlice kernel
                cur = _npu_aclnn_slice(cur, in_dim, start, stop, step)
            in_dim += 1
            continue

    # Insert any trailing None dims
    for _ in range(pending_none_count):
        cur = _npu_unsqueeze_view(cur, cur.dim())

    return cur


def _npu_select_view(tensor, dim, idx):
    """Select a single element along *dim* — returns a view with dim removed."""
    from ..._tensor import Tensor
    new_offset = tensor.offset + idx * tensor.stride[dim]
    new_shape = tensor.shape[:dim] + tensor.shape[dim + 1:]
    new_stride = tensor.stride[:dim] + tensor.stride[dim + 1:]
    return Tensor(tensor.storage(), new_shape, new_stride, new_offset)


def _npu_slice_view(tensor, dim, start, stop):
    """Step-1 slice as a view — adjust offset and shape[dim]."""
    from ..._tensor import Tensor
    length = max(0, stop - start)
    new_offset = tensor.offset + start * tensor.stride[dim]
    new_shape = tensor.shape[:dim] + (length,) + tensor.shape[dim + 1:]
    new_stride = tensor.stride  # stride unchanged for step==1
    return Tensor(tensor.storage(), new_shape, new_stride, new_offset)


def _npu_strided_slice_view(tensor, dim, start, stop, step):
    """Strided slice as a view — adjust offset, shape, and stride. step must be > 0."""
    from ..._tensor import Tensor
    length = len(range(start, stop, step))
    new_offset = tensor.offset + start * tensor.stride[dim]
    new_shape = tensor.shape[:dim] + (length,) + tensor.shape[dim + 1:]
    new_stride = tensor.stride[:dim] + (tensor.stride[dim] * step,) + tensor.stride[dim + 1:]
    return Tensor(tensor.storage(), new_shape, new_stride, new_offset)


def _npu_unsqueeze_view(tensor, dim):
    """Insert a size-1 dimension at *dim* — pure view."""
    from ..._tensor import Tensor
    new_shape = tensor.shape[:dim] + (1,) + tensor.shape[dim:]
    # Compute a stride that keeps the tensor contiguous-looking
    if dim < len(tensor.stride):
        new_s = tensor.stride[dim] * tensor.shape[dim]
    elif len(tensor.stride) > 0:
        new_s = 1
    else:
        new_s = 1
    new_stride = tensor.stride[:dim] + (new_s,) + tensor.stride[dim:]
    return Tensor(tensor.storage(), new_shape, new_stride, tensor.offset)


def _npu_aclnn_slice(tensor, dim, start, stop, step):
    """Strided slice via aclnnSlice kernel — returns a new contiguous tensor."""
    length = len(range(start, stop, step))
    out_shape = tensor.shape[:dim] + (length,) + tensor.shape[dim + 1:]
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(tensor.dtype)
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))

    if out_numel == 0:
        out_stride = npu_runtime._contiguous_stride(out_shape) if out_shape else ()
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), tensor.dtype, device=tensor.device)
        return _wrap_tensor(storage, out_shape, out_stride)

    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)

    # Compute the data pointer including any offset from prior view ops
    src_ptr = int(_unwrap_storage(tensor).data_ptr()) + tensor.offset * itemsize

    aclnn.slice_op(
        src_ptr,
        tensor.shape,
        tensor.stride,
        tensor.dtype,
        dim,
        start,
        stop,
        step,
        out_ptr,
        out_shape,
        out_stride,
        tensor.dtype,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, tensor.dtype, device=tensor.device)
    return _wrap_tensor(storage, out_shape, out_stride)


def _npu_data_ptr(tensor):
    """Return the effective data pointer for *tensor* (base + offset)."""
    itemsize = _dtype_itemsize(tensor.dtype)
    return int(_unwrap_storage(tensor).data_ptr()) + tensor.offset * itemsize


def _npu_assign_to_view(view, value):
    """Write *value* into a view tensor (which shares storage with the original).

    For contiguous views, use D2D memcpy.  Otherwise, use aclnnInplaceCopy.
    """
    runtime = npu_runtime.get_runtime((view.device.index or 0))
    stream = npu_state.current_stream((view.device.index or 0))
    itemsize = _dtype_itemsize(view.dtype)

    if isinstance(value, (int, float)):
        # Create a filled tensor matching the view shape, then copy
        from .creation import zeros_create
        temp = zeros_create(view.shape, dtype=view.dtype, device=view.device)
        temp = _scalar_to_npu_tensor(value, temp)
        value = temp

    if hasattr(value, 'storage'):
        if view.is_contiguous() and value.is_contiguous() and view.shape == value.shape:
            dst_ptr = _npu_data_ptr(view)
            numel = view.numel()
            copy_size = numel * itemsize
            if value.device.type != "npu":
                src_ptr = value.storage().data_ptr()
                ret = npu_runtime.acl.rt.memcpy(dst_ptr, copy_size, src_ptr, copy_size, 1)  # H2D
            else:
                src_ptr = _npu_data_ptr(value)
                ret = npu_runtime.acl.rt.memcpy(dst_ptr, copy_size, src_ptr, copy_size, 3)  # D2D
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy failed: {ret}")
        else:
            # Non-contiguous: use aclnnInplaceCopy
            dst_ptr = _npu_data_ptr(view)
            if value.device.type != "npu":
                # Move value to NPU first
                from .creation import tensor_create
                import numpy as np
                value = tensor_create(value._numpy_view().copy(), dtype=value.dtype, device=view.device)
            src_ptr = _npu_data_ptr(value)
            aclnn.inplace_copy(
                dst_ptr,
                src_ptr,
                view.shape,
                view.stride,
                view.dtype,
                value.shape,
                value.stride,
                value.dtype,
                runtime,
                stream=stream.stream,
            )
    else:
        raise TypeError(f"Cannot assign {type(value)} to NPU tensor view")


# ---------------------------------------------------------------------------
# Advanced indexing (Tensor, bool mask, list, mixed)
# ---------------------------------------------------------------------------

def _is_advanced_index(item):
    """True if *item* is a Tensor, list, or other advanced index."""
    from ..._tensor import Tensor
    if isinstance(item, Tensor):
        return True
    if isinstance(item, (list, tuple)):
        # A list/tuple of numbers is advanced indexing
        return True
    return False


def _to_npu_index_tensor(key, device, dtype_hint=None):
    """Convert a Python int/list/Tensor to an NPU int64 tensor for indexing."""
    from ..._tensor import Tensor
    from .creation import tensor_create
    import numpy as np

    if isinstance(key, Tensor):
        if key.dtype.name == 'bool':
            # Bool tensor → nonzero indices
            return _expand_bool_tensor(key)
        if key.device.type == "npu":
            if key.dtype == int64_dtype:
                return key
            # Cast to int64
            return _cast_to_int64(key)
        # CPU tensor → move to NPU
        arr = key._numpy_view().copy()
        return tensor_create(arr.astype(np.int64), dtype=int64_dtype, device=device)

    if isinstance(key, (list, tuple)):
        arr = np.array(key, dtype=np.int64)
        return tensor_create(arr, dtype=int64_dtype, device=device)

    if isinstance(key, (int, np.integer)):
        arr = np.array([int(key)], dtype=np.int64)
        t = tensor_create(arr, dtype=int64_dtype, device=device)
        return reshape(t, ())

    if isinstance(key, (bool, np.bool_)):
        arr = np.array([int(key)], dtype=np.int64)
        t = tensor_create(arr, dtype=int64_dtype, device=device)
        return reshape(t, ())

    raise TypeError(f"Cannot convert {type(key)} to index tensor")


def _cast_to_int64(tensor):
    """Cast an NPU tensor to int64 dtype."""
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    out_numel = _numel(tensor.shape)
    out_ptr = npu_runtime._alloc_device(out_numel * 8, runtime=runtime)  # int64 = 8 bytes
    src_ptr = _npu_data_ptr(tensor)
    aclnn.cast(
        src_ptr,
        out_ptr,
        tensor.shape,
        tensor.stride,
        tensor.dtype,
        int64_dtype,
        runtime,
        stream=stream.stream,
    )
    out_stride = npu_runtime._contiguous_stride(tensor.shape)
    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, int64_dtype, device=tensor.device)
    return _wrap_tensor(storage, tensor.shape, out_stride)


def _expand_bool_tensor(mask):
    """Convert a bool mask tensor to a tuple of int64 index tensors via nonzero."""
    result = nonzero(mask, as_tuple=True)
    return result


def _compute_broadcast_shape(shapes):
    """Broadcast multiple shapes together following NumPy rules."""
    if not shapes:
        return ()
    result = list(shapes[0])
    for shape in shapes[1:]:
        if len(shape) > len(result):
            result = [1] * (len(shape) - len(result)) + result
        elif len(result) > len(shape):
            shape = (1,) * (len(result) - len(shape)) + tuple(shape)
        new_result = []
        for a, b in zip(result, shape):
            if a == 1:
                new_result.append(b)
            elif b == 1:
                new_result.append(a)
            elif a == b:
                new_result.append(a)
            else:
                raise ValueError(f"Cannot broadcast shapes")
        result = new_result
    return tuple(result)


def _npu_advanced_getitem(tensor, key):
    """Full getitem supporting mixed basic + advanced indexing.

    Phase 1: Process basic indices (int, slice, None, Ellipsis) via views.
    Phase 2: Process advanced indices (Tensor, list, bool) via aclnnIndex.
    """
    from ..._tensor import Tensor

    keys = list(key) if isinstance(key, tuple) else [key]

    # Step 1: Expand bool Tensor indices BEFORE expanding Ellipsis.
    # A bool tensor of N dims consumes N real dims, and we need the correct
    # dim count for Ellipsis expansion.
    expanded_keys = []
    for item in keys:
        if isinstance(item, Tensor) and item.dtype.name == 'bool':
            nz_indices = nonzero(item, as_tuple=True)
            for idx_t in nz_indices:
                expanded_keys.append(idx_t)
        else:
            expanded_keys.append(item)
    keys = expanded_keys

    # Step 2: Expand Ellipsis (now with the correct real dim count)
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 1:
        keys = _expand_ellipsis(keys, tensor.dim())

    # Check if there are any remaining advanced indices
    has_advanced = any(isinstance(item, (Tensor, list)) for item in keys)

    if not has_advanced:
        # All basic — use view path (with potential aclnnSlice)
        return _npu_basic_getitem_with_strided_slices(tensor, keys)

    # Pad keys to ndim with slice(None)
    ndim = tensor.dim()
    real_keys = [k for k in keys if k is not None]
    while len(real_keys) < ndim:
        real_keys.append(slice(None))
        keys.append(slice(None))

    # Separate real-dim actions from None (newaxis) positions
    dim_idx = 0
    dim_actions = []  # (position_in_dim_actions, key_item)
    none_positions = []

    pos = 0
    for item in keys:
        if item is None:
            none_positions.append(pos)
            pos += 1
            continue
        dim_actions.append((dim_idx, item))
        dim_idx += 1
        pos += 1

    # Find which positions in dim_actions have advanced indices
    adv_dims = [i for i, (d, item) in enumerate(dim_actions) if isinstance(item, (Tensor, list))]

    if not adv_dims:
        return _npu_basic_getitem_with_strided_slices(tensor, keys)

    # Pre-apply basic indices on non-advanced dims (high → low to avoid shift)
    prepared = tensor
    dim_remap = list(range(len(dim_actions)))

    for i in range(len(dim_actions) - 1, -1, -1):
        d_orig, item = dim_actions[i]
        if i in adv_dims:
            continue
        cur_dim = dim_remap[i]
        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += prepared.shape[cur_dim]
            prepared = _npu_select_view(prepared, cur_dim, idx)
            for j in range(len(dim_remap)):
                if dim_remap[j] > cur_dim:
                    dim_remap[j] -= 1
            dim_remap[i] = -1  # removed
        elif isinstance(item, slice):
            start, stop, step = item.indices(prepared.shape[cur_dim])
            if step == 1:
                prepared = _npu_slice_view(prepared, cur_dim, start, stop)
            else:
                prepared = _npu_aclnn_slice(prepared, cur_dim, start, stop, step)

    # Build advanced index tensors and their current dim positions
    adv_index_tensors = []
    adv_current_dims = []
    for i in adv_dims:
        cur_dim = dim_remap[i]
        if cur_dim < 0:
            continue
        adv_current_dims.append(cur_dim)
        idx_tensor = _to_npu_index_tensor(dim_actions[i][1], prepared.device)
        adv_index_tensors.append(idx_tensor)

    if not adv_index_tensors:
        result = prepared
        for pos in none_positions:
            result = _npu_unsqueeze_view(result, pos)
        return result

    # Broadcast all advanced index tensors
    idx_shapes = [t.shape for t in adv_index_tensors]
    broadcast_shape = _compute_broadcast_shape(idx_shapes)

    expanded_idx_tensors = []
    for t in adv_index_tensors:
        if t.shape != broadcast_shape:
            t = _npu_expand(t, broadcast_shape)
        expanded_idx_tensors.append(t)

    # Build entries list for aclnnIndex (None for dims not indexed)
    entries = [None] * prepared.dim()
    for dim_pos, idx_t in zip(adv_current_dims, expanded_idx_tensors):
        entries[dim_pos] = (
            _npu_data_ptr(idx_t),
            idx_t.shape,
            idx_t.stride,
            idx_t.dtype,
        )

    # Compute output shape following PyTorch advanced indexing rules:
    # - If advanced dims are contiguous: broadcast_shape replaces them in-place
    # - If advanced dims are non-contiguous: broadcast_shape goes to the front
    adv_dim_positions = sorted(adv_current_dims)
    are_contiguous = all(
        adv_dim_positions[j] == adv_dim_positions[j - 1] + 1
        for j in range(1, len(adv_dim_positions))
    )

    out_shape_parts = []
    if are_contiguous:
        adv_inserted = False
        for i in range(prepared.dim()):
            if entries[i] is not None:
                if not adv_inserted:
                    out_shape_parts.extend(broadcast_shape)
                    adv_inserted = True
            else:
                out_shape_parts.append(prepared.shape[i])
    else:
        # Non-contiguous: broadcast shape goes to the front
        out_shape_parts.extend(broadcast_shape)
        for i in range(prepared.dim()):
            if entries[i] is None:
                out_shape_parts.append(prepared.shape[i])

    out_shape = tuple(out_shape_parts)
    out_numel = _numel(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    itemsize = _dtype_itemsize(prepared.dtype)

    runtime = npu_runtime.get_runtime((prepared.device.index or 0))
    stream = npu_state.current_stream((prepared.device.index or 0))

    if out_numel == 0:
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), prepared.dtype, device=prepared.device)
        return _wrap_tensor(storage, out_shape, out_stride)

    out_ptr = npu_runtime._alloc_device(out_numel * itemsize, runtime=runtime)
    src_ptr = _npu_data_ptr(prepared)

    aclnn.index(
        src_ptr,
        prepared.shape,
        prepared.stride,
        prepared.dtype,
        entries,
        out_ptr,
        out_shape,
        out_stride,
        prepared.dtype,
        runtime,
        stream=stream.stream,
    )

    storage = npu_typed_storage_from_ptr(out_ptr, out_numel, prepared.dtype, device=prepared.device)
    result = _wrap_tensor(storage, out_shape, out_stride)

    # Apply None insertions
    for pos in none_positions:
        result = _npu_unsqueeze_view(result, pos)

    return result


def _npu_expand(tensor, target_shape):
    """Expand tensor to target shape (broadcast — no data copy, just stride manipulation)."""
    from ..._tensor import Tensor

    src_shape = tensor.shape
    src_stride = tensor.stride
    ndiff = len(target_shape) - len(src_shape)

    # Pad shape/stride on the left with 1s/0s
    padded_shape = (1,) * ndiff + src_shape
    padded_stride = (0,) * ndiff + src_stride

    new_stride = []
    for i, (ts, ps, pst) in enumerate(zip(target_shape, padded_shape, padded_stride)):
        if ps == ts:
            new_stride.append(pst)
        elif ps == 1:
            new_stride.append(0)
        else:
            raise RuntimeError(f"Cannot expand dim {i} from {ps} to {ts}")

    return Tensor(tensor.storage(), tuple(target_shape), tuple(new_stride), tensor.offset)


def _npu_advanced_setitem(tensor, key, value):
    """Full setitem for advanced indexing using aclnnIndexPutImpl."""
    from ..._tensor import Tensor
    import numpy as np

    keys = list(key) if isinstance(key, tuple) else [key]

    # Step 1: Expand bool tensors BEFORE Ellipsis (same reason as getitem)
    expanded_keys = []
    for item in keys:
        if isinstance(item, Tensor) and item.dtype.name == 'bool':
            nz_indices = nonzero(item, as_tuple=True)
            for idx_t in nz_indices:
                expanded_keys.append(idx_t)
        else:
            expanded_keys.append(item)
    keys = expanded_keys

    # Step 2: Expand Ellipsis
    ellipsis_count = sum(1 for item in keys if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if ellipsis_count == 1:
        keys = _expand_ellipsis(keys, tensor.dim())

    # Remove None entries (newaxis doesn't apply to setitem destination)
    keys = [k for k in keys if k is not None]

    # Pad to ndim
    while len(keys) < tensor.dim():
        keys.append(slice(None))

    # Separate basic and advanced indices
    # For setitem, we apply basic slices as views on the target tensor,
    # then use index_put_impl for the advanced indices.

    prepared = tensor
    adv_dims = []

    dim_remap = list(range(len(keys)))

    for i, item in enumerate(keys):
        if isinstance(item, (Tensor, list)):
            adv_dims.append(i)

    if not adv_dims:
        # All basic — use view + assign
        view = _npu_basic_getitem_with_strided_slices(prepared, keys)
        if view is not None:
            _npu_assign_to_view(view, value)
            return

    # Apply basic slices on non-advanced dims (from high to low)
    dim_actions = list(enumerate(keys))
    for i in range(len(dim_actions) - 1, -1, -1):
        orig_i, item = dim_actions[i]
        if i in adv_dims:
            continue
        cur_dim = dim_remap[i]
        if _is_int_index(item):
            idx = int(item)
            if idx < 0:
                idx += prepared.shape[cur_dim]
            prepared = _npu_select_view(prepared, cur_dim, idx)
            for j in range(len(dim_remap)):
                if dim_remap[j] > cur_dim:
                    dim_remap[j] -= 1
            dim_remap[i] = -1
        elif isinstance(item, slice):
            start, stop, step = item.indices(prepared.shape[cur_dim])
            if step == 1:
                prepared = _npu_slice_view(prepared, cur_dim, start, stop)
            else:
                # For setitem with step!=1, we can't slice as view.
                # Keep the full dim and let index_put_impl handle it.
                # Convert slice to an index tensor
                import numpy as np
                from .creation import tensor_create
                indices = list(range(start, stop, step))
                idx_t = tensor_create(np.array(indices, dtype=np.int64), dtype=int64_dtype, device=prepared.device)
                adv_dims.append(i)
                keys[i] = idx_t

    # Build index tensors for the advanced dims
    adv_index_tensors = []
    for i in adv_dims:
        cur_dim = dim_remap[i]
        if cur_dim < 0:
            continue
        item = keys[i]
        idx_tensor = _to_npu_index_tensor(item, prepared.device)
        if isinstance(idx_tensor, tuple):
            for t in idx_tensor:
                adv_index_tensors.append(t)
        else:
            adv_index_tensors.append(idx_tensor)

    if not adv_index_tensors:
        return

    # Prepare value tensor
    if isinstance(value, (int, float)):
        from .creation import tensor_create
        import numpy as np
        val_arr = np.full((1,), value, dtype=npu_runtime._dtype_to_numpy(prepared.dtype))
        value_tensor = tensor_create(val_arr, dtype=prepared.dtype, device=prepared.device)
    elif hasattr(value, 'storage'):
        value_tensor = value
        if value_tensor.device.type != "npu":
            from .creation import tensor_create
            import numpy as np
            value_tensor = tensor_create(
                value_tensor._numpy_view().copy(),
                dtype=value_tensor.dtype,
                device=prepared.device,
            )
    else:
        from .creation import tensor_create
        import numpy as np
        value_tensor = tensor_create(
            np.array(value, dtype=npu_runtime._dtype_to_numpy(prepared.dtype)),
            dtype=prepared.dtype,
            device=prepared.device,
        )

    runtime = npu_runtime.get_runtime((prepared.device.index or 0))
    stream = npu_state.current_stream((prepared.device.index or 0))

    index_ptrs = [_npu_data_ptr(t) for t in adv_index_tensors]
    index_shapes = [t.shape for t in adv_index_tensors]
    index_strides = [t.stride for t in adv_index_tensors]
    index_dtypes = [t.dtype for t in adv_index_tensors]

    aclnn.index_put_impl(
        _npu_data_ptr(prepared),
        prepared.shape,
        prepared.stride,
        prepared.dtype,
        index_ptrs,
        index_shapes,
        index_strides,
        index_dtypes,
        _npu_data_ptr(value_tensor),
        value_tensor.shape,
        value_tensor.stride,
        value_tensor.dtype,
        False,  # accumulate
        False,  # unsafe
        runtime,
        stream=stream.stream,
    )


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



def _layer_norm_310b_fallback(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    n_norm = len(normalized_shape)
    if n_norm == 0:
        return input

    axis_dims = tuple(range(input.dim() - n_norm, input.dim()))
    lead = input.dim() - n_norm
    stats_shape = (1,) * lead + tuple(normalized_shape)

    x = input if input.dtype == float_dtype else _cast_tensor_dtype(input, float_dtype)
    mean_t = mean(x, dim=axis_dims, keepdim=True)
    diff = sub(x, mean_t)
    var = mean(mul(diff, diff), dim=axis_dims, keepdim=True)
    eps_t = _scalar_to_npu_tensor(float(eps), var)
    inv_std = rsqrt(add(var, eps_t))
    out = mul(diff, inv_std)

    if weight is not None:
        w = weight if weight.dtype == float_dtype else _cast_tensor_dtype(weight, float_dtype)
        w = reshape(w, stats_shape)
        out = mul(out, w)
    if bias is not None:
        b = bias if bias.dtype == float_dtype else _cast_tensor_dtype(bias, float_dtype)
        b = reshape(b, stats_shape)
        out = add(out, b)

    if input.dtype != float_dtype:
        out = _cast_tensor_dtype(out, input.dtype)
    return out



def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Compute layer normalization using aclnnLayerNorm."""
    if _use_310b_fallback("layer_norm"):
        return _layer_norm_310b_fallback(input, normalized_shape, weight=weight, bias=bias, eps=eps)

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
    # Allocate mean/rstd for backward pass (layer_norm backward needs them)
    stats_numel_val = max(stats_numel, 1)
    float_dtype = input.dtype  # same dtype for stats
    mean_ptr = npu_runtime._alloc_device(stats_numel_val * 4, runtime=runtime)  # float32
    rstd_ptr = npu_runtime._alloc_device(stats_numel_val * 4, runtime=runtime)  # float32
    # Wrap in Storage to prevent early deallocation
    mean_storage = npu_typed_storage_from_ptr(mean_ptr, stats_numel_val, float_dtype, device=input.device)
    rstd_storage = npu_typed_storage_from_ptr(rstd_ptr, stats_numel_val, float_dtype, device=input.device)

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

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    # Attach mean/rstd for backward pass
    out._backward_data = {
        "mean_ptr": mean_ptr, "rstd_ptr": rstd_ptr,
        "mean_storage": mean_storage, "rstd_storage": rstd_storage,
        "stats_shape": stats_shape, "stats_stride": stats_stride,
        "normalized_shape": tuple(normalized_shape),
    }
    return out


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
    if _use_310b_fallback("mish"):
        return mul(a, tanh(softplus(a)))
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




def _batch_norm_310b_fallback(input, running_mean, running_var, weight=None, bias=None,
                               training=False, momentum=0.1, eps=1e-5):
    if input.dim() < 2:
        raise ValueError("batch_norm expects input with at least 2 dims")

    C = int(input.shape[1])
    stats_shape = (1, C) + (1,) * (input.dim() - 2)

    if training or running_mean is None or running_var is None:
        dims = [0] + list(range(2, input.dim()))
        mean_t = mean(input, dim=dims, keepdim=True)
        diff = sub(input, mean_t)
        var_t = mean(mul(diff, diff), dim=dims, keepdim=True)

        if running_mean is not None:
            mean_reshaped = reshape(mean_t, (C,))
            new_rm = add(mul(running_mean, (1.0 - float(momentum))), mul(mean_reshaped, float(momentum)))
            copy_(running_mean, new_rm)
        if running_var is not None:
            var_reshaped = reshape(var_t, (C,))
            new_rv = add(mul(running_var, (1.0 - float(momentum))), mul(var_reshaped, float(momentum)))
            copy_(running_var, new_rv)
    else:
        mean_t = reshape(running_mean, stats_shape)
        var_t = reshape(running_var, stats_shape)

    eps_t = _scalar_to_npu_tensor(float(eps), mean_t)
    denom = sqrt(add(var_t, eps_t))
    out = div(sub(input, mean_t), denom)

    if weight is not None:
        w = reshape(weight, stats_shape)
        out = mul(out, w)
    if bias is not None:
        b = reshape(bias, stats_shape)
        out = add(out, b)
    return out


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """Compute batch normalization using aclnnBatchNorm."""
    if _use_310b_fallback("batch_norm"):
        return _batch_norm_310b_fallback(input, running_mean, running_var, weight=weight, bias=bias,
                                         training=training, momentum=momentum, eps=eps)

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

    # Allocate save_mean/save_invstd externally for backward pass
    C = input.shape[1] if len(input.shape) >= 2 else 1
    save_mean_ptr = npu_runtime._alloc_device(C * 4, runtime=runtime)
    save_invstd_ptr = npu_runtime._alloc_device(C * 4, runtime=runtime)
    # Wrap in Storage to prevent GC
    save_mean_storage = npu_typed_storage_from_ptr(save_mean_ptr, C, input.dtype, device=input.device)
    save_invstd_storage = npu_typed_storage_from_ptr(save_invstd_ptr, C, input.dtype, device=input.device)

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
        runtime, stream=stream.stream,
        ext_save_mean_ptr=save_mean_ptr,
        ext_save_invstd_ptr=save_invstd_ptr,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {
        "save_mean_ptr": save_mean_ptr, "save_invstd_ptr": save_invstd_ptr,
        "save_mean_storage": save_mean_storage, "save_invstd_storage": save_invstd_storage,
        "C": C, "training": training, "eps": eps,
    }
    return out


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


def _dropout_310b_mask(a, keep_prob):
    from .creation import empty_create
    from ... import npu as npu_mod

    numel = _numel(a.shape)
    if numel == 0:
        return empty_create(a.shape, dtype=bool_dtype, device=a.device)

    idx = _npu_arange_1d(numel, a.device)
    idx_f = _cast_tensor_dtype(idx, float_dtype)

    seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)
    seed_t = _scalar_to_npu_tensor(float(seed + offset), idx_f)

    val = sin(add(mul(idx_f, 12.9898), mul(seed_t, 78.233)))
    val = abs(mul(val, 43758.5453))
    val = frac(val)
    val = reshape(val, a.shape)

    keep_t = _scalar_to_npu_tensor(float(keep_prob), val)
    return lt(val, keep_t)



def dropout(a, p=0.5, training=True):
    """Compute dropout using aclnnDropoutGenMask + aclnnDropoutDoMask."""
    if not training or p == 0:
        return a

    if _use_310b_fallback("dropout"):
        if p >= 1:
            from .creation import zeros_create
            return zeros_create(a.shape, dtype=a.dtype, device=a.device)
        if not getattr(a.dtype, "is_floating_point", True):
            raise ValueError("NPU dropout expects floating-point tensors")
        keep_prob = 1.0 - float(p)
        keep = _dropout_310b_mask(a, keep_prob)
        out = where(keep, a, 0)
        return mul(out, 1.0 / keep_prob)

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
    seed, offset = npu_mod._get_and_advance_offset(device_index=(a.device.index or 0), increment=10)

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

    # Save mask for backward (dropout backward reuses the same mask)
    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {"mask_ptr": mask_ptr, "mask_numel": mask_numel, "p": p}
    return out


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

    # 310B static path: avoid SWhere by converting mask to int64 and blending arithmetically.
    if _use_310b_fallback("take_along_dim"):
        if not aclnn.cast_symbols_ok():
            raise RuntimeError("aclnnCast symbols not available")
        runtime = npu_runtime.get_runtime((indices.device.index or 0))
        stream = npu_state.current_stream((indices.device.index or 0))

        shape = tuple(indices.shape)
        stride = tuple(indices.stride)
        numel = max(_numel(shape), 1)

        mask_i64_ptr = npu_runtime._alloc_device(numel * _dtype_itemsize(int64_dtype), runtime=runtime)
        aclnn.cast(
            _unwrap_storage(neg_mask).data_ptr(),
            mask_i64_ptr,
            shape,
            stride,
            bool_dtype,
            int64_dtype,
            runtime,
            stream=stream.stream,
        )
        mask_i64_storage = npu_typed_storage_from_ptr(mask_i64_ptr, numel, int64_dtype, device=indices.device)
        mask_i64 = _wrap_tensor(mask_i64_storage, shape, stride)

        offset = _scalar_to_npu_tensor(int(dim_size), indices)
        return add(indices, mul(mask_i64, offset))

    return where(neg_mask, add(indices, _scalar_to_npu_tensor(int(dim_size), indices)), indices)


def _move_dim_to_last(a, dim):
    dim = _normalize_dim(dim, a.dim())
    out = a
    for i in range(dim, a.dim() - 1):
        out = view_backend.transpose(out, i, i + 1)
    return out


def _gather_310b_fallback(a, dim, index):
    from .creation import ones_create, zeros_create

    dim = _normalize_dim(dim, a.dim())
    dim_size = int(a.shape[dim])
    flat_idx = view_backend.reshape(index, (index.numel(),))
    n = int(flat_idx.shape[0])

    # Build one-hot(index) on NPU via scatter to avoid aclnnGather.
    base = zeros_create((n, dim_size), dtype=a.dtype, device=a.device)
    idx2d = view_backend.reshape(flat_idx, (n, 1))
    src = ones_create((n, 1), dtype=a.dtype, device=a.device)
    one_hot_2d = scatter(base, 1, idx2d, src)
    one_hot = view_backend.reshape(one_hot_2d, tuple(index.shape) + (dim_size,))

    # Move gather dim to last and broadcast over index dim.
    moved = _move_dim_to_last(a, dim)
    moved_shape = list(moved.shape)
    moved_shape.insert(dim, 1)
    moved = view_backend.reshape(moved, tuple(moved_shape))
    moved = _npu_broadcast_to(moved, one_hot.shape)

    weighted = mul(one_hot, moved)
    return sum_(weighted, dim=weighted.dim() - 1, keepdim=False)
def gather(a, dim, index):
    dim = _normalize_dim(dim, a.dim())
    _require_int64_indices(index, "gather")
    if index.dim() != a.dim():
        raise ValueError("index shape mismatch")
    for i, size in enumerate(index.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("index shape mismatch")
    _validate_index_bounds(index, a.shape[dim], allow_negative=False, name="gather")

    if _use_310b_fallback("gather"):
        return _gather_310b_fallback(a, dim, index)

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



# ---------------------------------------------------------------------------
# Tensor indexing / selection ops
# ---------------------------------------------------------------------------

def narrow(a, dim, start, length):
    """narrow — returns a view of the tensor narrowed along *dim*."""
    from ..._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    new_shape = a.shape[:d] + (int(length),) + a.shape[d + 1:]
    new_offset = a.offset + int(start) * a.stride[d]
    return Tensor(a.storage(), new_shape, a.stride, new_offset)


def select(a, dim, index):
    """select — returns a view with *dim* removed at *index*."""
    return _npu_select_view(a, dim if dim >= 0 else dim + a.dim(),
                            int(index) if index >= 0 else int(index) + a.shape[dim if dim >= 0 else dim + a.dim()])


def expand(a, sizes):
    """expand — broadcast-expand a tensor (view, no copy)."""
    from ..._tensor import Tensor
    sizes = tuple(sizes)
    ndiff = len(sizes) - a.dim()
    if ndiff < 0:
        raise RuntimeError("expand: number of sizes must be >= tensor dim")
    src_shape = (1,) * ndiff + a.shape
    src_stride = (0,) * ndiff + a.stride
    out_shape = []
    out_stride = []
    for i, sz in enumerate(sizes):
        if sz == -1:
            out_shape.append(src_shape[i])
            out_stride.append(src_stride[i])
        elif src_shape[i] == 1:
            out_shape.append(sz)
            out_stride.append(0)
        elif src_shape[i] == sz:
            out_shape.append(sz)
            out_stride.append(src_stride[i])
        else:
            raise RuntimeError(
                f"expand: size {sz} not compatible with dim size {src_shape[i]}"
            )
    return Tensor(a.storage(), tuple(out_shape), tuple(out_stride), a.offset)


def masked_fill(a, mask, value):
    """masked_fill — out-of-place masked fill (returns a copy)."""
    from ..._dispatch.dispatcher import dispatch
    result = dispatch("clone", a.device.type, a)
    return masked_fill_(result, mask, value)


def masked_fill_(a, mask, value):
    """masked_fill_ — in-place masked fill with scalar value."""
    if not aclnn.masked_fill_scalar_symbols_ok():
        raise RuntimeError("aclnnInplaceMaskedFillScalar symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_masked_fill_scalar(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        _npu_data_ptr(mask), mask.shape, mask.stride, mask.dtype,
        value, runtime, stream=stream.stream,
    )
    return a


def index_put_(a, indices, values, accumulate=False):
    """index_put_ — in-place index put using list of index tensors."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    idx_ptrs = [_npu_data_ptr(t) for t in indices]
    idx_shapes = [t.shape for t in indices]
    idx_strides = [t.stride for t in indices]
    idx_dtypes = [t.dtype for t in indices]

    if hasattr(values, 'storage'):
        val_ptr = _npu_data_ptr(values)
        val_shape = values.shape
        val_stride = values.stride
        val_dtype = values.dtype
    else:
        # scalar
        from .creation import tensor_create
        import numpy as np
        val_t = tensor_create(
            np.array(values, dtype=npu_runtime._dtype_to_numpy(a.dtype)),
            dtype=a.dtype, device=a.device,
        )
        val_ptr = _npu_data_ptr(val_t)
        val_shape = val_t.shape
        val_stride = val_t.stride
        val_dtype = val_t.dtype

    aclnn.index_put_impl(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        idx_ptrs, idx_shapes, idx_strides, idx_dtypes,
        val_ptr, val_shape, val_stride, val_dtype,
        accumulate, False,
        runtime, stream=stream.stream,
    )
    return a


def index_put(a, indices, values, accumulate=False):
    """index_put — out-of-place index put (returns a copy)."""
    from ..._dispatch.dispatcher import dispatch
    result = dispatch("clone", a.device.type, a)
    return index_put_(result, indices, values, accumulate)


def index_copy_(a, dim, index, source):
    """index_copy_ — in-place copy along dim using index tensor."""
    if not aclnn.index_copy_symbols_ok():
        raise RuntimeError("aclnnInplaceIndexCopy symbols not available")
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_index_copy(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        runtime, stream=stream.stream,
    )
    return a


def index_fill_(a, dim, index, value):
    """index_fill_ — in-place fill along dim using index tensor with scalar value."""
    if not aclnn.index_fill_symbols_ok():
        raise RuntimeError("aclnnInplaceIndexFill symbols not available")
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_index_fill(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        value, runtime, stream=stream.stream,
    )
    return a


def index_add_(a, dim, index, source, alpha=1.0):
    """index_add_ — in-place add along dim using index tensor with alpha."""
    if not aclnn.index_add_symbols_ok():
        raise RuntimeError("aclnnIndexAdd symbols not available")
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    # aclnnIndexAdd is not in-place: self → out. Use self as both.
    aclnn.index_add(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        float(alpha),
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        runtime, stream=stream.stream,
    )
    return a


def scatter_(a, dim, index, src):
    """scatter_ — in-place scatter along dim."""
    d = dim if dim >= 0 else dim + a.dim()
    _require_int64_indices(index, "scatter_")

    if hasattr(src, "shape"):
        src_tensor = src
    else:
        src_tensor = _scalar_to_npu_tensor(src, a)

    if src_tensor.shape != index.shape:
        src_tensor = _npu_broadcast_to(src_tensor, index.shape)

    if not aclnn.scatter_symbols_ok():
        raise RuntimeError("aclnnScatter symbols not available")

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    # Use self as both input and output for in-place
    aclnn.scatter(
        _npu_data_ptr(a),
        _npu_data_ptr(index),
        _npu_data_ptr(src_tensor),
        _npu_data_ptr(a),
        a.shape, a.stride, a.dtype,
        index.shape, index.stride, index.dtype,
        src_tensor.shape, src_tensor.stride, src_tensor.dtype,
        d, 0,
        runtime, stream=stream.stream,
    )
    return a


def scatter_add_(a, dim, index, src):
    """scatter_add_ — in-place scatter add along dim."""
    d = dim if dim >= 0 else dim + a.dim()
    _require_int64_indices(index, "scatter_add_")

    if not aclnn.scatter_add_symbols_ok():
        raise RuntimeError("aclnnScatterAdd symbols not available")

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(max(_numel(a.shape), 1) * _dtype_itemsize(a.dtype), runtime=runtime)
    out_stride = npu_runtime._contiguous_stride(a.shape)

    aclnn.scatter_add_op(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(src), src.shape, src.stride, src.dtype,
        out_ptr, a.shape, out_stride, a.dtype,
        runtime, stream=stream.stream,
    )
    # Copy result back to a
    nbytes = max(_numel(a.shape), 1) * _dtype_itemsize(a.dtype)
    ret = npu_runtime.acl.rt.memcpy(_npu_data_ptr(a), nbytes, out_ptr, nbytes, 3)
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
    runtime.defer_free(out_ptr)
    return a


def masked_scatter_(a, mask, source):
    """masked_scatter_ — in-place masked scatter."""
    if not aclnn.masked_scatter_symbols_ok():
        raise RuntimeError("aclnnInplaceMaskedScatter symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_masked_scatter(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        _npu_data_ptr(mask), mask.shape, mask.stride, mask.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        runtime, stream=stream.stream,
    )
    return a


def unfold(a, dimension, size, step):
    """unfold — returns a view of the original tensor with an additional dimension of size *size*."""
    from ..._tensor import Tensor
    d = dimension if dimension >= 0 else dimension + a.dim()
    dim_size = a.shape[d]
    n_windows = max(0, (dim_size - size) // step + 1)

    new_shape = a.shape[:d] + (n_windows,) + a.shape[d + 1:] + (size,)
    new_stride = a.stride[:d] + (a.stride[d] * step,) + a.stride[d + 1:] + (a.stride[d],)
    return Tensor(a.storage(), new_shape, new_stride, a.offset)


# ---------------------------------------------------------------------------
# Tensor indexing / selection ops
# ---------------------------------------------------------------------------

def narrow(a, dim, start, length):
    """Narrow: return a view of tensor along dim from start to start+length."""
    from ..._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    return Tensor(a.storage(), tuple(new_shape), a.stride, new_offset)


def select(a, dim, index):
    """Select: remove dim by indexing a single element along it (view op)."""
    from ..._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    idx = int(index)
    if idx < 0:
        idx += a.shape[d]
    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    new_offset = a.offset + idx * a.stride[d]
    del new_stride[d]
    return Tensor(a.storage(), tuple(new_shape), tuple(new_stride), new_offset)


def expand(a, sizes):
    """Expand: broadcast tensor to larger sizes (view op, no copy)."""
    from ..._tensor import Tensor
    sizes = tuple(sizes)
    ndiff = len(sizes) - a.dim()
    if ndiff < 0:
        raise RuntimeError("expand: number of sizes must be >= tensor dim")
    src_shape = (1,) * ndiff + a.shape
    src_stride = (0,) * ndiff + a.stride
    out_shape = []
    out_stride = []
    for i, sz in enumerate(sizes):
        if sz == -1:
            out_shape.append(src_shape[i])
            out_stride.append(src_stride[i])
        elif src_shape[i] == 1:
            out_shape.append(sz)
            out_stride.append(0)
        elif src_shape[i] == sz:
            out_shape.append(sz)
            out_stride.append(src_stride[i])
        else:
            raise RuntimeError(
                f"expand: size {sz} not compatible with dim size {src_shape[i]}"
            )
    return Tensor(a.storage(), tuple(out_shape), tuple(out_stride), a.offset)


def masked_fill(a, mask, value):
    """Non-inplace masked fill — returns a copy with mask applied."""
    result = a.clone()
    masked_fill_(result, mask, value)
    return result


def masked_fill_(a, mask, value):
    """In-place masked fill using aclnnInplaceMaskedFillScalar."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_masked_fill_scalar(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        _npu_data_ptr(mask), mask.shape, mask.stride, mask.dtype,
        value, runtime, stream=stream.stream,
    )
    return a


def index_put_(a, indices, values, accumulate=False):
    """In-place index_put_ using aclnnIndexPutImpl."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    index_ptrs = [_npu_data_ptr(t) for t in indices]
    index_shapes = [t.shape for t in indices]
    index_strides = [t.stride for t in indices]
    index_dtypes = [t.dtype for t in indices]
    val_ptr = _npu_data_ptr(values)
    aclnn.index_put_impl(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        index_ptrs, index_shapes, index_strides, index_dtypes,
        val_ptr, values.shape, values.stride, values.dtype,
        accumulate, False, runtime, stream=stream.stream,
    )
    return a


def index_put(a, indices, values, accumulate=False):
    """Non-inplace index_put — returns a copy."""
    result = a.clone()
    index_put_(result, indices, values, accumulate)
    return result


def index_copy_(a, dim, index, source):
    """In-place index_copy_ using aclnnInplaceIndexCopy."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_index_copy(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        runtime, stream=stream.stream,
    )
    return a


def index_fill_(a, dim, index, value):
    """In-place index_fill_ using aclnnInplaceIndexFill."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_index_fill(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        value, runtime, stream=stream.stream,
    )
    return a


def index_add_(a, dim, index, source, alpha=1.0):
    """In-place index_add_ using aclnnIndexAdd (writes to self as out)."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.index_add(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        float(alpha),
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        runtime, stream=stream.stream,
    )
    return a


def scatter_(a, dim, index, src):
    """In-place scatter_ — delegates to existing scatter with self as out."""
    from ..._tensor import Tensor
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(src, Tensor):
        src_ptr = _npu_data_ptr(src)
        src_shape = src.shape
        src_stride = src.stride
        src_dtype = src.dtype
    else:
        # Scalar src — create a filled tensor
        from .creation import tensor_create
        import numpy as np
        src_arr = np.full(a.shape, src, dtype=npu_runtime._dtype_to_numpy(a.dtype))
        src_t = tensor_create(src_arr, dtype=a.dtype, device=a.device)
        src_ptr = _npu_data_ptr(src_t)
        src_shape = src_t.shape
        src_stride = src_t.stride
        src_dtype = src_t.dtype
    aclnn.scatter(
        _npu_data_ptr(a),
        _npu_data_ptr(a),
        a.shape, a.stride, a.dtype,
        index.shape, index.stride, index.dtype,
        src_shape, src_stride, src_dtype,
        d, 0, runtime, stream=stream.stream,
    )
    return a


def scatter_add_(a, dim, index, src):
    """In-place scatter_add_ using aclnnScatterAdd."""
    d = dim if dim >= 0 else dim + a.dim()
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.scatter_add_op(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        d,
        _npu_data_ptr(index), index.shape, index.stride, index.dtype,
        _npu_data_ptr(src), src.shape, src.stride, src.dtype,
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        runtime, stream=stream.stream,
    )
    return a


def masked_scatter_(a, mask, source):
    """In-place masked_scatter_ using aclnnInplaceMaskedScatter."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    aclnn.inplace_masked_scatter(
        _npu_data_ptr(a), a.shape, a.stride, a.dtype,
        _npu_data_ptr(mask), mask.shape, mask.stride, mask.dtype,
        _npu_data_ptr(source), source.shape, source.stride, source.dtype,
        runtime, stream=stream.stream,
    )
    return a


def unfold(a, dimension, size, step):
    """Unfold along a dimension — returns a higher-dimensional view/copy."""
    d = dimension if dimension >= 0 else dimension + a.dim()
    dim_size = a.shape[d]
    n_windows = max(0, (dim_size - size) // step + 1)
    if n_windows == 0:
        new_shape = list(a.shape)
        new_shape[d] = 0
        new_shape.append(size)
        out_shape = tuple(new_shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_numel = 0
        itemsize = _dtype_itemsize(a.dtype)
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        out_ptr = npu_runtime._alloc_device(max(itemsize, 1), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
        return _wrap_tensor(storage, out_shape, out_stride)
    # Build result by gathering slices
    # Each window is a[..., i*step:i*step+size, ...] along dim d
    # Output shape: a.shape with a.shape[d] replaced by n_windows, plus trailing `size`
    slices = []
    for i in range(n_windows):
        start = i * step
        # Use the existing view-based slice
        sliced = _npu_slice_view(a, d, start, start + size)
        slices.append(sliced)
    # Stack slices along dim d, then the window elements are along d+1
    # Actually, unfold should have shape [..., n_windows, ..., size] with size at end
    # The simplest correct approach: use contiguous + D2D copies
    out_shape = list(a.shape)
    out_shape[d] = n_windows
    out_shape.append(size)
    out_shape = tuple(out_shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(max(out_numel * itemsize, 1), runtime=runtime)
    storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
    result = _wrap_tensor(storage, out_shape, out_stride)
    # Fill each window slot
    for i in range(n_windows):
        start = i * step
        sliced = _npu_slice_view(a, d, start, start + size)
        # sliced has shape [..., size, ...] with size at dim d
        # We need to copy this into result[..., i, ..., :] where i is at dim d and : is at the end
        # Build destination view
        dst = _npu_select_view(result, d, i)
        # dst has shape [..., ...rest..., size] — the original dims except d, plus trailing size
        # sliced needs to be transposed: move dim d to the end
        # Use contiguous copy approach
        sliced_contig = sliced.contiguous()
        dst_shape_flat = _numel(dst.shape)
        src_ptr = _npu_data_ptr(sliced_contig)
        dst_ptr_val = _npu_data_ptr(dst)
        copy_bytes = dst_shape_flat * itemsize
        if copy_bytes > 0:
            npu_runtime.acl.rt.memcpy(dst_ptr_val, copy_bytes, src_ptr, copy_bytes, 3)
    return result


def var_(a, dim=None, unbiased=True, keepdim=False):
    """Compute variance using aclnnVar."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

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
    out_shape = tuple(out_shape) if out_shape else (1,)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.var(
        _unwrap_storage(a).data_ptr(), out_ptr,
        a.shape, a.stride, a.dtype,
        dims, unbiased, keepdim,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def norm_(a, p=2, dim=None, keepdim=False):
    """Compute tensor norm using aclnnNorm."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    from ..._dtype import float32 as f32
    out_dtype = a.dtype if getattr(a.dtype, 'is_floating_point', True) else f32

    if dim is None:
        norm_dims = list(range(len(a.shape)))
    elif isinstance(dim, int):
        norm_dims = [dim if dim >= 0 else dim + len(a.shape)]
    else:
        norm_dims = [d if d >= 0 else d + len(a.shape) for d in dim]

    out_shape = list(a.shape)
    for d in sorted(norm_dims, reverse=True):
        if keepdim:
            out_shape[d] = 1
        else:
            out_shape.pop(d)
    out_shape = tuple(out_shape) if out_shape else (1,)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(out_dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.norm(
        _unwrap_storage(a).data_ptr(), out_ptr,
        a.shape, a.stride, a.dtype,
        p, norm_dims, keepdim,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), out_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def prod_(a, dim=None, keepdim=False):
    """Compute product reduction using aclnnProd / aclnnProdDim."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    if dim is not None:
        d = dim if dim >= 0 else dim + len(a.shape)
        out_shape = list(a.shape)
        if keepdim:
            out_shape[d] = 1
        else:
            out_shape.pop(d)
        out_shape = tuple(out_shape) if out_shape else (1,)
    else:
        out_shape = (1,) if keepdim else (1,)

    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.prod(
        _unwrap_storage(a).data_ptr(), out_ptr,
        a.shape, a.stride, a.dtype,
        dim, keepdim,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def floor_divide(a, b):
    """Compute floor division using aclnnFloorDivide."""
    from ..._tensor import Tensor
    if not isinstance(b, Tensor):
        from ..._creation import tensor as _tensor
        b = _tensor(float(b), device=a.device)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    out_shape = tuple(_broadcast_shape_checked(a.shape, b.shape, "floor_divide"))
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.floor_divide(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
    """Compute RMS normalization using aclnnRmsNorm."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    norm_shape = tuple(normalized_shape)
    y_shape = input.shape
    y_stride = npu_runtime._contiguous_stride(y_shape)
    y_numel = _numel(y_shape)

    # rstd shape: input shape with normalized dims reduced to 1
    rstd_shape = list(input.shape)
    for i in range(len(norm_shape)):
        rstd_shape[-(i + 1)] = 1
    rstd_shape = tuple(rstd_shape)
    rstd_stride = npu_runtime._contiguous_stride(rstd_shape)
    rstd_numel = _numel(rstd_shape)

    itemsize = _dtype_itemsize(input.dtype)
    y_ptr = npu_runtime._alloc_device(max(y_numel, 1) * itemsize, runtime=runtime)
    rstd_ptr = npu_runtime._alloc_device(max(rstd_numel, 1) * itemsize, runtime=runtime)

    gamma_ptr = _unwrap_storage(weight).data_ptr() if weight is not None else None
    gamma_shape = weight.shape if weight is not None else ()
    gamma_stride = weight.stride if weight is not None else ()

    if gamma_ptr is None:
        # aclnnRmsNorm requires gamma; create ones tensor
        from ..._creation import ones as _ones
        w = _ones(norm_shape, dtype=input.dtype, device=input.device)
        gamma_ptr = _unwrap_storage(w).data_ptr()
        gamma_shape = w.shape
        gamma_stride = w.stride

    aclnn.rms_norm(
        _unwrap_storage(input).data_ptr(), gamma_ptr, eps, y_ptr, rstd_ptr,
        input.shape, input.stride, gamma_shape, gamma_stride,
        y_shape, y_stride, rstd_shape, rstd_stride,
        input.dtype,
        runtime, stream=stream.stream,
    )

    y_storage = npu_typed_storage_from_ptr(y_ptr, max(y_numel, 1), input.dtype, device=input.device)
    rstd_storage = npu_typed_storage_from_ptr(rstd_ptr, max(rstd_numel, 1), input.dtype, device=input.device)
    out = _wrap_tensor(y_storage, y_shape, y_stride)
    out._backward_data = {
        "rstd_ptr": rstd_ptr, "rstd_storage": rstd_storage,
        "rstd_shape": rstd_shape, "rstd_stride": rstd_stride,
        "normalized_shape": tuple(normalized_shape),
    }
    return out


def conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Conv2d forward using aclnnConvolution."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C_in, H, W = input.shape
    C_out, C_in_g, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    H_out = (H + 2 * pH - ekH) // sH + 1
    W_out = (W + 2 * pW - ekW) // sW + 1
    out_shape = (N, C_out, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    bias_ptr = None
    bias_shape = None
    bias_stride = None
    if bias is not None:
        bias_ptr = _unwrap_storage(bias).data_ptr()
        bias_shape = bias.shape
        bias_stride = bias.stride

    aclnn.convolution(
        _unwrap_storage(input).data_ptr(),
        _unwrap_storage(weight).data_ptr(),
        bias_ptr,
        input.shape, input.stride,
        weight.shape, weight.stride,
        bias_shape, bias_stride,
        input.dtype,
        stride, padding, dilation,
        False,  # transposed
        (0, 0),  # output_padding
        groups,
        out_ptr, out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def conv1d(input, weight, bias=None, stride=(1,), padding=(0,), dilation=(1,), groups=1):
    """Conv1d forward via conv2d with unsqueezed spatial dim."""
    from ..common import view as view_backend
    # Unsqueeze: (N, C, L) -> (N, C, 1, L)
    input_4d = view_backend.unsqueeze(input, 2)
    weight_4d = view_backend.unsqueeze(weight, 2)
    out_4d = conv2d(input_4d, weight_4d, bias,
                    stride=(1, stride[0]),
                    padding=(0, padding[0]),
                    dilation=(1, dilation[0]),
                    groups=groups)
    # Squeeze: (N, C_out, 1, L_out) -> (N, C_out, L_out)
    return view_backend.squeeze(out_4d, 2)


def conv_transpose2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0),
                     output_padding=(0, 0), groups=1, dilation=(1, 1)):
    """ConvTranspose2d forward using aclnnConvolution with transposed=True."""
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C_in, H_in, W_in = input.shape
    C_in_w, C_out_g, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding
    opH, opW = output_padding
    dH, dW = dilation
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    H_out = (H_in - 1) * sH - 2 * pH + ekH + opH
    W_out = (W_in - 1) * sW - 2 * pW + ekW + opW
    C_out = C_out_g * groups
    out_shape = (N, C_out, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    bias_ptr = None
    bias_shape = None
    bias_stride = None
    if bias is not None:
        bias_ptr = _unwrap_storage(bias).data_ptr()
        bias_shape = bias.shape
        bias_stride = bias.stride

    aclnn.convolution(
        _unwrap_storage(input).data_ptr(),
        _unwrap_storage(weight).data_ptr(),
        bias_ptr,
        input.shape, input.stride,
        weight.shape, weight.stride,
        bias_shape, bias_stride,
        input.dtype,
        stride, padding, dilation,
        True,  # transposed
        output_padding,
        groups,
        out_ptr, out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def conv_transpose1d(input, weight, bias=None, stride=(1,), padding=(0,),
                     output_padding=(0,), groups=1, dilation=(1,)):
    """ConvTranspose1d forward via conv_transpose2d with unsqueezed spatial dim."""
    from ..common import view as view_backend
    input_4d = view_backend.unsqueeze(input, 2)
    weight_4d = view_backend.unsqueeze(weight, 2)
    out_4d = conv_transpose2d(input_4d, weight_4d, bias,
                              stride=(1, stride[0]),
                              padding=(0, padding[0]),
                              output_padding=(0, output_padding[0]),
                              groups=groups,
                              dilation=(1, dilation[0]))
    return view_backend.squeeze(out_4d, 2)


def max_pool2d(input, kernel_size, stride, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """MaxPool2d forward using aclnnMaxPool2dWithMask (supports fp32/fp16 on Ascend910B)."""
    import math as _math
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dH, dW = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    N, C, H, W = input.shape
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    if ceil_mode:
        H_out = _math.ceil((H + 2 * pH - ekH) / sH) + 1
        W_out = _math.ceil((W + 2 * pW - ekW) / sW) + 1
    else:
        H_out = (H + 2 * pH - ekH) // sH + 1
        W_out = (W + 2 * pW - ekW) // sW + 1

    out_shape = (N, C, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    # aclnnMaxPool2dWithMask returns a mask tensor (int8) used for backward.
    # mask shape: (N, C, kH*kW, (ceil(outH*outW/16)+1)*32)
    BLOCKSIZE = 16
    mask_H = kH * kW
    mask_W = (_math.ceil(H_out * W_out / BLOCKSIZE) + 1) * 32
    mask_shape = (N, C, mask_H, mask_W)
    mask_stride = npu_runtime._contiguous_stride(mask_shape)
    mask_numel = _numel(mask_shape)
    mask_ptr = npu_runtime._alloc_device(max(mask_numel, 1), runtime=runtime)  # int8 = 1 byte each

    aclnn.max_pool2d_with_mask(
        _unwrap_storage(input).data_ptr(), out_ptr, mask_ptr,
        input.shape, input.stride, input.dtype,
        [kH, kW], [sH, sW], [pH, pW], [dH, dW], ceil_mode,
        out_shape, out_stride, mask_shape, mask_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    out = _wrap_tensor(out_storage, out_shape, out_stride)
    out._backward_data = {
        "mask_ptr": mask_ptr, "mask_shape": mask_shape, "mask_stride": mask_stride,
        "kernel_size": (kH, kW), "strides": (sH, sW), "padding": (pH, pW),
        "dilation": (dH, dW), "ceil_mode": ceil_mode,
    }
    return out


def avg_pool2d(input, kernel_size, stride, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None):
    """AvgPool2d forward using aclnnAvgPool2d."""
    import math as _math
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sH, sW = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pH, pW = (padding, padding) if isinstance(padding, int) else tuple(padding)

    N, C, H, W = input.shape
    if ceil_mode:
        H_out = _math.ceil((H + 2 * pH - kH) / sH) + 1
        W_out = _math.ceil((W + 2 * pW - kW) / sW) + 1
    else:
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

    out_shape = (N, C, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.avg_pool2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        [kH, kW], [sH, sW], [pH, pW],
        ceil_mode, count_include_pad, divisor_override,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def adaptive_avg_pool2d(input, output_size):
    """AdaptiveAvgPool2d forward — composite implementation via avg_pool2d.

    Uses avg_pool2d with computed kernel_size/stride/padding to avoid
    aclnnAdaptiveAvgPool2d cross-op contamination issues on Ascend910B
    (CANN 8.3.RC2 bug where cubeMathType=1 ops corrupt AdaptiveAvgPool2d state).
    """
    import math as _math

    if isinstance(output_size, int):
        oH = oW = output_size
    else:
        oH, oW = output_size

    N, C, H, W = input.shape

    # Compute avg_pool2d parameters that produce the desired adaptive output.
    # PyTorch's adaptive pooling algorithm (from ATen/native/AdaptiveAveragePooling.cpp):
    #   start_index(i) = floor(i * input_size / output_size)
    #   end_index(i)   = ceil((i+1) * input_size / output_size)
    # When input_size is evenly divisible by output_size, this simplifies to
    # a regular avg_pool2d with stride = input_size // output_size and
    # kernel_size = input_size - (output_size - 1) * stride.

    def _can_use_regular_pool(in_sz, out_sz):
        """Check if adaptive pool can be expressed as regular avg_pool2d."""
        if out_sz == 0:
            return False
        if in_sz % out_sz == 0:
            return True
        # Also works when all windows have the same size
        stride = in_sz // out_sz
        kernel = in_sz - (out_sz - 1) * stride
        # Verify all windows produce valid output
        return stride > 0 and kernel > 0 and (out_sz - 1) * stride + kernel == in_sz

    if _can_use_regular_pool(H, oH) and _can_use_regular_pool(W, oW):
        sH = H // oH
        sW = W // oW
        kH = H - (oH - 1) * sH
        kW = W - (oW - 1) * sW
        return avg_pool2d(input, kernel_size=(kH, kW), stride=(sH, sW),
                          padding=0, ceil_mode=False,
                          count_include_pad=True, divisor_override=None)

    # Fallback: try the native ACLNN kernel for non-uniform window sizes
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    out_shape = (N, C, oH, oW)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.adaptive_avg_pool2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        [oH, oW], out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


# ---------------------------------------------------------------
# P1 ops: std, reciprocal, addmm, einsum, upsample_nearest2d,
#          upsample_bilinear2d, one_hot
# ---------------------------------------------------------------

def std_(a, dim=None, unbiased=True, keepdim=False):
    """Compute std as sqrt(var). aclnnStd/aclnnVar all-reduce fails on 910B."""
    if dim is None:
        # aclnnVar fails with 161002 for all-reduce; reshape to (1, N) and var(dim=1)
        n = 1
        for s in a.shape:
            n *= s
        flat = a.contiguous().view((1, n))
        v = var_(flat, dim=1, unbiased=unbiased, keepdim=False)
        return _unary_op(v, aclnn.sqrt, "sqrt")
    v = var_(a, dim=dim, unbiased=unbiased, keepdim=keepdim)
    return _unary_op(v, aclnn.sqrt, "sqrt")


def reciprocal_(a):
    return _unary_op(a, aclnn.reciprocal, "reciprocal")


def addmm(input, mat1, mat2, beta=1, alpha=1):
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    M, K = mat1.shape
    _, N = mat2.shape
    out_shape = (M, N)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.addmm(
        _unwrap_storage(input).data_ptr(),
        _unwrap_storage(mat1).data_ptr(),
        _unwrap_storage(mat2).data_ptr(),
        out_ptr,
        input.shape, input.stride, input.dtype,
        mat1.shape, mat1.stride,
        mat2.shape, mat2.stride,
        out_shape, out_stride,
        beta, alpha,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def _einsum_output_shape(equation, operands):
    """Parse einsum equation to determine output shape."""
    lhs, rhs = equation.replace(' ', '').split('->')
    inputs = lhs.split(',')

    label_sizes = {}
    for inp_labels, operand in zip(inputs, operands):
        for label, size in zip(inp_labels, operand.shape):
            label_sizes[label] = size

    return tuple(label_sizes[label] for label in rhs)


def _einsum_is_matmul(equation):
    """Check if einsum is a matmul pattern like ...ij,...jk->...ik"""
    eq = equation.replace(' ', '')
    if '->' not in eq:
        return False
    lhs, rhs = eq.split('->')
    inputs = lhs.split(',')
    if len(inputs) != 2:
        return False
    a_labels, b_labels = inputs
    if len(a_labels) < 2 or len(b_labels) < 2:
        return False
    # Check: last dim of A == first non-batch dim of B (contraction)
    # Patterns: ij,jk->ik  bij,bjk->bik  ...ij,...jk->...ik
    batch_a = a_labels[:-2]
    batch_b = b_labels[:-2]
    if batch_a != batch_b:
        return False
    i, j1 = a_labels[-2], a_labels[-1]
    j2, k = b_labels[-2], b_labels[-1]
    if j1 != j2:
        return False
    expected_rhs = batch_a + i + k
    return rhs == expected_rhs


def einsum_(equation, operands):
    """Compute einsum as composite (aclnnEinsum has 161002 on CANN 8.3.RC2).

    Supported patterns:
    - matmul:  ...ij,...jk->...ik
    - transpose: ij->ji, ...ij->...ji
    - inner product: i,i-> or ...i,...i->...
    - batch diagonal sum: ...ii->...i (trace-like)
    """
    from ..._dispatch import dispatch as _dispatch

    eq = equation.replace(' ', '')

    if len(operands) == 2 and _einsum_is_matmul(eq):
        return _dispatch("matmul", operands[0].device.type, operands[0], operands[1])

    # Parse equation
    if '->' not in eq:
        raise NotImplementedError(f"einsum implicit output not supported on NPU: {equation}")
    lhs, rhs = eq.split('->')
    inputs = lhs.split(',')

    # Single-operand transpose: ij->ji or ...ij->...ji
    if len(operands) == 1 and len(inputs) == 1:
        a = operands[0]
        in_labels = inputs[0]
        if len(in_labels) == len(rhs) and set(in_labels) == set(rhs):
            # Pure permutation
            perm = [in_labels.index(c) for c in rhs]
            return _dispatch("permute", a.device.type, a, perm)
        # Trace or reduction patterns
        label_sizes = {}
        for label, size in zip(in_labels, a.shape):
            label_sizes[label] = size
        # Sum over contracted labels
        contracted = [i for i, label in enumerate(in_labels) if label not in rhs]
        if contracted:
            result = a
            for dim in sorted(contracted, reverse=True):
                result = _dispatch("sum", result.device.type, result, dim=dim, keepdim=False)
            return result

    # Two-operand inner product: i,i-> or ...i,...i->...
    if len(operands) == 2 and len(inputs) == 2:
        a, b = operands
        a_labels, b_labels = inputs
        # Check if this is element-wise mul + sum pattern
        contracted = set(a_labels) & set(b_labels) - set(rhs)
        if contracted:
            prod = _dispatch("mul", a.device.type, a, b)
            # Sum over contracted dims (using a_labels ordering)
            sum_dims = sorted([i for i, label in enumerate(a_labels) if label in contracted], reverse=True)
            result = prod
            for dim in sum_dims:
                result = _dispatch("sum", result.device.type, result, dim=dim, keepdim=False)
            return result

    raise NotImplementedError(f"einsum pattern not supported on NPU: {equation}")


def upsample_nearest2d(input, output_size):
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C = input.shape[0], input.shape[1]
    oH, oW = output_size
    out_shape = (N, C, oH, oW)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.upsample_nearest2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        output_size, out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def upsample_bilinear2d(input, output_size, align_corners, scales_h, scales_w):
    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C = input.shape[0], input.shape[1]
    oH, oW = output_size
    out_shape = (N, C, oH, oW)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(input.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.upsample_bilinear2d(
        _unwrap_storage(input).data_ptr(), out_ptr,
        input.shape, input.stride, input.dtype,
        output_size, align_corners, scales_h, scales_w,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), input.dtype, device=input.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def one_hot(indices, num_classes=-1):
    runtime = npu_runtime.get_runtime((indices.device.index or 0))
    stream = npu_state.current_stream((indices.device.index or 0))

    from ..._dtype import float32 as f32, int64 as i64

    if num_classes < 0:
        max_val = amax(indices)
        import numpy as np
        storage = _unwrap_storage(max_val)
        nbytes = _numel(max_val.shape) * _dtype_itemsize(max_val.dtype)
        buf = (ctypes.c_uint8 * max(nbytes, 1))()
        npu_runtime._memcpy_d2h(ctypes.addressof(buf), nbytes, storage.data_ptr(), runtime=runtime)
        arr = np.frombuffer(buf, dtype=np.int64 if max_val.dtype == i64 else np.int32)
        num_classes = int(arr[0]) + 1

    import numpy as np
    on_data = np.array([1.0], dtype=np.float32)
    off_data = np.array([0.0], dtype=np.float32)
    on_ptr = npu_runtime._alloc_device(4, runtime=runtime)
    off_ptr = npu_runtime._alloc_device(4, runtime=runtime)
    npu_runtime._memcpy_h2d(on_ptr, 4, on_data.ctypes.data, runtime=runtime)
    npu_runtime._memcpy_h2d(off_ptr, 4, off_data.ctypes.data, runtime=runtime)

    out_shape = tuple(indices.shape) + (num_classes,)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * 4, runtime=runtime)

    aclnn.one_hot(
        _unwrap_storage(indices).data_ptr(), on_ptr, off_ptr, out_ptr,
        indices.shape, indices.stride, indices.dtype,
        (1,), (1,), f32,
        (1,), (1,), f32,
        out_shape, out_stride, f32,
        num_classes, -1,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), f32, device=indices.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def instance_norm(input, weight=None, bias=None, running_mean=None, running_var=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=False):
    """Instance normalization as composite of existing dispatched ops.

    Note: aclnnInstanceNorm returns 161002 on CANN 8.3.RC2 (Ascend910B),
    so we use composite implementation.
    """
    if input.dim() < 2:
        raise ValueError("instance_norm expects input with at least 2 dims")

    C = int(input.shape[1])
    ndim = input.dim()
    spatial_axes = list(range(2, ndim))

    if use_input_stats:
        mean_t = mean(input, dim=spatial_axes, keepdim=True)
        diff = sub(input, mean_t)
        var_t = mean(mul(diff, diff), dim=spatial_axes, keepdim=True)

        if running_mean is not None:
            batch_dims = [0] + spatial_axes
            global_mean = mean(input, dim=batch_dims, keepdim=False)
            new_rm = add(mul(running_mean, (1.0 - float(momentum))), mul(global_mean, float(momentum)))
            copy_(running_mean, new_rm)
        if running_var is not None:
            batch_dims = [0] + spatial_axes
            global_diff = sub(input, mean(input, dim=batch_dims, keepdim=True))
            global_var = mean(mul(global_diff, global_diff), dim=batch_dims, keepdim=False)
            new_rv = add(mul(running_var, (1.0 - float(momentum))), mul(global_var, float(momentum)))
            copy_(running_var, new_rv)
    else:
        stats_shape = (1, C) + (1,) * (ndim - 2)
        mean_t = reshape(running_mean, stats_shape)
        var_t = reshape(running_var, stats_shape)
        diff = sub(input, mean_t)

    eps_t = _scalar_to_npu_tensor(float(eps), mean_t)
    denom = sqrt(add(var_t, eps_t))
    out = div(diff, denom)

    if weight is not None:
        w_shape = (1, C) + (1,) * (ndim - 2)
        w = reshape(weight, w_shape)
        out = mul(out, w)
    if bias is not None:
        b_shape = (1, C) + (1,) * (ndim - 2)
        b = reshape(bias, b_shape)
        out = add(out, b)
    return out


# --- P1 ops ---

def baddbmm(self_tensor, batch1, batch2, beta=1.0, alpha=1.0):
    """beta * self + alpha * (batch1 @ batch2)"""
    runtime = npu_runtime.get_runtime((self_tensor.device.index or 0))
    stream = npu_state.current_stream((self_tensor.device.index or 0))
    self_storage = _unwrap_storage(self_tensor)
    b1_storage = _unwrap_storage(batch1)
    b2_storage = _unwrap_storage(batch2)
    # Output shape: (B, N, P) from (B, N, M) @ (B, M, P)
    B = batch1.shape[0]
    N = batch1.shape[1]
    P = batch2.shape[2]
    out_shape = (B, N, P)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(self_tensor.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    if hasattr(beta, "shape"):
        beta = float(_unwrap_storage(beta).to_numpy())
    if hasattr(alpha, "shape"):
        alpha = float(_unwrap_storage(alpha).to_numpy())
    aclnn.baddbmm(
        self_storage.data_ptr(), b1_storage.data_ptr(), b2_storage.data_ptr(), out_ptr,
        self_tensor.shape, self_tensor.stride, batch1.shape, batch1.stride,
        batch2.shape, batch2.stride, out_shape, out_stride,
        self_tensor.dtype, float(beta), float(alpha),
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), self_tensor.dtype, device=self_tensor.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def trace_op(a):
    """Sum of diagonal elements of a 2D matrix."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    out_shape = ()
    out_stride = ()
    out_size = max(1, _numel(out_shape)) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.strace(
        a_storage.data_ptr(), out_ptr,
        a.shape, a.stride, a.dtype,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, max(1, _numel(out_shape)), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def cummin_op(a, dim):
    """Cumulative minimum along a dimension. Returns namedtuple (values, indices)."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    ndim = len(a.shape)
    if dim < 0:
        dim = dim + ndim
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    values_size = out_numel * _dtype_itemsize(a.dtype)
    indices_size = out_numel * _dtype_itemsize("int64")
    values_ptr = npu_runtime._alloc_device(values_size, runtime=runtime)
    indices_ptr = npu_runtime._alloc_device(indices_size, runtime=runtime)
    aclnn.cummin(
        a_storage.data_ptr(), values_ptr, indices_ptr,
        a.shape, a.stride, a.dtype,
        dim, out_shape, out_stride,
        runtime, stream=stream.stream,
    )
    values_storage = npu_typed_storage_from_ptr(values_ptr, out_numel, a.dtype, device=a.device)
    indices_storage = npu_typed_storage_from_ptr(indices_ptr, out_numel, "int64", device=a.device)
    values = _wrap_tensor(values_storage, out_shape, out_stride)
    indices = _wrap_tensor(indices_storage, out_shape, out_stride)
    return values, indices


def logsumexp_op(a, dim, keepdim=False):
    """LogSumExp reduction along dim."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    ndim = len(a.shape)
    if isinstance(dim, int):
        dims = [dim % ndim if ndim > 0 else 0]
    else:
        dims = [d % ndim if ndim > 0 else 0 for d in dim]
    out_shape = _reduce_out_shape(a.shape, dims, keepdim)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = max(1, _numel(out_shape))
    out_size = out_numel * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.logsumexp(
        a_storage.data_ptr(), out_ptr,
        a.shape, a.stride, a.dtype,
        dims, keepdim,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, out_numel, a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def renorm_op(a, p, dim, maxnorm):
    """Renormalize sub-tensors along dim so that p-norm <= maxnorm."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    ndim = len(a.shape)
    if dim < 0:
        dim = dim + ndim
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.renorm(
        a_storage.data_ptr(), out_ptr,
        a.shape, a.stride, a.dtype,
        float(p), dim, float(maxnorm),
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def logical_xor(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.logical_xor(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
        out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def nansum(a, dim=None, keepdim=False):
    """Sum ignoring NaN values. Composite: where(isnan, 0, x) then sum.

    Note: aclnnReduceNansum returns 161002 on CANN 8.3.RC2 (Ascend910B).
    """
    zero = _scalar_to_npu_tensor(0.0, a)
    nan_mask = isnan(a)
    clean = where(nan_mask, zero, a)
    return sum_(clean, dim=dim, keepdim=keepdim)


def cross_op(a, b, dim=-1):
    """Cross product via aclnnLinalgCross."""
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(a.dtype), runtime=runtime)
    aclnn.linalg_cross(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
        out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype,
        int(dim),
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


# ---------- P0: ACLNN large-kernel ops ----------

def im2col_op(a, kernel_size, dilation, padding, stride):
    """F.unfold: extract sliding local blocks.

    Composite implementation: aclnnIm2col returns 561103 on CANN 8.3.RC2.
    Uses pad + flatten + gather with existing NPU ops.
    """
    from ..._dispatch.dispatcher import dispatch
    from ..common import view as view_backend

    N, C, H, W = a.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride
    ekH = (kH - 1) * dH + 1
    ekW = (kW - 1) * dW + 1
    out_H = (H + 2 * pH - ekH) // sH + 1
    out_W = (W + 2 * pW - ekW) // sW + 1
    L = out_H * out_W

    if pH > 0 or pW > 0:
        a = dispatch("pad", "npu", a, (pW, pW, pH, pH))
    a = contiguous(a)

    import numpy as _np
    _, _, H_pad, W_pad = a.shape

    # Build gather indices: for each kernel position, compute flat index into H_pad*W_pad plane
    patches = []
    for kh in range(kH):
        for kw in range(kW):
            row_indices = []
            for oh in range(out_H):
                for ow in range(out_W):
                    r = oh * sH + kh * dH
                    c = ow * sW + kw * dW
                    row_indices.append(r * W_pad + c)
            patches.append(row_indices)

    # Stack into (kH*kW, L), tile to (C*kH*kW, L) with per-channel offsets
    idx_2d = _np.stack([_np.array(p, dtype=_np.int64) for p in patches], axis=0)
    idx_full = _np.tile(idx_2d, (C, 1))

    offsets = _np.arange(C, dtype=_np.int64).reshape(C, 1) * (H_pad * W_pad)
    offsets_tiled = _np.repeat(offsets, kH * kW, axis=0)
    idx_with_offset = idx_full + offsets_tiled

    # Broadcast to (N, C*kH*kW, L), then flatten last two dims for gather
    idx_with_offset_batch = _np.broadcast_to(
        idx_with_offset[None], (N, C * kH * kW, L)
    ).copy()
    idx_flat = idx_with_offset_batch.reshape(N, C * kH * kW * L)

    # Flatten input to (N, C*H_pad*W_pad)
    a_fully_flat = view_backend.reshape(a, (N, C * H_pad * W_pad))

    # Copy index to NPU and gather
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    idx_ptr, _ = npu_runtime._copy_cpu_to_npu(idx_flat, runtime=runtime)
    idx_shape = (N, C * kH * kW * L)
    idx_stride = npu_runtime._contiguous_stride(idx_shape)
    idx_storage = npu_typed_storage_from_ptr(
        idx_ptr, _numel(idx_shape), int64_dtype, device=a.device
    )
    idx_tensor = _wrap_tensor(idx_storage, idx_shape, idx_stride)

    result = gather(a_fully_flat, -1, idx_tensor)
    out_shape = (N, C * kH * kW, L)
    result = view_backend.reshape(result, out_shape)
    return result


def grid_sample_op(input, grid, mode='bilinear', padding_mode='zeros',
                   align_corners=None):
    """F.grid_sample via aclnnGridSampler2D."""
    if align_corners is None:
        align_corners = False
    mode_map = {'bilinear': 0, 'nearest': 1, 'bicubic': 2}
    pad_map = {'zeros': 0, 'border': 1, 'reflection': 2}
    interp = mode_map.get(mode, 0)
    pad = pad_map.get(padding_mode, 0)

    runtime = npu_runtime.get_runtime((input.device.index or 0))
    stream = npu_state.current_stream((input.device.index or 0))

    N, C = input.shape[0], input.shape[1]
    H_out, W_out = grid.shape[1], grid.shape[2]
    out_shape = (N, C, H_out, W_out)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(
        _numel(out_shape) * _dtype_itemsize(input.dtype), runtime=runtime
    )
    aclnn.sgrid_sampler2d(
        _unwrap_storage(input).data_ptr(), _unwrap_storage(grid).data_ptr(),
        out_ptr,
        input.shape, input.stride, grid.shape, grid.stride,
        out_shape, out_stride, input.dtype,
        interp, pad, align_corners,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(
        out_ptr, _numel(out_shape), input.dtype, device=input.device
    )
    return _wrap_tensor(out_storage, out_shape, out_stride)


def affine_grid_op(theta, size, align_corners=None):
    """F.affine_grid via aclnnAffineGrid."""
    if align_corners is None:
        align_corners = False

    runtime = npu_runtime.get_runtime((theta.device.index or 0))
    stream = npu_state.current_stream((theta.device.index or 0))

    N = size[0]
    if len(size) == 4:
        H, W = size[2], size[3]
        out_shape = (N, H, W, 2)
    else:
        D, H, W = size[2], size[3], size[4]
        out_shape = (N, D, H, W, 3)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(
        _numel(out_shape) * _dtype_itemsize(theta.dtype), runtime=runtime
    )
    aclnn.saffine_grid(
        _unwrap_storage(theta).data_ptr(), out_ptr,
        theta.shape, theta.stride, theta.dtype,
        list(size), align_corners,
        out_shape, out_stride,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(
        out_ptr, _numel(out_shape), theta.dtype, device=theta.device
    )
    return _wrap_tensor(out_storage, out_shape, out_stride)


# ---------- P1: View / reshape composite ops ----------

def broadcast_to_op(a, shape):
    """Tensor.broadcast_to — delegates to expand."""
    return expand(a, shape)


def movedim_op(a, source, destination):
    """torch.movedim — compute permutation then delegate to permute."""
    from ..common import view as view_backend
    ndim = a.dim()
    if isinstance(source, int):
        source = [source]
    if isinstance(destination, int):
        destination = [destination]
    source = [s % ndim for s in source]
    destination = [d % ndim for d in destination]
    order = [i for i in range(ndim) if i not in source]
    dst_src = sorted(zip(destination, source))
    for dst, src in dst_src:
        order.insert(dst, src)
    return view_backend.permute(a, order)


def unflatten_op(a, dim, sizes):
    """Tensor.unflatten — reshape one dim into multiple dims."""
    from ..common import view as view_backend
    ndim = a.dim()
    d = dim if dim >= 0 else dim + ndim
    new_shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    return view_backend.reshape(a, new_shape)


def diagonal_op(a, offset=0, dim1=0, dim2=1):
    """torch.diagonal — permute + flatten + gather.

    Uses gather with pre-expanded numpy indices to avoid ACLNN offset bug
    (select creates views with non-zero offset that _create_tensor ignores).
    """
    from ..common import view as view_backend
    import numpy as _np

    ndim = a.dim()
    d1 = dim1 % ndim
    d2 = dim2 % ndim
    if d1 == d2:
        raise RuntimeError("diagonal: dim1 and dim2 cannot be equal")

    # Move d1, d2 to the last two dims
    dims = [i for i in range(ndim) if i != d1 and i != d2] + [d1, d2]
    t = view_backend.permute(a, dims)

    rows = t.shape[-2]
    cols = t.shape[-1]
    if offset >= 0:
        diag_len = max(0, min(rows, cols - offset))
    else:
        diag_len = max(0, min(rows + offset, cols))

    if diag_len == 0:
        batch_shape = t.shape[:-2]
        out_shape = batch_shape + (0,)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        out_ptr = npu_runtime._alloc_device(
            max(1, _numel(out_shape) * _dtype_itemsize(a.dtype)), runtime=runtime
        )
        out_storage = npu_typed_storage_from_ptr(
            out_ptr, max(1, _numel(out_shape)), a.dtype, device=a.device
        )
        return _wrap_tensor(out_storage, out_shape, out_stride)

    batch_shape = t.shape[:-2]
    flat_shape = batch_shape + (rows * cols,)
    t_flat = view_backend.reshape(contiguous(t), flat_shape)

    if offset >= 0:
        flat_idx = [(i * cols + i + offset) for i in range(diag_len)]
    else:
        flat_idx = [((i - offset) * cols + i) for i in range(diag_len)]

    idx_1d = _np.array(flat_idx, dtype=_np.int64)
    idx_expanded = _np.broadcast_to(idx_1d, batch_shape + (diag_len,)).copy()

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    idx_ptr, _ = npu_runtime._copy_cpu_to_npu(idx_expanded, runtime=runtime)
    idx_shape = batch_shape + (diag_len,)
    idx_stride = npu_runtime._contiguous_stride(idx_shape)
    idx_storage = npu_typed_storage_from_ptr(
        idx_ptr, _numel(idx_shape), int64_dtype, device=a.device
    )
    idx_tensor = _wrap_tensor(idx_storage, idx_shape, idx_stride)

    result = gather(t_flat, -1, idx_tensor)
    return result
