"""PyTorch-compatible tensor creation functions.

Uses MindSpore pyboost primitives for tensor creation instead of numpy.
"""

import numpy as np
import mindspore
from . import _dtype as dtype_mod
from ._tensor import Tensor, _data_to_numpy, _compute_strides
from ._storage import TypedStorage, UntypedStorage
from ._device import _get_default_device, device as device_cls
from .configs import DEVICE_TARGET


def _resolve_dtype(dtype, default=None):
    """Resolve dtype or return default."""
    if dtype is not None:
        return dtype
    return default or dtype_mod.float32


def _resolve_shape(args):
    """Accept shape as *args or tuple: zeros(3, 4) or zeros((3, 4))."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return tuple(args)


def _resolve_device(device):
    """Resolve device from argument or context, defaulting based on MindSpore context."""
    if device is not None:
        if isinstance(device, str):
            return device_cls(device)
        return device
    # Check context
    ctx_device = _get_default_device()
    if ctx_device is not None:
        return ctx_device
    # Default based on MindSpore device target
    if DEVICE_TARGET == 'Ascend':
        return device_cls("npu")
    return device_cls("cpu")


def _get_ms_device():
    """Get MindSpore device string ('CPU' or 'Ascend')."""
    return DEVICE_TARGET


def _get_wrap_result():
    """Get the appropriate _wrap_result function for the current device."""
    if DEVICE_TARGET == 'Ascend':
        from ._backends.pyboost_ascend import _wrap_result
    else:
        from ._backends.pyboost_cpu import _wrap_result
    return _wrap_result


def _dtype_to_ms(dtype):
    """Convert mindtorch dtype to mindspore dtype."""
    _map = {
        dtype_mod.float16: mindspore.float16,
        dtype_mod.float32: mindspore.float32,
        dtype_mod.float64: mindspore.float64,
        dtype_mod.bfloat16: mindspore.bfloat16,
        dtype_mod.int8: mindspore.int8,
        dtype_mod.int16: mindspore.int16,
        dtype_mod.int32: mindspore.int32,
        dtype_mod.int64: mindspore.int64,
        dtype_mod.uint8: mindspore.uint8,
        dtype_mod.bool: mindspore.bool_,
    }
    return _map.get(dtype, mindspore.float32)


def _create_meta_tensor(data, dtype, device, requires_grad):
    """Create a meta tensor with shape/dtype but no actual storage."""
    # Convert data to numpy to get shape and dtype info
    arr = _data_to_numpy(data, dtype)
    shape = arr.shape

    # Create storage with no actual data
    storage = TypedStorage.__new__(TypedStorage)
    storage._ms_tensor = None  # No actual data
    storage._size = 0
    storage._dtype = dtype_mod.numpy_to_dtype(arr.dtype) if dtype is None else dtype
    storage._device = device

    # Create tensor with the storage
    result = Tensor.__new__(Tensor)
    result._storage = storage
    result._shape = shape
    result._stride = _compute_strides(shape)
    result._storage_offset = 0
    result._dtype = storage._dtype
    result._device = device
    result._requires_grad = requires_grad
    result._grad_fn = None
    result._grad = None
    result._version = 0
    result._hooks = {}
    result._hook_counter = 0

    return result


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    """Create tensor from data (always copies). Equivalent to torch.tensor()."""
    resolved_device = _resolve_device(device)

    # Handle meta device specially - create tensor with shape but no storage
    if resolved_device.type == "meta":
        return _create_meta_tensor(data, dtype, resolved_device, requires_grad)

    return Tensor(data, dtype=dtype, device=str(resolved_device), requires_grad=requires_grad)


def zeros(*size, dtype=None, device=None, requires_grad=False, out=None):
    """Create tensor filled with zeros using MindSpore kernel."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    ms_tensor = mindspore.ops.zeros(shape, ms_dtype)
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def ones(*size, dtype=None, device=None, requires_grad=False, out=None):
    """Create tensor filled with ones using MindSpore kernel."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    ms_tensor = mindspore.ops.ones(shape, ms_dtype)
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def empty(*size, dtype=None, device=None, requires_grad=False, out=None, memory_format=None, pin_memory=False, **kwargs):
    """Create uninitialized tensor using MindSpore kernel."""
    # Handle size= keyword argument
    if 'size' in kwargs:
        shape = tuple(kwargs['size'])
    elif len(size) == 0:
        shape = ()
    else:
        shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    # Use zeros as empty (MindSpore doesn't have uninitialized allocation)
    ms_tensor = mindspore.ops.zeros(shape, ms_dtype)
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def full(size, fill_value, *, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with fill_value using MindSpore kernel."""
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    try:
        ms_tensor = mindspore.ops.full(size, fill_value, dtype=ms_dtype)
    except RuntimeError:
        # Fallback to numpy when MindSpore graph ops unavailable
        np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
        ms_tensor = mindspore.Tensor(np.full(size, fill_value, dtype=np_dtype))
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with fill_value, with same shape as input."""
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else str(input.device)
    return full(input.shape, fill_value, dtype=dt, device=dev, requires_grad=requires_grad)


def arange(*args, dtype=None, device=None, requires_grad=False):
    """arange(end), arange(start, end), arange(start, end, step)."""
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    # Parse args: arange(end), arange(start, end), arange(start, end, step)
    if len(args) == 1:
        start, end, step = 0, args[0], 1
    elif len(args) == 2:
        start, end, step = args[0], args[1], 1
    else:
        start, end, step = args[0], args[1], args[2]

    ms_dtype = _dtype_to_ms(dtype) if dtype is not None else None
    try:
        ms_tensor = mindspore.ops.arange(start, end, step, dtype=ms_dtype)
    except RuntimeError:
        # Fallback to numpy when MindSpore graph ops unavailable
        np_dtype = dtype_mod.dtype_to_numpy(dtype) if dtype is not None else None
        ms_tensor = mindspore.Tensor(np.arange(start, end, step, dtype=np_dtype))
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False):
    """Create evenly spaced tensor using MindSpore kernel."""
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    try:
        ms_tensor = mindspore.ops.linspace(mindspore.Tensor(start, ms_dtype),
                                            mindspore.Tensor(end, ms_dtype),
                                            steps)
    except RuntimeError:
        # Fallback to numpy when MindSpore graph ops unavailable
        np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
        ms_tensor = mindspore.Tensor(np.linspace(start, end, steps, dtype=np_dtype))
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def eye(n, m=None, *, dtype=None, device=None, requires_grad=False):
    """Create identity matrix using MindSpore kernel."""
    if m is None:
        m = n
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    ms_tensor = mindspore.ops.eye(n, m, ms_dtype)
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def randn(*size, dtype=None, device=None, requires_grad=False, generator=None):
    """Create tensor with random normal values using MindSpore kernel."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    try:
        ms_tensor = mindspore.ops.randn(shape, dtype=ms_dtype)
    except RuntimeError:
        # Fallback to numpy when MindSpore graph ops unavailable
        np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
        ms_tensor = mindspore.Tensor(np.random.randn(*shape).astype(np_dtype))
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def rand(*size, dtype=None, device=None, requires_grad=False, generator=None):
    """Create tensor with random uniform [0, 1) values using MindSpore kernel."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    try:
        ms_tensor = mindspore.ops.rand(shape, dtype=ms_dtype)
    except RuntimeError:
        # Fallback to numpy when MindSpore graph ops unavailable
        np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
        ms_tensor = mindspore.Tensor(np.random.rand(*shape).astype(np_dtype))
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def zeros_like(input, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create zero tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return zeros(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def ones_like(input, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create ones tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return ones(*input.shape, dtype=dt, device=device or str(input.device),
                requires_grad=requires_grad)


def empty_like(input, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create empty tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return empty(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def from_numpy(ndarray):
    """Create tensor from numpy array (shares memory where possible)."""
    arr = np.ascontiguousarray(ndarray)
    dt = dtype_mod.numpy_to_dtype(arr.dtype)
    resolved_device = _resolve_device(None)
    return Tensor(arr, dtype=dt, device=str(resolved_device))


def as_tensor(data, dtype=None, device=None):
    """Convert data to tensor, sharing memory when possible.

    Unlike tensor(), as_tensor() avoids copying data when possible.
    If data is already a Tensor with matching dtype and device, it is returned as-is.
    If data is a numpy array with compatible dtype, memory may be shared.

    Args:
        data: Initial data - can be list, tuple, numpy array, scalar, or Tensor
        dtype: Desired dtype (optional)
        device: Desired device (optional)

    Returns:
        Tensor: The resulting tensor
    """
    resolved_device = _resolve_device(device)
    # If already a tensor, check if we need to convert
    if isinstance(data, Tensor):
        if dtype is None and device is None:
            return data
        if dtype is not None and data.dtype != dtype:
            return data.to(dtype=dtype, device=device)
        if device is not None and str(data.device) != str(device):
            return data.to(device=device)
        return data

    # For numpy arrays, try to share memory if possible
    if isinstance(data, np.ndarray):
        target_dtype = dtype
        if target_dtype is None:
            target_dtype = dtype_mod.numpy_to_dtype(data.dtype)

        np_target = dtype_mod.dtype_to_numpy(target_dtype)
        if np_target is not None and data.dtype == np_target:
            # Can share memory
            return Tensor(data, dtype=target_dtype, device=str(resolved_device))
        else:
            # Need to convert dtype
            if np_target is not None:
                arr = data.astype(np_target)
            else:
                arr = data
            return Tensor(arr, dtype=target_dtype, device=str(resolved_device))

    # For other types (lists, scalars, etc.), create a new tensor
    return Tensor(data, dtype=dtype, device=str(resolved_device))


def asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False):
    """Convert input to a tensor.

    This is similar to as_tensor() but also supports Storage objects,
    which is needed for safetensors compatibility.

    Args:
        obj: Input data - can be tensor, numpy array, storage, list, etc.
        dtype: Desired dtype (optional)
        device: Desired device (optional)
        copy: Whether to copy data (None means copy only if needed)
        requires_grad: Whether to track gradients

    Returns:
        Tensor: The resulting tensor
    """
    resolved_device = _resolve_device(device)
    dev_str = str(resolved_device)

    # Handle Storage objects (needed for safetensors)
    if isinstance(obj, (UntypedStorage, TypedStorage)):
        # Get raw data from storage
        if isinstance(obj, UntypedStorage):
            # UntypedStorage - need dtype to interpret bytes
            if dtype is None:
                dtype = dtype_mod.uint8
            raw_data = obj._data.asnumpy()
            # Calculate element count based on dtype
            element_size = dtype.itemsize
            count = len(raw_data) // element_size
            # Reinterpret bytes as target dtype
            np_dtype = dtype_mod.dtype_to_numpy(dtype)
            if np_dtype is not None:
                arr = np.frombuffer(raw_data.tobytes(), dtype=np_dtype, count=count)
            else:
                # Handle bfloat16
                if dtype == dtype_mod.bfloat16:
                    try:
                        import ml_dtypes
                        arr = np.frombuffer(raw_data.tobytes(), dtype=ml_dtypes.bfloat16, count=count)
                    except ImportError:
                        arr = np.frombuffer(raw_data.tobytes(), dtype=np.uint16, count=count)
                else:
                    raise TypeError(f"Unsupported dtype: {dtype}")
            return Tensor(arr.copy(), dtype=dtype, device=dev_str, requires_grad=requires_grad)
        else:
            # TypedStorage - dtype is already known
            target_dtype = dtype if dtype is not None else obj.dtype
            arr = obj._ms_tensor.asnumpy()
            return Tensor(arr.copy(), dtype=target_dtype, device=dev_str, requires_grad=requires_grad)

    # Handle Tensor
    if isinstance(obj, Tensor):
        if copy is True:
            return obj.clone().to(dtype=dtype, device=device)
        if dtype is None and device is None:
            return obj
        return obj.to(dtype=dtype, device=device)

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        target_dtype = dtype
        if target_dtype is None:
            target_dtype = dtype_mod.numpy_to_dtype(obj.dtype)
        if copy is True:
            obj = obj.copy()
        return Tensor(obj, dtype=target_dtype, device=dev_str, requires_grad=requires_grad)

    # Handle other types (lists, scalars, etc.)
    return Tensor(obj, dtype=dtype, device=dev_str, requires_grad=requires_grad)


def frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False):
    """Create a 1-D tensor from a buffer object that exposes the buffer interface.

    Args:
        buffer: An object that exposes the buffer interface (bytes, bytearray, memoryview, etc.)
        dtype: The desired dtype of the tensor
        count: Number of elements to read (-1 means all elements)
        offset: Byte offset to start reading from
        requires_grad: Whether to track gradients

    Returns:
        Tensor: A 1-D tensor containing the buffer data
    """
    resolved_device = _resolve_device(None)
    dev_str = str(resolved_device)

    # Handle bfloat16 specially since numpy doesn't natively support it
    if dtype == dtype_mod.bfloat16:
        try:
            import ml_dtypes
            # Use ml_dtypes bfloat16 which MindSpore can convert
            arr = np.frombuffer(buffer, dtype=ml_dtypes.bfloat16, count=count, offset=offset)
            arr = arr.copy()
            return Tensor(arr, dtype=dtype, device=dev_str, requires_grad=requires_grad)
        except ImportError:
            # Fallback: read as uint16 and convert through float32
            arr = np.frombuffer(buffer, dtype=np.uint16, count=count, offset=offset)
            arr = arr.copy()
            # Convert bfloat16 bits to float32: bf16 is upper 16 bits of f32
            arr_f32 = np.zeros(arr.shape, dtype=np.float32)
            arr_f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
            t = Tensor(arr_f32, dtype=dtype_mod.float32, device=dev_str, requires_grad=requires_grad)
            return t.to(dtype)

    # Convert dtype to numpy dtype
    np_dtype = dtype_mod.dtype_to_numpy(dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported dtype: {dtype}")

    # Create numpy array from buffer
    arr = np.frombuffer(buffer, dtype=np_dtype, count=count, offset=offset)

    # Create tensor (don't share memory since buffer might be temporary)
    arr = arr.copy()
    return Tensor(arr, dtype=dtype, device=dev_str, requires_grad=requires_grad)


def randint(low, high=None, size=None, *, dtype=None, device=None, requires_grad=False):
    """Create tensor with random integers in [low, high).

    Args:
        low: Lowest integer (inclusive), or high if high is None
        high: One above highest integer (exclusive)
        size: Shape of output tensor
    """
    if high is None:
        high = low
        low = 0
    if size is None:
        size = ()
    if isinstance(size, int):
        size = (size,)

    dt = dtype or dtype_mod.int64
    ms_dtype = _dtype_to_ms(dt)
    resolved_device = _resolve_device(device)
    _wrap_result = _get_wrap_result()

    try:
        ms_tensor = mindspore.ops.randint(low, high, size, dtype=ms_dtype)
    except RuntimeError:
        # Fallback to numpy when MindSpore graph ops unavailable
        np_dtype = dtype_mod.dtype_to_numpy(dt) or np.int64
        ms_tensor = mindspore.Tensor(np.random.randint(low, high, size=size).astype(np_dtype))
    result = _wrap_result(ms_tensor, device=str(resolved_device))
    result._requires_grad = requires_grad
    return result


def rand_like(input, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create tensor with random values [0, 1) with same shape as input.

    Args:
        input: Input tensor to get shape from
        dtype: Optional dtype (defaults to input dtype)
        device: Optional device (defaults to input device)
        requires_grad: If True, tensor will track gradients
        memory_format: Memory format (ignored)
    """
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else str(input.device)
    return rand(*input.shape, dtype=dt, device=dev, requires_grad=requires_grad)
