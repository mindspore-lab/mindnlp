"""PyTorch-compatible tensor creation functions."""

import numpy as np
import mindspore
from . import _dtype as dtype_mod
from ._tensor import Tensor, _data_to_numpy, _compute_strides
from ._storage import TypedStorage, UntypedStorage
from ._device import _get_default_device, device as device_cls


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
    """Resolve device from argument or context, defaulting to cpu."""
    if device is not None:
        if isinstance(device, str):
            return device_cls(device)
        return device
    # Check context
    ctx_device = _get_default_device()
    if ctx_device is not None:
        return ctx_device
    return device_cls("cpu")


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
    """Create tensor filled with zeros."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.zeros(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False, out=None):
    """Create tensor filled with ones."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.ones(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def empty(*size, dtype=None, device=None, requires_grad=False, out=None, memory_format=None, pin_memory=False, **kwargs):
    """Create uninitialized tensor. Supports empty(3,4) and empty(size=(3,4))."""
    # Handle size= keyword argument
    if 'size' in kwargs:
        shape = tuple(kwargs['size'])
    elif len(size) == 0:
        shape = ()
    else:
        shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.empty(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def full(size, fill_value, *, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with fill_value."""
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.full(size, fill_value, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with fill_value, with same shape as input."""
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else str(input.device)
    return full(input.shape, fill_value, dtype=dt, device=dev, requires_grad=requires_grad)


def zeros_like(input, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create tensor filled with zeros, with same shape as input."""
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else str(input.device)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.zeros(input.shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=dev, requires_grad=requires_grad)


def ones_like(input, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create tensor filled with ones, with same shape as input."""
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else str(input.device)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.ones(input.shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=dev, requires_grad=requires_grad)


def empty_like(input, *, dtype=None, device=None, requires_grad=False, memory_format=None):
    """Create uninitialized tensor, with same shape as input."""
    dt = dtype if dtype is not None else input.dtype
    dev = device if device is not None else str(input.device)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.empty(input.shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=dev, requires_grad=requires_grad)


def arange(*args, dtype=None, device=None, requires_grad=False):
    """arange(end), arange(start, end), arange(start, end, step)."""
    arr = np.arange(*args)
    if dtype is not None:
        np_dtype = dtype_mod.dtype_to_numpy(dtype)
        arr = arr.astype(np_dtype)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)


def linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False):
    """Create evenly spaced tensor."""
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.linspace(start, end, steps, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def eye(n, m=None, *, dtype=None, device=None, requires_grad=False):
    """Create identity matrix."""
    if m is None:
        m = n
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.eye(n, m, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def randn(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor with random normal values."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
    arr = np.random.randn(*shape).astype(np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def rand(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor with random uniform [0, 1) values."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
    arr = np.random.rand(*shape).astype(np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def zeros_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create zero tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return zeros(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def ones_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create ones tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return ones(*input.shape, dtype=dt, device=device or str(input.device),
                requires_grad=requires_grad)


def empty_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create empty tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return empty(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def from_numpy(ndarray):
    """Create tensor from numpy array (shares memory where possible)."""
    arr = np.ascontiguousarray(ndarray)
    dt = dtype_mod.numpy_to_dtype(arr.dtype)
    return Tensor(arr, dtype=dt)


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
            return Tensor(data, dtype=target_dtype, device=device)
        else:
            # Need to convert dtype
            if np_target is not None:
                arr = data.astype(np_target)
            else:
                arr = data
            return Tensor(arr, dtype=target_dtype, device=device)

    # For other types (lists, scalars, etc.), create a new tensor
    return Tensor(data, dtype=dtype, device=device)


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
            return Tensor(arr.copy(), dtype=dtype, device=device, requires_grad=requires_grad)
        else:
            # TypedStorage - dtype is already known
            target_dtype = dtype if dtype is not None else obj.dtype
            arr = obj._ms_tensor.asnumpy()
            return Tensor(arr.copy(), dtype=target_dtype, device=device, requires_grad=requires_grad)

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
        return Tensor(obj, dtype=target_dtype, device=device, requires_grad=requires_grad)

    # Handle other types (lists, scalars, etc.)
    return Tensor(obj, dtype=dtype, device=device, requires_grad=requires_grad)


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
    # Handle bfloat16 specially since numpy doesn't natively support it
    if dtype == dtype_mod.bfloat16:
        try:
            import ml_dtypes
            # Use ml_dtypes bfloat16 which MindSpore can convert
            arr = np.frombuffer(buffer, dtype=ml_dtypes.bfloat16, count=count, offset=offset)
            arr = arr.copy()
            return Tensor(arr, dtype=dtype, requires_grad=requires_grad)
        except ImportError:
            # Fallback: read as uint16 and convert through float32
            arr = np.frombuffer(buffer, dtype=np.uint16, count=count, offset=offset)
            arr = arr.copy()
            # Convert bfloat16 bits to float32: bf16 is upper 16 bits of f32
            arr_f32 = np.zeros(arr.shape, dtype=np.float32)
            arr_f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
            tensor = Tensor(arr_f32, dtype=dtype_mod.float32, requires_grad=requires_grad)
            return tensor.to(dtype)

    # Convert dtype to numpy dtype
    np_dtype = dtype_mod.dtype_to_numpy(dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported dtype: {dtype}")

    # Create numpy array from buffer
    arr = np.frombuffer(buffer, dtype=np_dtype, count=count, offset=offset)

    # Create tensor (don't share memory since buffer might be temporary)
    arr = arr.copy()
    return Tensor(arr, dtype=dtype, requires_grad=requires_grad)


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
    arr = np.random.randint(low, high, size=size)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)
