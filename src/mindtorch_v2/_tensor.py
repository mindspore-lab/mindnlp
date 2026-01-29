"""PyTorch-compatible Tensor class with proper Storage-based semantics.

This Tensor does NOT inherit from MindSpore's Tensor. It holds a reference
to a Storage (which wraps a MindSpore tensor), plus view metadata:
shape, stride, storage_offset.
"""

import math
import numpy as np
import mindspore
from . import _dtype as dtype_mod
from ._storage import TypedStorage
from ._device import device as device_cls

import builtins as _builtins
builtins_bool = _builtins.bool
builtins_float = _builtins.float
builtins_int = _builtins.int


def _compute_strides(shape):
    """Compute contiguous (row-major) strides for a given shape."""
    if len(shape) == 0:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _data_to_numpy(data, dtype=None):
    """Convert input data to numpy array with appropriate dtype."""
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (list, tuple)):
        arr = np.array(data)
    elif isinstance(data, (int, float, bool)):
        arr = np.array(data)
    elif isinstance(data, np.generic):
        # Handle numpy scalar types (e.g., np.float32, np.int64)
        arr = np.array(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to Tensor")

    # Infer dtype
    if dtype is not None:
        np_dtype = dtype_mod.dtype_to_numpy(dtype)
        if np_dtype is not None:
            arr = arr.astype(np_dtype)
    else:
        # Default: float for float-like, int64 for int-like
        if arr.dtype.kind == 'f':
            arr = arr.astype(np.float32)
        elif arr.dtype.kind == 'i':
            arr = arr.astype(np.int64)
        elif arr.dtype.kind == 'b':
            pass  # keep bool
        elif arr.dtype.kind == 'u':
            pass  # keep uint
        elif arr.dtype.kind == 'c':
            arr = arr.astype(np.complex64)

    return arr


def _resolve_neg_one(shape, numel):
    """Resolve -1 in shape tuple."""
    neg_one_idx = None
    known_product = 1
    for i, s in enumerate(shape):
        if s == -1:
            if neg_one_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            neg_one_idx = i
        else:
            known_product *= s

    if neg_one_idx is not None:
        inferred = numel // known_product
        shape = list(shape)
        shape[neg_one_idx] = inferred
        shape = tuple(shape)

    return shape


def _expand_ellipsis(key, ndim):
    """Expand Ellipsis in index tuple."""
    # Count non-None, non-Ellipsis entries
    n_ellipsis = sum(1 for k in key if k is Ellipsis)
    if n_ellipsis == 0:
        return key
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis (...)")

    # Find ellipsis position
    idx = key.index(Ellipsis)
    n_none = sum(1 for k in key if k is None)
    n_specified = len(key) - 1 - n_none  # -1 for ellipsis itself
    n_expand = ndim - n_specified

    expanded = key[:idx] + (slice(None),) * n_expand + key[idx + 1:]
    return expanded


class RemovableHandle:
    """Handle returned by register_hook that allows removing the hook."""

    def __init__(self, hooks_dict, hook_id):
        self._hooks_dict = hooks_dict
        self._hook_id = hook_id

    def remove(self):
        if self._hook_id in self._hooks_dict:
            del self._hooks_dict[self._hook_id]


class Tensor:
    """PyTorch-compatible Tensor backed by Storage.

    Attributes:
        _storage: TypedStorage holding the contiguous data
        _shape: tuple of ints
        _stride: tuple of ints
        _storage_offset: int (offset into storage, in elements)
        _requires_grad: bool
        _grad_fn: autograd Node or None
        _grad: accumulated gradient Tensor or None
        _version: int (incremented on in-place mutation)
    """

    def __init__(self, data=None, *, dtype=None, device=None, requires_grad=False,
                 _storage=None, _shape=None, _stride=None, _storage_offset=0):
        """Create a Tensor.

        Public usage:
            Tensor([1.0, 2.0, 3.0])
            Tensor([[1, 2], [3, 4]], dtype=torch.float64)

        Internal usage (for views):
            Tensor(_storage=s, _shape=(2,3), _stride=(3,1), _storage_offset=0)
        """
        if _storage is not None:
            # Internal: create view over existing storage
            self._storage = _storage
            self._shape = tuple(_shape)
            self._stride = tuple(_stride) if _stride is not None else _compute_strides(self._shape)
            self._storage_offset = _storage_offset
            self._dtype = _storage.dtype
        elif data is not None:
            # Public: create from data
            arr = _data_to_numpy(data, dtype)
            shape = arr.shape
            flat = arr.ravel()
            ms_tensor = mindspore.Tensor(flat)
            self._storage = TypedStorage(ms_tensor)
            self._shape = shape
            self._stride = _compute_strides(shape)
            self._storage_offset = 0
            self._dtype = self._storage.dtype
        else:
            raise ValueError("Must provide either data or _storage")

        self._device = device_cls(device or "cpu")
        self._requires_grad = requires_grad
        self._grad_fn = None
        self._grad = None
        self._version = 0
        self._hooks = {}  # Dict of hook_id -> hook_fn
        self._hook_counter = 0

    # --- Shape / metadata ---

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        """Return the device of this tensor.

        For meta tensors, returns the meta device.
        For regular tensors, returns the device from _device attribute.
        """
        # Check if this is a meta tensor (storage has no actual data)
        if hasattr(self._storage, '_ms_tensor') and self._storage._ms_tensor is None:
            # This is a meta tensor, return meta device
            if hasattr(self, '_device'):
                return self._device
        return self._device

    @property
    def layout(self):
        """Return tensor layout. MindTorch only supports strided tensors."""
        from . import strided
        return strided

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def data(self):
        """Returns a Tensor sharing storage but detached from autograd."""
        return Tensor(
            _storage=self._storage,
            _shape=self._shape,
            _stride=self._stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
        )

    @data.setter
    def data(self, value):
        if isinstance(value, Tensor):
            self._storage = value._storage
            self._shape = value._shape
            self._stride = value._stride
            self._storage_offset = value._storage_offset
            self._dtype = value._dtype
            self._version += 1

    def size(self, dim=None):
        if dim is not None:
            return self._shape[dim]
        return self._shape

    def stride(self, dim=None):
        if dim is not None:
            return self._stride[dim]
        return self._stride

    def storage_offset(self):
        return self._storage_offset

    def storage(self):
        return self._storage

    def data_ptr(self):
        """Return pointer to this tensor's data start.

        This includes the storage offset, so tensors viewing the same storage
        at different offsets will return different pointers.
        This is critical for safetensors shared tensor detection.
        """
        base_ptr = self._storage.data_ptr()
        # Add offset in bytes
        offset_bytes = self._storage_offset * self._dtype.itemsize
        return base_ptr + offset_bytes

    def untyped_storage(self):
        """Return the underlying untyped storage.

        This is needed for compatibility with safetensors which calls
        tensor.untyped_storage().nbytes() to determine storage size.
        """
        return self._storage

    def dim(self):
        return len(self._shape)

    def numel(self):
        result = 1
        for s in self._shape:
            result *= s
        return result

    # Alias for numel (PyTorch compatibility)
    nelement = numel

    def element_size(self):
        return self._dtype.itemsize

    def is_floating_point(self):
        """Check if tensor is of a floating point dtype."""
        floating_point_dtypes = {'float16', 'float32', 'float64', 'bfloat16',
                                 'half', 'float', 'double'}
        return str(self._dtype).split('.')[-1] in floating_point_dtypes

    def is_contiguous(self, memory_format=None):
        """Check if tensor is contiguous in row-major order."""
        expected = _compute_strides(self._shape)
        return self._stride == expected

    def contiguous(self, memory_format=None):
        """Return contiguous tensor (copy if not already contiguous)."""
        if self.is_contiguous():
            return self
        # Force a contiguous copy
        arr = self.numpy()
        arr = np.ascontiguousarray(arr)
        result = Tensor(arr, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

        # Track autograd if needed
        if self._requires_grad:
            from ._autograd import is_grad_enabled
            from ._autograd.node import AccumulateGrad
            from ._autograd.functions import ContiguousBackward

            if is_grad_enabled():
                grad_fn = ContiguousBackward()

                if self._grad_fn is not None:
                    grad_fn._next_functions = ((self._grad_fn, 0),)
                else:
                    acc_grad = AccumulateGrad(self)
                    grad_fn._next_functions = ((acc_grad, 0),)

                result._grad_fn = grad_fn

        return result

    # --- Conversion ---

    def _to_numpy_flat(self):
        """Get the raw flat numpy data from storage."""
        return self._storage._ms_tensor.asnumpy()

    def numpy(self):
        """Convert to numpy array. Must not require grad."""
        # Check if this is a meta tensor
        if self.device.type == "meta":
            raise RuntimeError("Cannot access data of tensor on meta device")
        flat = self._to_numpy_flat()
        if self.is_contiguous() and self._storage_offset == 0:
            return flat[:self.numel()].reshape(self._shape)
        # Non-contiguous: use stride-based indexing
        return self._strided_numpy(flat)

    def _strided_numpy(self, flat):
        """Extract data via strides from flat buffer."""
        result = np.empty(self._shape, dtype=flat.dtype)
        max_idx = len(flat) - 1
        for idx in np.ndindex(*self._shape):
            flat_idx = self._storage_offset + sum(i * s for i, s in zip(idx, self._stride))
            flat_idx = min(flat_idx, max_idx)
            result[idx] = flat[flat_idx]
        return result

    def asnumpy(self):
        """Convert to numpy array (alias for numpy() for MindSpore compatibility)."""
        return self.numpy()

    def item(self):
        """Extract scalar value."""
        if self.numel() != 1:
            raise ValueError(f"only one element tensors can be converted to Python scalars, got {self.numel()}")
        return self.numpy().item()

    def tolist(self):
        return self.numpy().tolist()

    def to_mindspore(self):
        """Convert to MindSpore tensor (for interop)."""
        arr = self.numpy()
        return mindspore.Tensor(arr)

    # --- Repr ---

    def __repr__(self):
        # Handle meta tensors specially
        if self.device.type == "meta":
            return f"tensor(..., device='{self.device}', size={self.shape})"
        arr = self.numpy()
        data_str = np.array2string(arr, separator=', ', prefix='tensor(')
        if self._requires_grad:
            return f"tensor({data_str}, requires_grad=True)"
        if self._dtype is not dtype_mod.float32:
            return f"tensor({data_str}, dtype={self._dtype})"
        return f"tensor({data_str})"

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]

    def __bool__(self):
        if self.numel() != 1:
            raise RuntimeError(
                f"Boolean value of Tensor with more than one element is ambiguous"
            )
        return builtins_bool(self.item())

    def __contains__(self, item):
        """Check if item is contained in this tensor."""
        if isinstance(item, Tensor):
            item = item.item() if item.numel() == 1 else item.numpy()
        arr = self.numpy()
        return item in arr

    def __float__(self):
        return builtins_float(self.item())

    def __int__(self):
        return builtins_int(self.item())

    # --- View operations ---

    def view(self, *shape, dtype=None):
        """Return a view with a different shape or dtype.

        If dtype is provided (or shape[0] is a dtype), reinterprets the raw bytes
        as the new dtype without copying data. This is used by safetensors.

        Args:
            *shape: New shape dimensions, or a single dtype for reinterpretation
            dtype: New dtype for reinterpretation (alternative to passing as shape[0])

        Returns:
            Tensor with new shape or dtype
        """
        from . import _dtype as dtype_mod

        # Check if first argument is a dtype (for dtype reinterpretation)
        if len(shape) == 1 and isinstance(shape[0], dtype_mod.DType):
            dtype = shape[0]
            shape = None
        elif dtype is not None:
            shape = None

        # Handle dtype reinterpretation (view as different dtype)
        if dtype is not None:
            if not self.is_contiguous():
                raise RuntimeError("view() with dtype requires contiguous tensor")

            # Get raw bytes and reinterpret
            old_dtype = self._dtype
            new_dtype = dtype

            # Calculate new shape based on element sizes
            old_size = old_dtype.itemsize
            new_size = new_dtype.itemsize
            total_bytes = self.numel() * old_size

            if total_bytes % new_size != 0:
                raise RuntimeError(
                    f"view dtype from {old_dtype} to {new_dtype} is not supported: "
                    f"total bytes ({total_bytes}) must be divisible by new element size ({new_size})"
                )

            new_numel = total_bytes // new_size

            # Handle meta tensors - return meta tensor with new dtype
            if self._storage._ms_tensor is None:
                from ._creation import empty
                result = empty((new_numel,), dtype=new_dtype, device='meta')
                return result

            # Get raw bytes from storage
            storage_data = self._storage._ms_tensor.asnumpy()
            raw_bytes = storage_data.tobytes()

            # Reinterpret as new dtype
            np_dtype = dtype_mod.dtype_to_numpy(new_dtype)
            if np_dtype is not None:
                new_arr = np.frombuffer(raw_bytes, dtype=np_dtype)
            else:
                # Handle bfloat16
                if new_dtype == dtype_mod.bfloat16:
                    try:
                        import ml_dtypes
                        new_arr = np.frombuffer(raw_bytes, dtype=ml_dtypes.bfloat16)
                    except ImportError:
                        new_arr = np.frombuffer(raw_bytes, dtype=np.uint16)
                else:
                    raise TypeError(f"Unsupported dtype for view: {new_dtype}")

            # Create new tensor with reinterpreted data
            from ._creation import Tensor as TensorCreate
            return Tensor(new_arr.copy(), dtype=new_dtype, device=str(self._device),
                         requires_grad=self._requires_grad)

        # Original shape-based view logic
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if not self.is_contiguous():
            raise RuntimeError("view size is not compatible with input tensor's "
                             "size and stride (at least one dimension spans "
                             "across two contiguous subspaces). Use .reshape() instead.")

        # Resolve -1
        new_shape = _resolve_neg_one(shape, self.numel())
        new_stride = _compute_strides(new_shape)

        result = Tensor(
            _storage=self._storage,
            _shape=new_shape,
            _stride=new_stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

        # Track autograd if needed
        if self._requires_grad:
            from ._autograd import is_grad_enabled
            from ._autograd.node import AccumulateGrad
            from ._autograd.functions import ViewBackward

            if is_grad_enabled():
                grad_fn = ViewBackward()
                grad_fn._input_shape = self._shape

                if self._grad_fn is not None:
                    grad_fn._next_functions = ((self._grad_fn, 0),)
                else:
                    acc_grad = AccumulateGrad(self)
                    grad_fn._next_functions = ((acc_grad, 0),)

                result._grad_fn = grad_fn

        return result

    def reshape(self, *shape):
        """Reshape tensor. Returns view if possible, copy otherwise."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        new_shape = _resolve_neg_one(shape, self.numel())

        if self.is_contiguous():
            return self.view(*new_shape)
        # Non-contiguous: must copy
        return self.contiguous().view(*new_shape)

    def transpose(self, dim0, dim1):
        """Swap two dimensions. Returns a view with autograd support."""
        ndim = self.dim()
        # Normalize negative dims
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim

        # Swap shape and stride at the two dimensions
        new_shape = list(self._shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

        new_stride = list(self._stride)
        new_stride[dim0], new_stride[dim1] = new_stride[dim1], new_stride[dim0]

        result = Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

        # Track autograd if needed
        if self._requires_grad:
            from ._autograd import is_grad_enabled
            from ._autograd.node import AccumulateGrad
            from ._autograd.functions import TransposeBackward

            if is_grad_enabled():
                grad_fn = TransposeBackward()
                grad_fn._dims = (dim0, dim1)

                if self._grad_fn is not None:
                    grad_fn._next_functions = ((self._grad_fn, 0),)
                else:
                    acc_grad = AccumulateGrad(self)
                    grad_fn._next_functions = ((acc_grad, 0),)

                result._grad_fn = grad_fn

        return result

    def t(self):
        """Transpose 2D tensor."""
        if self.dim() != 2:
            raise RuntimeError(f"t() expects a 2D tensor, but self is {self.dim()}D")
        return self.transpose(0, 1)

    def permute(self, *dims):
        """Permute dimensions. Returns a view."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])

        ndim = self.dim()
        dims = tuple(d % ndim if d < 0 else d for d in dims)

        new_shape = tuple(self._shape[d] for d in dims)
        new_stride = tuple(self._stride[d] for d in dims)

        result = Tensor(
            _storage=self._storage,
            _shape=new_shape,
            _stride=new_stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

        # Track autograd if needed
        if self._requires_grad:
            from ._autograd import is_grad_enabled
            from ._autograd.node import AccumulateGrad
            from ._autograd.functions import PermuteBackward

            if is_grad_enabled():
                grad_fn = PermuteBackward()
                grad_fn._dims = dims

                if self._grad_fn is not None:
                    grad_fn._next_functions = ((self._grad_fn, 0),)
                else:
                    acc_grad = AccumulateGrad(self)
                    grad_fn._next_functions = ((acc_grad, 0),)

                result._grad_fn = grad_fn

        return result

    def unsqueeze(self, dim):
        """Insert a dimension of size 1."""
        from ._autograd import is_grad_enabled
        from ._autograd.node import Node, AccumulateGrad

        if dim < 0:
            dim += self.dim() + 1

        new_shape = list(self._shape)
        new_shape.insert(dim, 1)

        new_stride = list(self._stride)
        # Stride for new dim: product of shape * stride at that position
        if dim < len(self._stride):
            new_stride.insert(dim, self._shape[dim] * self._stride[dim] if dim < len(self._shape) else 1)
        else:
            new_stride.insert(dim, 1)

        result = Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

        # Set up autograd if needed
        if is_grad_enabled() and self._requires_grad:
            class UnsqueezeBackward(Node):
                def __init__(self, squeeze_dim):
                    super().__init__()
                    self._dim = squeeze_dim
                    self._name = "UnsqueezeBackward"

                def backward(self, grad_outputs):
                    grad = grad_outputs[0]
                    # Squeeze back to original shape
                    return (grad.squeeze(self._dim),)

            grad_fn = UnsqueezeBackward(dim)
            if self.grad_fn is not None:
                grad_fn._next_functions = ((self.grad_fn, 0),)
            else:
                grad_fn._next_functions = ((AccumulateGrad(self), 0),)
            result._grad_fn = grad_fn

        return result

    def squeeze(self, dim=None):
        """Remove dimensions of size 1."""
        from ._autograd import is_grad_enabled
        from ._autograd.node import Node, AccumulateGrad

        if dim is not None:
            if dim < 0:
                dim += self.dim()
            if self._shape[dim] != 1:
                return self
            new_shape = list(self._shape)
            new_stride = list(self._stride)
            new_shape.pop(dim)
            new_stride.pop(dim)
            squeezed_dims = [dim]
        else:
            new_shape = []
            new_stride = []
            squeezed_dims = []
            for i, (s, st) in enumerate(zip(self._shape, self._stride)):
                if s != 1:
                    new_shape.append(s)
                    new_stride.append(st)
                else:
                    squeezed_dims.append(i)

        result = Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

        # Set up autograd if needed
        if is_grad_enabled() and self._requires_grad and squeezed_dims:
            class SqueezeBackward(Node):
                def __init__(self, original_shape, dims):
                    super().__init__()
                    self._original_shape = original_shape
                    self._dims = dims
                    self._name = "SqueezeBackward"

                def backward(self, grad_outputs):
                    grad = grad_outputs[0]
                    # Unsqueeze back to original shape
                    for d in sorted(self._dims):
                        grad = grad.unsqueeze(d)
                    return (grad,)

            grad_fn = SqueezeBackward(self._shape, squeezed_dims)
            if self.grad_fn is not None:
                grad_fn._next_functions = ((self.grad_fn, 0),)
            else:
                grad_fn._next_functions = ((AccumulateGrad(self), 0),)
            result._grad_fn = grad_fn

        return result

    def expand(self, *sizes):
        """Expand tensor to a larger size. Returns a view with stride 0 for expanded dims."""
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])

        # Pad self shape/stride with leading 1s to match len(sizes)
        ndim_diff = len(sizes) - self.dim()
        old_shape = (1,) * ndim_diff + self._shape
        old_stride = (0,) * ndim_diff + self._stride

        new_shape = []
        new_stride = []
        for i, (s_new, s_old, st_old) in enumerate(zip(sizes, old_shape, old_stride)):
            if s_new == -1:
                s_new = s_old
            if s_old == 1 and s_new != 1:
                new_shape.append(s_new)
                new_stride.append(0)
            elif s_old == s_new:
                new_shape.append(s_old)
                new_stride.append(st_old)
            else:
                raise RuntimeError(
                    f"The expanded size of the tensor ({s_new}) must match the existing "
                    f"size ({s_old}) at non-singleton dimension {i}."
                )

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten dimensions from start_dim to end_dim."""
        ndim = self.dim()
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim

        if start_dim == end_dim:
            return self

        new_shape = (
            self._shape[:start_dim]
            + (math.prod(self._shape[start_dim:end_dim + 1]),)
            + self._shape[end_dim + 1:]
        )
        return self.reshape(*new_shape)

    # --- Indexing ---

    def __getitem__(self, key):
        """Index the tensor. Supports int, slice, None, Ellipsis, bool/int tensors."""
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Check for advanced indexing (bool/int tensors, lists of ints, numpy arrays)
        # Lists of slices are not advanced indexing - they're basic indexing
        def is_advanced_index(k):
            if isinstance(k, Tensor):
                return True
            elif isinstance(k, np.ndarray):
                return True
            elif isinstance(k, list):
                # A list of ONLY slices is basic indexing (unpack to tuple)
                # A list with any integers is advanced indexing (integer array)
                if all(isinstance(item, slice) for item in k):
                    return False  # List of slices only - basic indexing
                return True  # List contains integers or other - advanced indexing
            return False

        has_advanced = any(is_advanced_index(k) for k in key)

        if has_advanced:
            return self._advanced_getitem(key)

        # Convert any list of slices to tuple for basic indexing
        processed_key = []
        for k in key:
            if isinstance(k, list) and all(isinstance(item, slice) for item in k):
                # This is a list of slices only - unpack it into the key
                processed_key.extend(k)
            else:
                processed_key.append(k)
        key = tuple(processed_key)

        # Use gradient-tracking version if requires_grad
        if self._requires_grad:
            return self._basic_getitem_with_grad(key)

        return self._basic_getitem(key)

    def _basic_getitem(self, key):
        """Basic indexing: int, slice, None, Ellipsis. Returns a view."""
        # Expand Ellipsis
        key = _expand_ellipsis(key, self.dim())

        new_shape = []
        new_stride = []
        offset = self._storage_offset
        src_dim = 0

        for k in key:
            if k is None:
                # Insert new dimension of size 1
                new_shape.append(1)
                if src_dim < len(self._stride):
                    new_stride.append(self._stride[src_dim] * self._shape[src_dim] if src_dim < len(self._shape) else 1)
                else:
                    new_stride.append(1)
            elif isinstance(k, builtins_int):
                # Select one index along this dim - removes the dim
                dim_size = self._shape[src_dim]
                if k < -dim_size or k >= dim_size:
                    raise IndexError(f"index {k} is out of bounds for dimension {src_dim} with size {dim_size}")
                if k < 0:
                    k += dim_size
                offset += k * self._stride[src_dim]
                src_dim += 1
            elif isinstance(k, slice):
                start, stop, step = k.indices(self._shape[src_dim])
                length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
                offset += start * self._stride[src_dim]
                new_shape.append(length)
                new_stride.append(self._stride[src_dim] * step)
                src_dim += 1
            else:
                raise TypeError(f"Unsupported index type: {type(k)}")

        # Remaining dimensions
        while src_dim < self.dim():
            new_shape.append(self._shape[src_dim])
            new_stride.append(self._stride[src_dim])
            src_dim += 1

        if len(new_shape) == 0:
            # Scalar result
            new_shape = ()
            new_stride = ()

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def _basic_getitem_with_grad(self, key):
        """Basic indexing with gradient tracking."""
        from ._autograd import is_grad_enabled
        from ._autograd.node import AccumulateGrad

        result = self._basic_getitem(key)

        # Track autograd if needed
        if is_grad_enabled() and self._requires_grad:
            from ._autograd.functions import SelectBackward
            grad_fn = SelectBackward()
            grad_fn._input_shape = self._shape
            grad_fn._key = key

            if self._grad_fn is not None:
                grad_fn._next_functions = ((self._grad_fn, 0),)
            else:
                acc_grad = AccumulateGrad(self)
                grad_fn._next_functions = ((acc_grad, 0),)

            result._grad_fn = grad_fn
            result._requires_grad = True

        return result

    def _advanced_getitem(self, key):
        """Advanced indexing: bool/int tensors, lists, numpy arrays. Returns a copy."""
        from ._autograd import is_grad_enabled
        from ._autograd.node import Node, AccumulateGrad

        # Convert to numpy for advanced indexing
        arr = self.numpy()
        np_key = []
        original_key = key  # Keep original for backward
        for k in key:
            if isinstance(k, Tensor):
                k_np = k.numpy()
                # Handle different dtype kinds
                if k_np.dtype.kind == 'b':
                    # Boolean indexing - use as-is
                    np_key.append(k_np)
                elif k_np.dtype.kind == 'f':
                    # Float to integer index
                    k_np = k_np.astype(np.intp)
                    np_key.append(k_np)
                elif k_np.dtype.kind == 'i' or k_np.dtype.kind == 'u':
                    # Integer types - convert to numpy integer
                    k_np = k_np.astype(np.intp)
                    np_key.append(k_np)
                else:
                    np_key.append(k_np)
            elif isinstance(k, np.ndarray):
                # NumPy array indexing
                if k.dtype.kind == 'b':
                    np_key.append(k)
                elif k.dtype.kind in ('i', 'u'):
                    np_key.append(k.astype(np.intp))
                elif k.dtype.kind == 'f':
                    np_key.append(k.astype(np.intp))
                else:
                    np_key.append(k)
            elif isinstance(k, list):
                # Convert list to numpy array for indexing
                k_np = np.array(k)
                if k_np.dtype.kind in ('i', 'u', 'f'):
                    k_np = k_np.astype(np.intp)
                np_key.append(k_np)
            else:
                # Slices, integers, None, Ellipsis - pass through
                np_key.append(k)
        result_arr = arr[tuple(np_key)]
        result = Tensor(result_arr, dtype=self._dtype, device=str(self._device),
                       requires_grad=self._requires_grad)

        # Set up autograd for embedding-style indexing
        if is_grad_enabled() and self._requires_grad:
            class IndexBackward(Node):
                def __init__(self, input_shape, indices, input_dtype):
                    super().__init__()
                    self._input_shape = input_shape
                    self._indices = indices
                    self._input_dtype = input_dtype
                    self._name = "IndexBackward"

                def backward(self, grad_outputs):
                    grad = grad_outputs[0]
                    # Create zero gradient of input shape
                    grad_input = np.zeros(self._input_shape, dtype=np.float32)
                    # Scatter add the gradient
                    np.add.at(grad_input, tuple(self._indices), grad.numpy())
                    return (Tensor(grad_input, dtype=self._input_dtype),)

            grad_fn = IndexBackward(self._shape, np_key, self._dtype)
            if self.grad_fn is not None:
                grad_fn._next_functions = ((self.grad_fn, 0),)
            else:
                grad_fn._next_functions = ((AccumulateGrad(self), 0),)
            result._grad_fn = grad_fn

        return result

    def __setitem__(self, key, value):
        """Set elements by index. Mutates in-place."""
        # Get numpy data, modify, write back
        arr = self.numpy().copy()

        if isinstance(key, Tensor):
            np_key = key.numpy()
        elif isinstance(key, tuple):
            np_key = tuple(
                k.numpy() if isinstance(k, Tensor) else k
                for k in key
            )
        else:
            np_key = key

        if isinstance(value, Tensor):
            arr[np_key] = value.numpy()
        else:
            arr[np_key] = value

        # Write back to storage
        flat = arr.ravel()
        self._storage._ms_tensor = mindspore.Tensor(flat)
        self._storage._size = len(flat)
        self._version += 1

    # --- Arithmetic methods ---

    def add(self, other, *, alpha=1):
        from . import add as torch_add, mul as torch_mul
        if alpha != 1:
            other = torch_mul(other, alpha)
        return torch_add(self, other)

    def sub(self, other, *, alpha=1):
        from . import sub as torch_sub, mul as torch_mul
        if alpha != 1:
            other = torch_mul(other, alpha)
        return torch_sub(self, other)

    def mul(self, other):
        from . import mul as torch_mul
        return torch_mul(self, other)

    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        from . import baddbmm as torch_baddbmm
        return torch_baddbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    def div(self, other, *, rounding_mode=None):
        from . import div as torch_div
        return torch_div(self, other, rounding_mode=rounding_mode)

    def neg(self):
        from . import neg as torch_neg
        return torch_neg(self)

    def abs(self):
        from . import abs as torch_abs
        return torch_abs(self)

    def pow(self, exponent):
        from . import pow as torch_pow
        return torch_pow(self, exponent)

    def exp(self):
        from . import exp as torch_exp
        return torch_exp(self)

    def log(self):
        from . import log as torch_log
        return torch_log(self)

    def sqrt(self):
        from . import sqrt as torch_sqrt
        return torch_sqrt(self)

    def matmul(self, other):
        from ._dispatch import dispatch
        return dispatch("matmul", self, other)

    # --- Operator overloads ---

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        from . import add as torch_add
        return torch_add(other, self)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        from . import sub as torch_sub
        return torch_sub(other, self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        from . import mul as torch_mul
        return torch_mul(other, self)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        from . import div as torch_div
        return torch_div(other, self)

    def __neg__(self):
        return self.neg()

    def __abs__(self):
        return self.abs()

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __matmul__(self, other):
        return self.matmul(other)

    def __invert__(self):
        """Bitwise NOT (~). For bool tensors, returns logical not."""
        arr = self.numpy()
        if arr.dtype == np.bool_:
            result = ~arr
        else:
            result = np.bitwise_not(arr)
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def __or__(self, other):
        """Bitwise OR (|)."""
        arr = self.numpy()
        other_arr = other.numpy() if isinstance(other, Tensor) else np.asarray(other)
        result = np.bitwise_or(arr, other_arr)
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def __ror__(self, other):
        """Reflected bitwise OR (|)."""
        other_arr = other.numpy() if isinstance(other, Tensor) else np.asarray(other)
        result = np.bitwise_or(other_arr, self.numpy())
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def __and__(self, other):
        """Bitwise AND (&)."""
        arr = self.numpy()
        other_arr = other.numpy() if isinstance(other, Tensor) else np.asarray(other)
        result = np.bitwise_and(arr, other_arr)
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def __rand__(self, other):
        """Reflected bitwise AND (&)."""
        other_arr = other.numpy() if isinstance(other, Tensor) else np.asarray(other)
        result = np.bitwise_and(other_arr, self.numpy())
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def __xor__(self, other):
        """Bitwise XOR (^)."""
        arr = self.numpy()
        other_arr = other.numpy() if isinstance(other, Tensor) else np.asarray(other)
        result = np.bitwise_xor(arr, other_arr)
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def __rxor__(self, other):
        """Reflected bitwise XOR (^)."""
        other_arr = other.numpy() if isinstance(other, Tensor) else np.asarray(other)
        result = np.bitwise_xor(other_arr, self.numpy())
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    # --- Reduction methods ---

    def sum(self, dim=None, keepdim=False, *, dtype=None):
        from . import sum as torch_sum
        return torch_sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def mean(self, dim=None, keepdim=False, *, dtype=None, axis=None):
        from . import mean as torch_mean
        if axis is not None and dim is None:
            dim = axis
        return torch_mean(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def max(self, dim=None, keepdim=False):
        from . import max as torch_max
        return torch_max(self, dim=dim, keepdim=keepdim)

    def min(self, dim=None, keepdim=False):
        from . import min as torch_min
        return torch_min(self, dim=dim, keepdim=keepdim)

    def amax(self, dim=None, keepdim=False):
        """Returns the maximum value along the specified dimension(s).

        Unlike max(), only returns values, not indices.
        """
        arr = self.numpy()
        if dim is None:
            result = np.max(arr)
        else:
            result = np.max(arr, axis=dim, keepdims=keepdim)
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def amin(self, dim=None, keepdim=False):
        """Returns the minimum value along the specified dimension(s).

        Unlike min(), only returns values, not indices.
        """
        arr = self.numpy()
        if dim is None:
            result = np.min(arr)
        else:
            result = np.min(arr, axis=dim, keepdims=keepdim)
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def prod(self, dim=None, keepdim=False, *, dtype=None):
        from . import prod as torch_prod
        return torch_prod(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def argmax(self, dim=None, keepdim=False):
        from . import argmax as torch_argmax
        return torch_argmax(self, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        from . import argmin as torch_argmin
        return torch_argmin(self, dim=dim, keepdim=keepdim)

    # --- Comparison methods ---

    def eq(self, other):
        from . import eq as torch_eq
        return torch_eq(self, other)

    def ne(self, other):
        from . import ne as torch_ne
        return torch_ne(self, other)

    def gt(self, other):
        from . import gt as torch_gt
        return torch_gt(self, other)

    def lt(self, other):
        from . import lt as torch_lt
        return torch_lt(self, other)

    def ge(self, other):
        from . import ge as torch_ge
        return torch_ge(self, other)

    def le(self, other):
        from . import le as torch_le
        return torch_le(self, other)

    # --- Comparison operators ---

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __hash__(self):
        """Enable hashing based on object identity.

        Note: This makes tensors hashable but only by identity,
        not by value. Two tensors with identical values will have
        different hashes if they are different objects.
        """
        return id(self)

    def __gt__(self, other):
        return self.gt(other)

    def __lt__(self, other):
        return self.lt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __le__(self, other):
        return self.le(other)

    # --- Autograd methods ---

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        """Compute gradients of this tensor w.r.t. graph leaves.

        Args:
            gradient: Gradient w.r.t. this tensor. Required for non-scalar tensors.
            retain_graph: If True, graph is retained for future backward calls.
            create_graph: If True, graph of gradient computation is constructed.
        """
        from ._autograd import backward as autograd_backward
        autograd_backward(self, gradient, retain_graph=retain_graph, create_graph=create_graph)

    def register_hook(self, hook):
        """Register a backward hook on the tensor.

        The hook will be called every time a gradient with respect to the
        tensor is computed. The hook should have the signature:
            hook(grad) -> Tensor or None

        Returns a handle that can be used to remove the hook.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )

        hook_id = self._hook_counter
        self._hook_counter += 1
        self._hooks[hook_id] = hook

        return RemovableHandle(self._hooks, hook_id)

    def _call_hooks(self, grad):
        """Call all registered hooks on a gradient."""
        for hook in self._hooks.values():
            result = hook(grad)
            if result is not None:
                grad = result
        return grad

    def zero_grad_(self):
        """Zero the gradient of this tensor in-place."""
        self._grad = None
        return self

    # --- In-place initialization methods ---

    def normal_(self, mean=0.0, std=1.0):
        """Fill tensor with values from normal distribution N(mean, std) in-place."""
        import numpy as np
        import mindspore
        # Meta tensors have no actual storage - just return self
        if self._storage._ms_tensor is None:
            return self
        # Get current data and modify in place
        arr = np.random.normal(mean, std, self._shape).astype(np.float32)
        # Calculate the storage indices we need to update
        storage_arr = self._storage._ms_tensor.asnumpy().copy()
        # Ensure storage is large enough
        required_size = self._storage_offset + self.numel()
        if len(storage_arr) < required_size:
            # Resize storage
            new_storage = np.zeros(required_size, dtype=np.float32)
            new_storage[:len(storage_arr)] = storage_arr
            storage_arr = new_storage
        # Update the portion of storage that this tensor uses
        storage_arr[self._storage_offset:self._storage_offset + self.numel()] = arr.ravel()
        self._storage._ms_tensor = mindspore.Tensor(storage_arr)
        self._version += 1
        return self

    def uniform_(self, a=0.0, b=1.0):
        """Fill tensor with values from uniform distribution U(a, b) in-place."""
        import numpy as np
        import mindspore
        # Meta tensors have no actual storage - just return self
        if self._storage._ms_tensor is None:
            return self
        arr = np.random.uniform(a, b, self._shape).astype(np.float32)
        storage_arr = self._storage._ms_tensor.asnumpy().copy()
        required_size = self._storage_offset + self.numel()
        if len(storage_arr) < required_size:
            new_storage = np.zeros(required_size, dtype=np.float32)
            new_storage[:len(storage_arr)] = storage_arr
            storage_arr = new_storage
        storage_arr[self._storage_offset:self._storage_offset + self.numel()] = arr.ravel()
        self._storage._ms_tensor = mindspore.Tensor(storage_arr)
        self._version += 1
        return self

    def fill_(self, value):
        """Fill tensor with a constant value in-place."""
        from ._dispatch import dispatch
        return dispatch("fill_", self, value)

    def zero_(self):
        """Fill tensor with zeros in-place."""
        from ._dispatch import dispatch
        return dispatch("zero_", self)

    def ones_(self):
        """Fill tensor with ones in-place."""
        return self.fill_(1.0)

    def add_(self, other, *, alpha=1):
        """In-place element-wise addition."""
        from ._dispatch import dispatch
        if alpha != 1:
            other = other * alpha
        return dispatch("add_", self, other)

    def sub_(self, other, *, alpha=1):
        """In-place element-wise subtraction."""
        from ._dispatch import dispatch
        if alpha != 1:
            other = other * alpha
        return dispatch("sub_", self, other)

    def mul_(self, other):
        """In-place element-wise multiplication."""
        from ._dispatch import dispatch
        return dispatch("mul_", self, other)

    def div_(self, other, *, rounding_mode=None):
        """In-place element-wise division."""
        from ._dispatch import dispatch
        return dispatch("div_", self, other)

    def copy_(self, src, non_blocking=False):
        """Copy data from src tensor into self."""
        from ._dispatch import dispatch
        return dispatch("copy_", self, src, non_blocking)

    def bernoulli_(self, p=0.5):
        """Fill tensor with values from Bernoulli distribution in-place."""
        import numpy as np
        import mindspore
        arr = (np.random.random(self._shape) < p).astype(np.float32)
        flat = arr.ravel()
        self._storage._ms_tensor = mindspore.Tensor(flat)
        self._version += 1
        return self

    def random_(self, from_=0, to=None, *, generator=None):
        """Fill tensor with random integers in-place."""
        import numpy as np
        import mindspore
        if to is None:
            # For float types, fill with [0, 1)
            arr = np.random.random(self._shape).astype(np.float32)
        else:
            arr = np.random.randint(from_, to, self._shape).astype(np.float32)
        flat = arr.ravel()
        self._storage._ms_tensor = mindspore.Tensor(flat)
        self._version += 1
        return self

    def copy_(self, src, non_blocking=False):
        """Copy data from src tensor to self in-place."""
        import numpy as np
        import mindspore
        src_arr = src.numpy()
        # Reshape if needed
        if src_arr.shape != self._shape:
            src_arr = np.broadcast_to(src_arr, self._shape)
        flat = src_arr.ravel().astype(np.float32)
        self._storage._ms_tensor = mindspore.Tensor(flat)
        self._version += 1
        return self

    def detach(self):
        """Returns a new Tensor detached from the current graph.

        The result will never require gradient.
        """
        return Tensor(
            _storage=self._storage,
            _shape=self._shape,
            _stride=self._stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=False,
        )

    def detach_(self):
        """Detach this tensor from the computation graph in-place.

        Sets requires_grad to False and clears grad_fn.
        """
        self._requires_grad = False
        self._grad_fn = None
        return self

    def requires_grad_(self, requires_grad: bool = True):
        """Change if this tensor requires gradients, in-place.

        Returns self.
        """
        self._requires_grad = requires_grad
        return self

    # --- Additional tensor manipulation methods ---

    def clone(self, *, memory_format=None):
        """Create a copy of tensor with new storage."""
        from ._dispatch import dispatch
        return dispatch("clone", self)

    def repeat(self, *sizes):
        """Repeat tensor along dimensions."""
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        arr = self.numpy()
        result = np.tile(arr, sizes)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def masked_fill(self, mask, value):
        """Fill elements where mask is True with value."""
        arr = self.numpy().copy()
        mask_np = mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask)
        arr[mask_np] = value
        return Tensor(arr, dtype=self._dtype, device=str(self._device))

    def masked_fill_(self, mask, value):
        """Fill elements where mask is True with value, in-place."""
        arr = self.numpy()
        mask_np = mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask)
        arr[mask_np] = value
        # Write back to storage
        flat = arr.ravel()
        import mindspore
        self._storage._ms_tensor = mindspore.Tensor(flat)
        self._version += 1
        return self

    def clamp(self, min=None, max=None):
        """Clamp values to range [min, max]."""
        from . import clamp as torch_clamp
        return torch_clamp(self, min=min, max=max)

    def clamp_(self, min=None, max=None):
        """Clamp values to range [min, max], in-place."""
        arr = np.clip(self.numpy(), min, max)
        flat = arr.ravel()
        import mindspore
        self._storage._ms_tensor = mindspore.Tensor(flat)
        self._version += 1
        return self

    def all(self, dim=None, keepdim=False):
        """Test if all elements evaluate to True."""
        from . import all as torch_all
        return torch_all(self, dim=dim, keepdim=keepdim)

    def any(self, dim=None, keepdim=False):
        """Test if any element evaluates to True."""
        from . import any as torch_any
        return torch_any(self, dim=dim, keepdim=keepdim)

    def cumsum(self, dim):
        """Cumulative sum along a dimension."""
        arr = self.numpy()
        result = np.cumsum(arr, axis=dim)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def cumprod(self, dim):
        """Cumulative product along a dimension."""
        arr = self.numpy()
        result = np.cumprod(arr, axis=dim)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    # --- Type conversion methods ---

    def long(self):
        """Convert tensor to int64 dtype."""
        arr = self.numpy().astype(np.int64)
        return Tensor(arr, dtype=dtype_mod.int64, device=str(self._device),
                      requires_grad=False)

    def int(self):
        """Convert tensor to int32 dtype."""
        arr = self.numpy().astype(np.int32)
        return Tensor(arr, dtype=dtype_mod.int32, device=str(self._device),
                      requires_grad=False)

    def float(self):
        """Convert tensor to float32 dtype."""
        arr = self.numpy().astype(np.float32)
        return Tensor(arr, dtype=dtype_mod.float32, device=str(self._device),
                      requires_grad=self._requires_grad)

    def double(self):
        """Convert tensor to float64 dtype."""
        arr = self.numpy().astype(np.float64)
        return Tensor(arr, dtype=dtype_mod.float64, device=str(self._device),
                      requires_grad=self._requires_grad)

    def half(self):
        """Convert tensor to float16 dtype."""
        arr = self.numpy().astype(np.float16)
        return Tensor(arr, dtype=dtype_mod.float16, device=str(self._device),
                      requires_grad=self._requires_grad)

    def bool(self):
        """Convert tensor to bool dtype."""
        arr = self.numpy().astype(np.bool_)
        return Tensor(arr, dtype=dtype_mod.bool, device=str(self._device),
                      requires_grad=False)

    def to(self, *args, **kwargs):
        """Move tensor to device and/or change dtype."""
        from ._autograd import is_grad_enabled
        from ._autograd.node import Node, AccumulateGrad

        # Parse args: can be (device), (dtype), (device, dtype), or keyword args
        dtype = kwargs.get('dtype', None)
        device = kwargs.get('device', None)

        for arg in args:
            if isinstance(arg, device_cls):
                device = arg
            elif isinstance(arg, str):
                if arg in ('cpu', 'cuda', 'mps'):
                    device = device_cls(arg)
                else:
                    # Might be a dtype string
                    pass
            elif isinstance(arg, dtype_mod.DType):
                dtype = arg
            elif arg is None:
                pass

        # Apply dtype conversion if needed
        if dtype is not None:
            # If same dtype, return self (no conversion needed)
            if dtype == self._dtype:
                return self

            np_dtype = dtype_mod.dtype_to_numpy(dtype)
            arr = self.numpy().astype(np_dtype)
            result = Tensor(arr, dtype=dtype, device=str(device or self._device),
                            requires_grad=self._requires_grad)

            # Set up autograd if needed
            if is_grad_enabled() and self._requires_grad:
                class ToBackward(Node):
                    def __init__(self, input_tensor, orig_dtype):
                        super().__init__()
                        self._input = input_tensor
                        self._orig_dtype = orig_dtype
                        self._name = "ToBackward"

                    def backward(self, grad_outputs):
                        grad = grad_outputs[0]
                        # Convert gradient back to original dtype
                        np_dtype = dtype_mod.dtype_to_numpy(self._orig_dtype)
                        grad_arr = grad.numpy().astype(np_dtype)
                        return (Tensor(grad_arr, dtype=self._orig_dtype),)

                grad_fn = ToBackward(self, self._dtype)
                if self.grad_fn is not None:
                    grad_fn._next_functions = ((self.grad_fn, 0),)
                else:
                    grad_fn._next_functions = ((AccumulateGrad(self), 0),)
                result._grad_fn = grad_fn

            return result

        # Device only change (no-op for CPU backend)
        return self

    def type_as(self, other):
        """Cast tensor to the same dtype as another tensor.

        Args:
            other: Tensor whose dtype to match

        Returns:
            Tensor with same dtype as other
        """
        return self.to(dtype=other.dtype)

    def type(self, dtype=None):
        """Return or set tensor type.

        Args:
            dtype: If provided, convert to this dtype. If None, return type string.

        Returns:
            String type name if dtype is None, else converted tensor.
        """
        if dtype is None:
            # Return type string
            dtype_name = str(self._dtype).replace('torch.', '')
            return f'torch.{dtype_name.capitalize()}Tensor'

        # Convert to dtype
        if isinstance(dtype, str):
            # Parse string like 'torch.FloatTensor'
            dtype_map = {
                'torch.FloatTensor': dtype_mod.float32,
                'torch.DoubleTensor': dtype_mod.float64,
                'torch.HalfTensor': dtype_mod.float16,
                'torch.LongTensor': dtype_mod.int64,
                'torch.IntTensor': dtype_mod.int32,
                'torch.ShortTensor': dtype_mod.int16,
                'torch.ByteTensor': dtype_mod.uint8,
                'torch.BoolTensor': dtype_mod.bool,
            }
            dtype = dtype_map.get(dtype, dtype_mod.float32)

        return self.to(dtype=dtype)

    # --- Factory methods (new_*) ---

    def new_ones(self, *size, dtype=None, device=None, requires_grad=False):
        """Create a new tensor filled with ones with same dtype/device."""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        target_dtype = dtype if dtype is not None else self._dtype
        target_device = device if device is not None else str(self._device)
        np_dtype = dtype_mod.dtype_to_numpy(target_dtype)
        arr = np.ones(size, dtype=np_dtype)
        return Tensor(arr, dtype=target_dtype, device=target_device, requires_grad=requires_grad)

    def new_zeros(self, *size, dtype=None, device=None, requires_grad=False):
        """Create a new tensor filled with zeros with same dtype/device."""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        target_dtype = dtype if dtype is not None else self._dtype
        target_device = device if device is not None else str(self._device)
        np_dtype = dtype_mod.dtype_to_numpy(target_dtype)
        arr = np.zeros(size, dtype=np_dtype)
        return Tensor(arr, dtype=target_dtype, device=target_device, requires_grad=requires_grad)

    def new_empty(self, *size, dtype=None, device=None, requires_grad=False):
        """Create a new uninitialized tensor with same dtype/device."""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        target_dtype = dtype if dtype is not None else self._dtype
        target_device = device if device is not None else str(self._device)
        np_dtype = dtype_mod.dtype_to_numpy(target_dtype)
        arr = np.empty(size, dtype=np_dtype)
        return Tensor(arr, dtype=target_dtype, device=target_device, requires_grad=requires_grad)

    def new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False):
        """Create a new tensor filled with fill_value with same dtype/device."""
        if isinstance(size, builtins_int):
            size = (size,)
        target_dtype = dtype if dtype is not None else self._dtype
        target_device = device if device is not None else str(self._device)
        np_dtype = dtype_mod.dtype_to_numpy(target_dtype)
        arr = np.full(size, fill_value, dtype=np_dtype)
        return Tensor(arr, dtype=target_dtype, device=target_device, requires_grad=requires_grad)

    def new_tensor(self, data, dtype=None, device=None, requires_grad=False):
        """Create a new tensor from data with same dtype/device."""
        target_dtype = dtype if dtype is not None else self._dtype
        target_device = device if device is not None else str(self._device)
        return Tensor(data, dtype=target_dtype, device=target_device, requires_grad=requires_grad)

    # --- Memory management methods (stubs for compatibility) ---

    def pin_memory(self, device=None):
        """Pin tensor to CPU memory (no-op for CPU backend).

        This is a no-op since MindSpore handles memory management internally.
        Returns self for compatibility with PyTorch API.
        """
        return self

    def is_pinned(self, device=None):
        """Check if tensor is pinned (always False for CPU backend)."""
        return False

    def cuda(self, device=None, non_blocking=False, memory_format=None):
        """Move tensor to CUDA device (no-op for CPU backend)."""
        # MindSpore handles device placement automatically
        return self

    def cpu(self, memory_format=None):
        """Move tensor to CPU (no-op since we're already on CPU)."""
        return self

    def is_cuda(self):
        """Check if tensor is on CUDA device."""
        return False

    def is_cpu(self):
        """Check if tensor is on CPU."""
        return True

    def retain_grad(self):
        """Retain gradient for non-leaf tensors.

        This allows the gradient to be retained for non-leaf tensors
        during backward pass.
        """
        if self.is_leaf():
            return  # Leaf tensors automatically retain gradient

        if not self.requires_grad:
            raise RuntimeError(
                "can only retain gradients on tensors that require grad"
            )

        # Track this via a flag
        self._retain_grad = True

        # Register this tensor with its grad_fn for gradient retention
        if self._grad_fn is not None:
            self._grad_fn.register_output_tensor(self)

    def is_leaf(self):
        """Check if tensor is a leaf in the computation graph.

        A tensor is a leaf if it was not created by an operation
        (i.e., it has no grad_fn).
        """
        return self._grad_fn is None

    # --- Shape info methods ---

    @property
    def T(self):
        """Transpose of tensor (swaps last two dimensions)."""
        if self.dim() < 2:
            return self
        return self.permute(*range(self.dim() - 2), self.dim() - 1, self.dim() - 2)

    @property
    def mT(self):
        """Transpose of last two dimensions (matrix transpose)."""
        return self.transpose(-2, -1)

    @property
    def mH(self):
        """Conjugate transpose of last two dimensions (Hermitian transpose)."""
        # For real tensors, this is the same as mT
        return self.transpose(-2, -1)

    def nbytes(self):
        """Return total bytes consumed by the tensor."""
        return self.numel() * self.element_size()

    def tobytes(self):
        """Return tensor data as bytes (for safetensors compatibility)."""
        return self.numpy().tobytes()

    def itemsize(self):
        """Alias for element_size()."""
        return self.element_size()

    # --- Additional math methods ---

    def tanh(self):
        """Element-wise hyperbolic tangent."""
        from . import tanh as torch_tanh
        return torch_tanh(self)

    def sigmoid(self):
        """Element-wise sigmoid."""
        arr = self.numpy()
        result = 1 / (1 + np.exp(-arr))
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def softmax(self, dim):
        """Softmax along a dimension."""
        arr = self.numpy()
        exp_arr = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
        result = exp_arr / np.sum(exp_arr, axis=dim, keepdims=True)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def log_softmax(self, dim):
        """Log-softmax along a dimension."""
        arr = self.numpy()
        max_arr = np.max(arr, axis=dim, keepdims=True)
        shifted = arr - max_arr
        log_sum_exp = max_arr + np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
        result = arr - log_sum_exp
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def relu(self):
        """Element-wise ReLU."""
        arr = self.numpy()
        result = np.maximum(arr, 0)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    # --- Additional operations ---

    def flip(self, dims):
        """Flip tensor along specified dimensions."""
        arr = self.numpy()
        if isinstance(dims, builtins_int):
            dims = [dims]
        result = np.flip(arr, axis=dims)
        return Tensor(result.copy(), dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def roll(self, shifts, dims=None):
        """Roll tensor along specified dimensions."""
        arr = self.numpy()
        result = np.roll(arr, shifts, axis=dims)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def unfold(self, dimension, size, step):
        """Return a view with dimension unfolded."""
        arr = self.numpy()

        # Get dimension size
        dim_size = arr.shape[dimension]

        # Calculate number of windows
        n_windows = (dim_size - size) // step + 1

        # Create unfolded array
        shape = list(arr.shape)
        shape[dimension] = n_windows
        shape.append(size)

        result = np.empty(shape, dtype=arr.dtype)

        for i in range(n_windows):
            idx = [slice(None)] * len(arr.shape)
            idx[dimension] = slice(i * step, i * step + size)
            result_idx = [slice(None)] * len(shape)
            result_idx[dimension] = i
            result[tuple(result_idx)] = np.take(arr, range(i * step, i * step + size), axis=dimension)

        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def gather(self, dim, index):
        """Gather values along an axis."""
        arr = self.numpy()
        index_arr = index.numpy() if isinstance(index, Tensor) else np.asarray(index)
        result = np.take_along_axis(arr, index_arr, axis=dim)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def scatter(self, dim, index, src):
        """Scatter values into tensor."""
        arr = self.numpy().copy()
        index_arr = index.numpy() if isinstance(index, Tensor) else np.asarray(index)
        src_arr = src.numpy() if isinstance(src, Tensor) else np.full(index_arr.shape, src)
        np.put_along_axis(arr, index_arr, src_arr, axis=dim)
        return Tensor(arr, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def scatter_(self, dim, index, src, reduce=None):
        """Scatter values into tensor in-place."""
        import mindspore
        arr = self.numpy().copy()
        index_arr = index.numpy() if isinstance(index, Tensor) else np.asarray(index)
        src_arr = src.numpy() if isinstance(src, Tensor) else np.full(index_arr.shape, src)
        np.put_along_axis(arr, index_arr, src_arr, axis=dim)
        self._storage._ms_tensor = mindspore.Tensor(arr.ravel())
        self._version += 1
        return self

    def index_select(self, dim, index):
        """Select values along a dimension using indices."""
        arr = self.numpy()
        index_arr = index.numpy() if isinstance(index, Tensor) else np.asarray(index)
        result = np.take(arr, index_arr, axis=dim)
        return Tensor(result, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def narrow(self, dim, start, length):
        """Return a narrowed version of tensor."""
        arr = self.numpy()
        slices = [slice(None)] * arr.ndim
        slices[dim] = slice(start, start + length)
        result = arr[tuple(slices)]
        return Tensor(result.copy(), dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    def split(self, split_size, dim=0):
        """Split tensor into chunks."""
        from . import split as torch_split
        return torch_split(self, split_size, dim=dim)

    def chunk(self, chunks, dim=0):
        """Split tensor into specified number of chunks."""
        from . import chunk as torch_chunk
        return torch_chunk(self, chunks, dim=dim)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        """Return top k values and indices."""
        from . import topk as torch_topk
        return torch_topk(self, k, dim=dim, largest=largest, sorted=sorted)

    def sort(self, dim=-1, descending=False):
        """Sort tensor along a dimension."""
        arr = self.numpy()
        if descending:
            sorted_indices = np.argsort(-arr, axis=dim)
        else:
            sorted_indices = np.argsort(arr, axis=dim)
        sorted_values = np.take_along_axis(arr, sorted_indices, axis=dim)
        return (
            Tensor(sorted_values, dtype=self._dtype, device=str(self._device)),
            Tensor(sorted_indices.astype(np.int64), dtype=dtype_mod.int64, device=str(self._device))
        )

    def argsort(self, dim=-1, descending=False):
        """Return indices that would sort the tensor."""
        arr = self.numpy()
        if descending:
            result = np.argsort(-arr, axis=dim)
        else:
            result = np.argsort(arr, axis=dim)
        return Tensor(result.astype(np.int64), dtype=dtype_mod.int64, device=str(self._device))

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        """Return unique elements."""
        arr = self.numpy()
        result = np.unique(arr, return_inverse=return_inverse, return_counts=return_counts, axis=dim)
        if return_inverse or return_counts:
            outputs = [Tensor(result[0], dtype=self._dtype, device=str(self._device))]
            for i, r in enumerate(result[1:]):
                outputs.append(Tensor(r.astype(np.int64), dtype=dtype_mod.int64, device=str(self._device)))
            return tuple(outputs)
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def nonzero(self, as_tuple=False):
        """Return indices of non-zero elements."""
        arr = self.numpy()
        indices = np.nonzero(arr)
        if as_tuple:
            return tuple(Tensor(idx.astype(np.int64), dtype=dtype_mod.int64, device=str(self._device))
                        for idx in indices)
        result = np.stack(indices, axis=1)
        return Tensor(result.astype(np.int64), dtype=dtype_mod.int64, device=str(self._device))

    def where(self, condition, other):
        """Return elements selected from self or other based on condition."""
        from . import where as torch_where
        return torch_where(condition, self, other)

    def norm(self, p=2, dim=None, keepdim=False, dtype=None):
        """Return the matrix norm or vector norm of a given tensor.

        Args:
            p: Order of norm (default: 2)
            dim: Dimension(s) to reduce over
            keepdim: Whether to keep the reduced dimensions
            dtype: Data type of output tensor
        """
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype_mod.dtype_to_numpy(dtype))

        if dim is None:
            result = np.linalg.norm(arr.flatten(), ord=p)
            if keepdim:
                result = np.array(result).reshape([1] * arr.ndim)
        else:
            result = np.linalg.norm(arr, ord=p, axis=dim, keepdims=keepdim)

        out_dtype = dtype if dtype is not None else dtype_mod.float32
        return Tensor(np.atleast_1d(result).astype(np.float32), dtype=out_dtype,
                      device=str(self._device), requires_grad=self._requires_grad)
