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

    # --- Shape / metadata ---

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

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

    def dim(self):
        return len(self._shape)

    def numel(self):
        result = 1
        for s in self._shape:
            result *= s
        return result

    def element_size(self):
        return self._dtype.itemsize

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
        return Tensor(arr, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    # --- Conversion ---

    def _to_numpy_flat(self):
        """Get the raw flat numpy data from storage."""
        return self._storage._ms_tensor.asnumpy()

    def numpy(self):
        """Convert to numpy array. Must not require grad."""
        flat = self._to_numpy_flat()
        if self.is_contiguous() and self._storage_offset == 0:
            return flat[:self.numel()].reshape(self._shape)
        # Non-contiguous: use stride-based indexing
        return self._strided_numpy(flat)

    def _strided_numpy(self, flat):
        """Extract data via strides from flat buffer."""
        result = np.empty(self._shape, dtype=flat.dtype)
        for idx in np.ndindex(*self._shape):
            flat_idx = self._storage_offset + sum(i * s for i, s in zip(idx, self._stride))
            result[idx] = flat[flat_idx]
        return result

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

    def __float__(self):
        return builtins_float(self.item())

    def __int__(self):
        return builtins_int(self.item())

    # --- View operations ---

    def view(self, *shape):
        """Return a view with a different shape. Must be contiguous."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if not self.is_contiguous():
            raise RuntimeError("view size is not compatible with input tensor's "
                             "size and stride (at least one dimension spans "
                             "across two contiguous subspaces). Use .reshape() instead.")

        # Resolve -1
        new_shape = _resolve_neg_one(shape, self.numel())
        new_stride = _compute_strides(new_shape)

        return Tensor(
            _storage=self._storage,
            _shape=new_shape,
            _stride=new_stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

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
        """Swap two dimensions. Returns a view."""
        ndim = self.dim()
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim

        new_shape = list(self._shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

        new_stride = list(self._stride)
        new_stride[dim0], new_stride[dim1] = new_stride[dim1], new_stride[dim0]

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

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

        return Tensor(
            _storage=self._storage,
            _shape=new_shape,
            _stride=new_stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def unsqueeze(self, dim):
        """Insert a dimension of size 1."""
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

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def squeeze(self, dim=None):
        """Remove dimensions of size 1."""
        if dim is not None:
            if dim < 0:
                dim += self.dim()
            if self._shape[dim] != 1:
                return self
            new_shape = list(self._shape)
            new_stride = list(self._stride)
            new_shape.pop(dim)
            new_stride.pop(dim)
        else:
            new_shape = []
            new_stride = []
            for s, st in zip(self._shape, self._stride):
                if s != 1:
                    new_shape.append(s)
                    new_stride.append(st)

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

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

        # Check for advanced indexing (bool/int tensors)
        has_advanced = any(
            isinstance(k, Tensor) for k in key
        )

        if has_advanced:
            return self._advanced_getitem(key)

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
                if k < 0:
                    k += self._shape[src_dim]
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

    def _advanced_getitem(self, key):
        """Advanced indexing: bool/int tensors. Returns a copy."""
        # Convert to numpy for advanced indexing
        arr = self.numpy()
        np_key = tuple(
            k.numpy() if isinstance(k, Tensor) else k
            for k in key
        )
        result = arr[np_key]
        return Tensor(result, dtype=self._dtype, device=str(self._device))

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

    # --- Reduction methods ---

    def sum(self, dim=None, keepdim=False, *, dtype=None):
        from . import sum as torch_sum
        return torch_sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def mean(self, dim=None, keepdim=False, *, dtype=None):
        from . import mean as torch_mean
        return torch_mean(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def max(self, dim=None, keepdim=False):
        from . import max as torch_max
        return torch_max(self, dim=dim, keepdim=keepdim)

    def min(self, dim=None, keepdim=False):
        from . import min as torch_min
        return torch_min(self, dim=dim, keepdim=keepdim)

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

    def __gt__(self, other):
        return self.gt(other)

    def __lt__(self, other):
        return self.lt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __le__(self, other):
        return self.le(other)
