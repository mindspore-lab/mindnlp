"""mindtorch v2 - PyTorch-compatible API on MindSpore backend."""

__version__ = "0.1.0"

from ._device import device, _get_default_device
from ._storage import TypedStorage, UntypedStorage
from ._dtype import (
    # Core dtypes
    float16, float32, float64, bfloat16,
    # Float8 dtypes
    float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    bool, complex64, complex128,
    # Aliases
    half, float, double, long, int, short,
    cfloat, cdouble,
    # Conversion functions
    dtype_to_numpy, numpy_to_dtype, from_mindspore_dtype,
    # DType class
    DType,
)
# Alias for PyTorch compatibility: torch.dtype is the dtype class
dtype = DType
from ._tensor import Tensor
from ._creation import (
    tensor, zeros, ones, empty, full, full_like,
    arange, linspace, eye,
    randn, rand, randint, rand_like,
    zeros_like, ones_like, empty_like,
    from_numpy, as_tensor, frombuffer, asarray,
)
from ._functional import (
    add, sub, mul, div, neg, abs, pow,
    exp, log, sqrt, square, sin, cos, tanh,
    matmul, addmm,
    sum, mean, max, min, prod, argmax, argmin,
    eq, ne, gt, lt, ge, le,
    # Tensor manipulation
    cat, stack, split, chunk, clone, where,
    # Additional math ops
    var, std, clamp, rsqrt, reciprocal, bmm, baddbmm,
    # Boolean reductions
    all, any,
    # Element testing
    isin,
    # Top-k and sampling
    topk, multinomial,
    # Cumulative ops
    cumsum, cumprod,
    # Rounding ops
    floor, trunc, ceil, round, sign,
    # Modular arithmetic
    fmod, remainder,
    # Log variants
    log10, log2, log1p, expm1,
    # Trigonometric
    acos, asin, atan, atan2,
    # Hyperbolic
    cosh, sinh, acosh, asinh, atanh,
    # Activation functions
    relu, sigmoid,
    # Tensor manipulation
    squeeze, unsqueeze, flip, roll, gather, index_select,
    repeat_interleave, unique_consecutive, einsum,
    sort, reshape, permute, transpose, narrow, masked_fill,
    # Math utilities
    norm, isnan, isinf, isfinite,
    # Logical ops
    logical_not, logical_and, logical_or,
    # Comparison
    equal,
    # Linear algebra
    svd_lowrank,
    # Difference operations
    diff, maximum, minimum, take_along_dim,
    # Complex number operations
    conj, conj_physical,
    # Binning and histogram
    bucketize, histc,
    # Index operations
    nonzero, argsort,
)
from ._autograd import (
    is_grad_enabled,
    set_grad_enabled,
    no_grad,
    enable_grad,
)

# Import backends to register ops based on device target
from .configs import DEVICE_TARGET

# Always import CPU backend as fallback
from ._backends import cpu

# Conditionally import Ascend backend when running on Ascend NPU
if DEVICE_TARGET == 'Ascend':
    from ._backends import ascend

# Import submodules
from . import nn
from . import optim
from . import _autograd as autograd
from . import npu  # NPU (Ascend) device support

# Import fft and distributions modules from stubs
from ._torch_proxy.stubs import fft
from ._torch_proxy.stubs import distributions

# Aliases for API compatibility
concat = cat  # torch.concat is an alias for torch.cat

# Memory format constants (for PyTorch API compatibility)
class memory_format:
    """Memory format enumeration."""
    pass

contiguous_format = memory_format()
preserve_format = memory_format()
channels_last = memory_format()
channels_last_3d = memory_format()


# Layout constants (for PyTorch API compatibility)
class layout:
    """Tensor layout enumeration."""
    def __eq__(self, other):
        return type(self) == type(other)

    def __repr__(self):
        return 'torch.strided'

strided = layout()
sparse_coo = layout()
sparse_csr = layout()
sparse_csc = layout()
sparse_bsr = layout()
sparse_bsc = layout()

# Default dtype management
_default_dtype = float32

def get_default_dtype():
    """Get the current default floating point dtype."""
    return _default_dtype

def set_default_dtype(dtype):
    """Set the default floating point dtype."""
    global _default_dtype
    _default_dtype = dtype


# Default device management
from ._device import _set_default_device

def get_default_device():
    """Get the current default device.

    Returns the device from the device context manager if active,
    otherwise returns the device based on MindSpore context.
    Matches PyTorch 2.3+ API.
    """
    ctx_device = _get_default_device()
    if ctx_device is not None:
        return ctx_device
    # Return device based on MindSpore context
    if DEVICE_TARGET == 'Ascend':
        return device("npu")
    return device("cpu")


def set_default_device(dev):
    """Set the default device for tensor creation.

    Args:
        dev: Device to set as default. Can be a device object, string, or None.
            If None, resets to the default based on MindSpore context.
    """
    if dev is None:
        _set_default_device(None)
    elif isinstance(dev, str):
        _set_default_device(device(dev))
    else:
        _set_default_device(dev)


# Tensor type classes (for isinstance checks and type creation)
class _TensorTypeMeta(type):
    """Metaclass for tensor type classes that enables isinstance checks."""
    def __instancecheck__(cls, instance):
        if not isinstance(instance, Tensor):
            return False
        return instance.dtype == cls._dtype


class BoolTensor(metaclass=_TensorTypeMeta):
    """Boolean tensor type."""
    _dtype = bool

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=bool, *args, **kwargs)


class FloatTensor(metaclass=_TensorTypeMeta):
    """32-bit floating point tensor type."""
    _dtype = float32

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=float32, *args, **kwargs)


class DoubleTensor(metaclass=_TensorTypeMeta):
    """64-bit floating point tensor type."""
    _dtype = float64

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=float64, *args, **kwargs)


class HalfTensor(metaclass=_TensorTypeMeta):
    """16-bit floating point tensor type."""
    _dtype = float16

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=float16, *args, **kwargs)


class LongTensor(metaclass=_TensorTypeMeta):
    """64-bit integer tensor type."""
    _dtype = int64

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=int64, *args, **kwargs)


class IntTensor(metaclass=_TensorTypeMeta):
    """32-bit integer tensor type."""
    _dtype = int32

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=int32, *args, **kwargs)


class ShortTensor(metaclass=_TensorTypeMeta):
    """16-bit integer tensor type."""
    _dtype = int16

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=int16, *args, **kwargs)


class ByteTensor(metaclass=_TensorTypeMeta):
    """8-bit unsigned integer tensor type."""
    _dtype = uint8

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=uint8, *args, **kwargs)


class CharTensor(metaclass=_TensorTypeMeta):
    """8-bit signed integer tensor type."""
    _dtype = int8

    def __new__(cls, data=None, *args, **kwargs):
        return Tensor(data, dtype=int8, *args, **kwargs)
