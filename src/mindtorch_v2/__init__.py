"""mindtorch v2 - PyTorch-compatible API on MindSpore backend."""

__version__ = "0.1.0"

from ._device import device
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
from ._tensor import Tensor
from ._creation import (
    tensor, zeros, ones, empty, full,
    arange, linspace, eye,
    randn, rand, randint,
    zeros_like, ones_like, empty_like,
    from_numpy,
)
from ._functional import (
    add, sub, mul, div, neg, abs, pow,
    exp, log, sqrt, sin, cos, tanh,
    matmul,
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
)
from ._autograd import (
    is_grad_enabled,
    set_grad_enabled,
    no_grad,
    enable_grad,
)

# Import backends to register ops
from ._backends import cpu

# Import submodules
from . import nn
from . import optim

# Memory format constants (for PyTorch API compatibility)
class memory_format:
    """Memory format enumeration."""
    pass

contiguous_format = memory_format()
preserve_format = memory_format()
channels_last = memory_format()
channels_last_3d = memory_format()
