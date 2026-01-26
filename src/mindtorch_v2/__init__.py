"""mindtorch v2 - PyTorch-compatible API on MindSpore backend."""

__version__ = "0.1.0"

from ._device import device
from ._storage import TypedStorage, UntypedStorage
from ._dtype import (
    # Core dtypes
    float16, float32, float64, bfloat16,
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
    randn, rand,
    zeros_like, ones_like, empty_like,
    from_numpy,
)
from ._functional import (
    add, sub, mul, div, neg, abs, pow,
    exp, log, sqrt, sin, cos, tanh,
    matmul,
    sum, mean, max, min, prod, argmax, argmin,
    eq, ne, gt, lt, ge, le,
)

# Import backends to register ops
from ._backends import cpu
