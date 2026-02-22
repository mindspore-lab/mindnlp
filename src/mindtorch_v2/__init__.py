__version__ = "0.1.0"

from ._dtype import (
    DType,
    float16, float32, float64, bfloat16,
    int8, int16, int32, int64, uint8,
    bool,
    complex64, complex128,
    # aliases
    half, double, short, long, byte, cfloat, cdouble,
)
from ._dtype import float as float  # noqa: F811
from ._dtype import int as int  # noqa: F811
from ._device import device as Device, _default_device, get_default_device, set_default_device
from ._tensor import Tensor
from ._creation import tensor, zeros, ones
from ._storage import UntypedStorage, TypedStorage
from ._functional import add, mul, matmul, relu, sum, abs, neg, exp, log, sqrt
from ._functional import sin, cos, tan, tanh, sigmoid, floor, ceil, round, trunc, frac
from ._functional import pow, log2, log10, exp2, rsqrt
from ._functional import sign, signbit, isnan, isinf, isfinite
from ._functional import sinh, cosh, asinh, acosh, atanh, erf, erfc, softplus
from ._functional import clamp, clamp_min, clamp_max, relu6, hardtanh
from ._functional import min, max, amin, amax, fmin, fmax, where
from ._functional import atan, atan2, asin, acos, lerp, addcmul, addcdiv
from ._functional import logaddexp, logaddexp2, hypot, remainder, fmod
from ._printing import set_printoptions, get_printoptions
from ._dispatch import pipeline_context
from ._backends import cpu
from ._autograd.grad_mode import is_grad_enabled, set_grad_enabled, no_grad, enable_grad
from . import _autograd as autograd
from . import npu
from . import _C
from . import distributed


def pipeline():
    return pipeline_context()


__all__ = [
    "Device",
    "Tensor",
    "DType",
    # dtypes
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8",
    "bool",
    "complex64", "complex128",
    # dtype aliases
    "half", "float", "double",
    "short", "int", "long", "byte",
    "cfloat", "cdouble",
    # creation
    "tensor",
    "zeros",
    "ones",
    # ops
    "add",
    "mul",
    "matmul",
    "relu",
    "abs",
    "neg",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sigmoid",
    "floor",
    "ceil",
    "round",
    "trunc",
    "frac",
    "pow",
    "log2",
    "log10",
    "exp2",
    "rsqrt",
    "sign",
    "signbit",
    "isnan",
    "isinf",
    "isfinite",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "erfc",
    "softplus",
    "clamp",
    "clamp_min",
    "clamp_max",
    "relu6",
    "hardtanh",
    "min",
    "max",
    "amin",
    "amax",
    "fmin",
    "fmax",
    "where",
    "atan",
    "atan2",
    "asin",
    "acos",
    "lerp",
    "addcmul",
    "addcdiv",
    "logaddexp",
    "logaddexp2",
    "hypot",
    "remainder",
    "fmod",
    "sum",
    # printing
    "set_printoptions",
    "get_printoptions",
    # pipeline
    "pipeline",
    "pipeline_context",
    # device
    "get_default_device",
    "set_default_device",
    "npu",
    # autograd
    "autograd",
    # distributed
    "distributed",
]
