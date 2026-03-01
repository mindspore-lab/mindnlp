__version__ = "0.1.0"

from ._dtype import (
    DType,
    float8_e4m3fn, float8_e5m2, float8_e8m0fnu,
    float16, float32, float64, bfloat16,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    bool,
    complex64, complex128,
    # aliases
    half, double, short, long, byte, cfloat, cdouble,
)
from ._dtype import float as float  # noqa: F811
from ._dtype import int as int  # noqa: F811
from ._dtype import DType as dtype  # torch.dtype compatibility
from ._dtype import DType as Dtype  # schema/type alias compatibility
from ._device import device as Device, _default_device, get_default_device, set_default_device
from ._device import device
from ._tensor import Tensor

# Tensor type aliases for torch API compatibility
FloatTensor = Tensor
DoubleTensor = Tensor
HalfTensor = Tensor
BFloat16Tensor = Tensor
ByteTensor = Tensor
CharTensor = Tensor
ShortTensor = Tensor
IntTensor = Tensor
LongTensor = Tensor
BoolTensor = Tensor
ComplexFloatTensor = Tensor
ComplexDoubleTensor = Tensor
Size = tuple
from ._creation import tensor, zeros, ones, empty, arange, linspace, full, logspace, eye, range, randn, rand
from ._functional import zeros_like
from ._storage import UntypedStorage, TypedStorage
from ._functional import add, mul, matmul, relu, sum, all, any, argmax, argmin, count_nonzero, masked_select, flip, roll, rot90, repeat, repeat_interleave, tile, nonzero, allclose, isclose, equal, cumsum, cumprod, cummax, argsort, sort, topk, stack, cat, concat, concatenate, hstack, vstack, row_stack, dstack, column_stack, pad_sequence, block_diag, tril, triu, diag, cartesian_prod, chunk, split, vsplit, hsplit, dsplit, unbind, tril_indices, triu_indices, take, take_along_dim, index_select, gather, scatter, abs, neg, exp, log, sqrt, div, true_divide, mean, std
from ._functional import sin, cos, tan, tanh, sigmoid, floor, ceil, round, trunc, frac
from ._functional import pow, log2, log10, exp2, rsqrt
from ._functional import sign, signbit, isnan, isinf, isfinite
from ._functional import sinh, cosh, asinh, acosh, atanh, erf, erfc, softplus
from ._functional import clamp, clamp_min, clamp_max, relu6, hardtanh
from ._functional import min, max, amin, amax, fmin, fmax, where
from ._functional import atan, atan2, asin, acos, lerp, addcmul, addcdiv
from ._functional import reshape, transpose
from ._functional import logaddexp, logaddexp2, hypot, remainder, fmod
from ._printing import set_printoptions, get_printoptions
from ._dispatch import pipeline_context, functionalize_context
from ._backends import cpu
from ._autograd.grad_mode import is_grad_enabled, set_grad_enabled, no_grad, enable_grad, inference_mode
from . import _autograd as autograd
from . import npu
from . import _C
from . import distributed
from . import onnx
from . import futures
from . import amp
from . import compiler
from .ops import ops
from . import library
from . import optim
from . import jit
from . import profiler
from . import multiprocessing
from ._random import (
    manual_seed, seed, initial_seed, get_rng_state, set_rng_state,
    Generator, default_generator,
)
from . import _random as random


def pipeline():
    return pipeline_context()


def functionalize():
    return functionalize_context()


def compile(model=None, *args, **kwargs):
    if callable(model):
        return model
    def decorator(fn):
        return fn
    return decorator


__all__ = [
    "Device",
    "device",
    "Tensor",
    "Size",
    "FloatTensor", "DoubleTensor", "HalfTensor", "BFloat16Tensor",
    "ByteTensor", "CharTensor", "ShortTensor", "IntTensor", "LongTensor",
    "BoolTensor", "ComplexFloatTensor", "ComplexDoubleTensor",
    "DType",
    "dtype",
    "Dtype",
    # dtypes
    "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu",
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
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
    "empty",
    "randn",
    "rand",
    "arange",
    "linspace",
    "full",
    "logspace",
    "eye",
    "range",
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
    "all",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "masked_select",
    "flip",
    "roll",
    "rot90",
    "repeat",
    "repeat_interleave",
    "tile",
    "nonzero",
    "cumsum",
    "cumprod",
    "cummax",
    "argsort",
    "sort",
    "topk",
    "stack",
    "cat",
    "concat",
    "concatenate",
    "hstack",
    "vstack",
    "row_stack",
    "dstack",
    "column_stack",
    "pad_sequence",
    "block_diag",
    "tril",
    "triu",
    "diag",
    "cartesian_prod",
    "chunk",
    "split",
    "vsplit",
    "hsplit",
    "dsplit",
    "unbind",
    "tril_indices",
    "triu_indices",
    "take",
    "take_along_dim",
    "index_select",
    "gather",
    "scatter",
    "allclose",
    "isclose",
    "equal",
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
    "std",
    "reshape",
    "transpose",
    # printing
    "set_printoptions",
    "get_printoptions",
    # pipeline
    "pipeline",
    "pipeline_context",
    "functionalize",
    "functionalize_context",
    # device
    "get_default_device",
    "set_default_device",
    "npu",
    # autograd
    "autograd",
    "is_grad_enabled",
    "set_grad_enabled",
    "no_grad",
    "enable_grad",
    "inference_mode",
    # distributed
    "distributed",
    "multiprocessing",
    "onnx",
    # amp
    "amp",
    "ops",
    "library",
    "compiler",
    "optim",
    "jit",
    "profiler",
    "compile",
]
