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
    # info classes
    finfo, iinfo,
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
from ._creation import tensor, zeros, ones, empty, arange, linspace, full, logspace, eye, range, randn, rand, randint, randperm, from_numpy, as_tensor, normal
from ._functional import zeros_like
from ._functional import ones_like, empty_like, full_like, randn_like, rand_like, randint_like
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
from ._functional import squeeze, unsqueeze, permute
from ._functional import var, norm, prod
from ._functional import reciprocal, addmm, einsum
from ._functional import mm, bmm
from ._functional import floor_divide
from ._functional import narrow, flatten
from ._functional import logical_and, logical_or, logical_not
from ._functional import sub, log1p, expm1, maximum, minimum
from ._functional import dot, outer, inner, mv, cross, tensordot
from ._functional import logical_xor
from ._functional import baddbmm, trace, cummin, logsumexp, renorm
from ._functional import bitwise_and, bitwise_or, bitwise_xor, bitwise_not
from ._functional import unflatten, broadcast_to, movedim, moveaxis, diagonal
from ._functional import unique, searchsorted, kthvalue, median
# Category A: Export existing functions
from ._functional import eq, ne, lt, le, gt, ge
from ._functional import select, expand, masked_fill, unfold
from ._functional import scatter_, scatter_add_
from ._functional import index_add_, index_copy_, index_fill_
from ._functional import index_put, index_put_
from ._functional import masked_fill_, masked_scatter_
# Category B: Wrapper + export
from ._functional import nansum, nanmean, det, dist, matrix_power, argwhere
# Category C1: Pure-Python functions
from ._functional import meshgrid, atleast_1d, atleast_2d, atleast_3d
from ._functional import broadcast_tensors, broadcast_shapes
from ._functional import complex, polar
# Category C2: Dispatch-based functions
from ._functional import diff, bincount, cdist, aminmax
from ._functional import quantile, nanquantile, nanmedian
from ._functional import histc, histogram, bucketize
from ._functional import isneginf, isposinf, isreal, isin, heaviside
# P0 dtype utilities & query functions
from ._functional import is_tensor, is_floating_point, is_complex, numel, square
from ._printing import set_printoptions, get_printoptions
from ._dispatch import (
    pipeline_context,
    functionalize_context,
    set_pipeline_config,
    get_pipeline_config,
)
from ._backends import cpu
from ._autograd.grad_mode import is_grad_enabled, set_grad_enabled, no_grad, enable_grad, inference_mode
from . import _autograd as autograd
from ._backends import autograd as _autograd_kernels
from . import cuda
from . import npu
from . import mps
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
from . import linalg
from . import fft
from . import special
from . import testing
from ._random import (
    manual_seed, seed, initial_seed, get_rng_state, set_rng_state,
    Generator, default_generator,
    bernoulli, multinomial, poisson,
    fork_rng,
)
from . import _random as random
from .serialization import save, load
from .amp.state import (
    is_autocast_enabled,
    set_autocast_enabled,
    get_autocast_dtype,
    set_autocast_dtype,
    is_autocast_cache_enabled,
    set_autocast_cache_enabled,
)


def pipeline(**kwargs):
    return pipeline_context(**kwargs)


def pipeline_config(**kwargs):
    return set_pipeline_config(**kwargs)


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
    "cuda",
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
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "randn_like",
    "rand_like",
    "randint_like",
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
    "squeeze",
    "unsqueeze",
    "permute",
    "var",
    "norm",
    "prod",
    "mm",
    "bmm",
    "floor_divide",
    # P1 ops
    "reciprocal",
    "addmm",
    "einsum",
    "randint",
    "randperm",
    "from_numpy",
    "as_tensor",
    # Batch 1 ops
    "narrow",
    "flatten",
    "logical_and",
    "logical_or",
    "logical_not",
    # printing
    "set_printoptions",
    "get_printoptions",
    # pipeline
    "pipeline",
    "pipeline_context",
    "pipeline_config",
    "get_pipeline_config",
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
    "is_autocast_enabled",
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",
    "is_autocast_cache_enabled",
    "set_autocast_cache_enabled",
    "ops",
    "library",
    "compiler",
    "optim",
    "jit",
    "profiler",
    "compile",
    "save",
    "load",
    # new creation ops
    "randint",
    "randperm",
    # new math ops
    "sub",
    "log1p",
    "expm1",
    "reciprocal",
    "maximum",
    "minimum",
    "dot",
    "outer",
    "inner",
    "mv",
    "cross",
    "tensordot",
    "einsum",
    # new logical ops
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    # new bitwise ops
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    # new shape ops
    "flatten",
    "unflatten",
    "broadcast_to",
    "movedim",
    "moveaxis",
    "diagonal",
    # new search ops
    "unique",
    "searchsorted",
    "kthvalue",
    "median",
    # P1 new ops
    "baddbmm",
    "trace",
    "cummin",
    "logsumexp",
    "renorm",
    # new random ops
    "bernoulli",
    "multinomial",
    # Category A: Comparison ops
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    # Category A: Indexing/mutation ops
    "select",
    "expand",
    "masked_fill",
    "masked_fill_",
    "unfold",
    "scatter_",
    "scatter_add_",
    "index_add_",
    "index_copy_",
    "index_fill_",
    "index_put",
    "index_put_",
    "masked_scatter_",
    # Category B: Wrapper ops
    "nansum",
    "nanmean",
    "det",
    "dist",
    "matrix_power",
    "argwhere",
    # Category C1: Pure-Python ops
    "meshgrid",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_tensors",
    "broadcast_shapes",
    "complex",
    "polar",
    # Category C2: Dispatch-based ops
    "diff",
    "bincount",
    "cdist",
    "aminmax",
    "quantile",
    "nanquantile",
    "nanmedian",
    "histc",
    "histogram",
    "bucketize",
    "isneginf",
    "isposinf",
    "isreal",
    "isin",
    "heaviside",
    # P0 dtype utilities & query functions
    "finfo",
    "iinfo",
    "is_tensor",
    "is_floating_point",
    "is_complex",
    "numel",
    "square",
    # submodules
    "linalg",
    "fft",
    "special",
    "testing",
]
