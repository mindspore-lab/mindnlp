"""PyTorch-compatible dtype definitions backed by MindSpore dtypes."""

import builtins as _builtins

import numpy as np
import mindspore


class DType:
    """Represents a tensor data type, wrapping a MindSpore dtype."""

    def __init__(self, name, ms_dtype, np_dtype, size, is_float=False, is_complex_type=False):
        self.name = name
        self._ms_dtype = ms_dtype
        self._np_dtype = np_dtype
        self._itemsize = size
        self._is_floating_point = is_float
        self._is_complex = is_complex_type

    @property
    def is_floating_point(self):
        return self._is_floating_point

    @property
    def is_complex(self):
        return self._is_complex

    @property
    def itemsize(self):
        return self._itemsize

    def to_mindspore(self):
        return self._ms_dtype

    def to_numpy(self):
        return self._np_dtype

    def __repr__(self):
        return f"torch.{self.name}"

    def __str__(self):
        return f"torch.{self.name}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        return NotImplemented


# --- Core dtypes ---
float16 = DType("float16", mindspore.float16, np.float16, 2, is_float=True)
float32 = DType("float32", mindspore.float32, np.float32, 4, is_float=True)
float64 = DType("float64", mindspore.float64, np.float64, 8, is_float=True)
bfloat16 = DType("bfloat16", mindspore.bfloat16, None, 2, is_float=True)

int8 = DType("int8", mindspore.int8, np.int8, 1)
int16 = DType("int16", mindspore.int16, np.int16, 2)
int32 = DType("int32", mindspore.int32, np.int32, 4)
int64 = DType("int64", mindspore.int64, np.int64, 8)

uint8 = DType("uint8", mindspore.uint8, np.uint8, 1)
uint16 = DType("uint16", mindspore.uint16, np.uint16, 2)
uint32 = DType("uint32", mindspore.uint32, np.uint32, 4)
uint64 = DType("uint64", mindspore.uint64, np.uint64, 8)

bool = DType("bool", mindspore.bool_, np.bool_, 1)

complex64 = DType("complex64", mindspore.complex64, np.complex64, 8, is_complex_type=True)
complex128 = DType("complex128", mindspore.complex128, np.complex128, 16, is_complex_type=True)

# --- Aliases ---
half = float16
float = float32
double = float64
long = int64
int = int32
short = int16

cfloat = complex64
cdouble = complex128

# --- Conversion maps ---
_ms_to_dtype = {
    mindspore.float16: float16,
    mindspore.float32: float32,
    mindspore.float64: float64,
    mindspore.bfloat16: bfloat16,
    mindspore.int8: int8,
    mindspore.int16: int16,
    mindspore.int32: int32,
    mindspore.int64: int64,
    mindspore.uint8: uint8,
    mindspore.uint16: uint16,
    mindspore.uint32: uint32,
    mindspore.uint64: uint64,
    mindspore.bool_: bool,
    mindspore.complex64: complex64,
    mindspore.complex128: complex128,
}

_dtype_to_np = {d: d._np_dtype for d in _ms_to_dtype.values() if d._np_dtype is not None}

_np_to_dtype = {v: k for k, v in _dtype_to_np.items()}

_py_to_dtype = {
    _builtins.bool: bool,
    _builtins.float: float,
    _builtins.int: int64,
}


def from_mindspore_dtype(ms_dtype):
    """Convert MindSpore dtype to mindtorch_v2 dtype."""
    return _ms_to_dtype.get(ms_dtype)


def dtype_to_numpy(dtype):
    """Convert mindtorch_v2 dtype to numpy dtype."""
    return _dtype_to_np.get(dtype)


def numpy_to_dtype(np_dtype):
    """Convert numpy dtype to mindtorch_v2 dtype."""
    np_dtype = np.dtype(np_dtype).type
    return _np_to_dtype.get(np_dtype)
