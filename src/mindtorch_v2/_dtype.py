import numpy as np


class DType:
    def __init__(self, name, numpy_dtype, itemsize, is_floating_point=False,
                 is_complex=False, is_signed=True):
        self.name = name
        self._numpy_dtype = numpy_dtype
        self.itemsize = itemsize
        self._is_floating_point = is_floating_point
        self._is_complex = is_complex
        self._is_signed = is_signed

    @property
    def is_floating_point(self):
        return self._is_floating_point

    @property
    def is_complex(self):
        return self._is_complex

    @property
    def is_signed(self):
        return self._is_signed

    def __repr__(self):
        return f"torch.{self.name}"

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


# Floating point types
float16 = DType("float16", np.float16, 2, is_floating_point=True)
float32 = DType("float32", np.float32, 4, is_floating_point=True)
float64 = DType("float64", np.float64, 8, is_floating_point=True)
# bfloat16: stored as uint16 bit pattern on CPU, computed in float32
bfloat16 = DType("bfloat16", np.uint16, 2, is_floating_point=True)

# Integer types
int8 = DType("int8", np.int8, 1)
int16 = DType("int16", np.int16, 2)
int32 = DType("int32", np.int32, 4)
int64 = DType("int64", np.int64, 8)
uint8 = DType("uint8", np.uint8, 1, is_signed=False)

# Boolean
bool = DType("bool", np.bool_, 1, is_signed=False)

# Complex types
complex64 = DType("complex64", np.complex64, 8, is_complex=True)
complex128 = DType("complex128", np.complex128, 16, is_complex=True)

# Aliases (matching PyTorch)
half = float16
float = float32
double = float64
short = int16
int = int32
long = int64
byte = uint8
cfloat = complex64
cdouble = complex128


_NUMPY_DTYPE_MAP = {
    float16: np.float16,
    float32: np.float32,
    float64: np.float64,
    bfloat16: np.uint16,
    int8: np.int8,
    int16: np.int16,
    int32: np.int32,
    int64: np.int64,
    uint8: np.uint8,
    bool: np.bool_,
    complex64: np.complex64,
    complex128: np.complex128,
}

# Reverse map: numpy dtype -> DType
_FROM_NUMPY_MAP = {
    np.dtype(np.float16): float16,
    np.dtype(np.float32): float32,
    np.dtype(np.float64): float64,
    np.dtype(np.int8): int8,
    np.dtype(np.int16): int16,
    np.dtype(np.int32): int32,
    np.dtype(np.int64): int64,
    np.dtype(np.uint8): uint8,
    np.dtype(np.bool_): bool,
    np.dtype(np.complex64): complex64,
    np.dtype(np.complex128): complex128,
}

# Name -> DType lookup
_NAME_MAP = {
    "float16": float16, "half": float16,
    "float32": float32, "float": float32,
    "float64": float64, "double": float64,
    "bfloat16": bfloat16,
    "int8": int8,
    "int16": int16, "short": int16,
    "int32": int32, "int": int32,
    "int64": int64, "long": int64,
    "uint8": uint8, "byte": uint8,
    "bool": bool,
    "complex64": complex64, "cfloat": complex64,
    "complex128": complex128, "cdouble": complex128,
}


def to_numpy_dtype(dtype):
    return _NUMPY_DTYPE_MAP.get(dtype, np.float32)


def from_numpy_dtype(np_dtype):
    """Convert a numpy dtype to a mindtorch DType."""
    return _FROM_NUMPY_MAP.get(np.dtype(np_dtype), float32)


def from_name(name):
    """Convert a dtype name string to a DType."""
    return _NAME_MAP.get(name)
