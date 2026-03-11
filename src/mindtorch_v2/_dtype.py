import builtins as _builtins
import numpy as np

builtins_int = _builtins.int
builtins_float = _builtins.float


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
float8_e4m3fn = DType("float8_e4m3fn", np.uint8, 1, is_floating_point=True)
float8_e5m2 = DType("float8_e5m2", np.uint8, 1, is_floating_point=True)
float8_e8m0fnu = DType("float8_e8m0fnu", np.uint8, 1, is_floating_point=True)
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
uint16 = DType("uint16", np.uint16, 2, is_signed=False)
uint32 = DType("uint32", np.uint32, 4, is_signed=False)
uint64 = DType("uint64", np.uint64, 8, is_signed=False)

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
    float8_e4m3fn: np.uint8,
    float8_e5m2: np.uint8,
    float8_e8m0fnu: np.uint8,
    float16: np.float16,
    float32: np.float32,
    float64: np.float64,
    bfloat16: np.uint16,
    int8: np.int8,
    int16: np.int16,
    int32: np.int32,
    int64: np.int64,
    uint8: np.uint8,
    uint16: np.uint16,
    uint32: np.uint32,
    uint64: np.uint64,
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
    np.dtype(np.uint16): uint16,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
    np.dtype(np.bool_): bool,
    np.dtype(np.complex64): complex64,
    np.dtype(np.complex128): complex128,
}

# Name -> DType lookup
_NAME_MAP = {
    "float16": float16, "half": float16,
    "float8_e4m3fn": float8_e4m3fn,
    "float8_e5m2": float8_e5m2,
    "float8_e8m0fnu": float8_e8m0fnu,
    "float32": float32, "float": float32,
    "float64": float64, "double": float64,
    "bfloat16": bfloat16,
    "int8": int8,
    "int16": int16, "short": int16,
    "int32": int32, "int": int32,
    "int64": int64, "long": int64,
    "uint8": uint8, "byte": uint8,
    "uint16": uint16,
    "uint32": uint32,
    "uint64": uint64,
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


class finfo:
    """Analogous to ``torch.finfo``. Wraps ``numpy.finfo`` for float dtypes."""

    def __init__(self, dtype):
        from ._tensor import Tensor
        if isinstance(dtype, Tensor):
            dtype = dtype.dtype
        if not isinstance(dtype, DType):
            raise TypeError(f"finfo() requires a floating point DType, got {type(dtype)}")
        if not dtype.is_floating_point:
            raise TypeError(f"finfo() requires a floating point dtype, got {dtype}")

        if dtype is bfloat16:
            # numpy doesn't natively support bfloat16
            self.bits = 16
            self.eps = 0.0078125  # 2^-7
            self.max = 3.3895313892515355e+38
            self.min = -3.3895313892515355e+38
            self.smallest_normal = 1.1754943508222875e-38
            self.tiny = 1.1754943508222875e-38
            self.resolution = 0.01
        else:
            np_info = np.finfo(to_numpy_dtype(dtype))
            self.bits = np_info.bits
            self.eps = builtins_float(np_info.eps)
            self.max = builtins_float(np_info.max)
            self.min = builtins_float(np_info.min)
            self.smallest_normal = builtins_float(np_info.smallest_normal)
            self.tiny = builtins_float(np_info.tiny)
            self.resolution = builtins_float(np_info.resolution)
        self.dtype = dtype

    def __repr__(self):
        return (
            f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, "
            f"eps={self.eps}, smallest_normal={self.smallest_normal}, "
            f"tiny={self.tiny}, dtype={self.dtype.name})"
        )


class iinfo:
    """Analogous to ``torch.iinfo``. Wraps ``numpy.iinfo`` for integer dtypes."""

    def __init__(self, dtype):
        from ._tensor import Tensor
        if isinstance(dtype, Tensor):
            dtype = dtype.dtype
        if not isinstance(dtype, DType):
            raise TypeError(f"iinfo() requires an integer DType, got {type(dtype)}")

        np_info = np.iinfo(to_numpy_dtype(dtype))
        self.bits = np_info.bits
        self.max = builtins_int(np_info.max)
        self.min = builtins_int(np_info.min)
        self.dtype = dtype

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype.name})"
