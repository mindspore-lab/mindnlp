import warnings
import numpy as np
from mindspore.common.dtype import *
from mindspore._c_expression import typing
from mindspore._c_expression.typing import Type

from .configs import ON_A1, SUPPORT_BF16, DEVICE_TARGET

if SUPPORT_BF16:
    from mindspore.common.np_dtype import bfloat16 as np_bfloat16# pylint: disable=import-error
else:
    from ml_dtypes import bfloat16 as np_bfloat16

bool_alias = bool
float_alias = float
int_alias = int

if ON_A1 or DEVICE_TARGET == 'GPU':
    warnings.warn('MindSpore on GPU/910A do not support bfloat16, use float16 instead.')
    bfloat16 = float16

dtype = Type

@property
def _is_floating_point(self):
    return isinstance(self, (typing.Float, typing.BFloat))

@property
def _is_complex(self):
    return isinstance(self, typing.Complex)

Type.is_floating_point = _is_floating_point
Type.is_complex = _is_complex
Type.__str__ = Type.__repr__


@property
def itemsize(self):
    return ITEM_SIZE[self]

Type.itemsize = itemsize

def __gt__(self, other):
    return self.itemsize > other.itemsize

Type.__gt__ = __gt__

half = float16
float = float32
double = float64

long = int64
int = int32
bool = bool_

float8_e4m3fn = None # TODO: not support fp8 for now
float8_e5m2 = None
float8_e4m3fnuz = None
float8_e5m2fnuz = None
complex32 = None
cfloat = complex64
cdouble = complex128

uint1 = None
uint2 = None
uint3 = None
uint4 = None
uint5 = None
uint6 = None
uint7 = None

ITEM_SIZE = {
    bool    : 1,
    int8    : 1,
    int16   : 2,
    int32   : 4,
    int64   : 8,
    uint8   : 1,
    uint16  : 2,
    uint32  : 4,
    uint64  : 8,
    float16 : 2,
    bfloat16 : 2,
    float32 : 4,
    float64 : 8,
    complex64: 8
}

np2dtype = {
    np.bool_: bool,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.uint64: uint64,
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
}

dtype2np = {
    bool    : np.bool_,
    int8    : np.int8,
    int16   : np.int16,
    int32   : np.int32,
    int64   : np.int64,
    uint8   : np.uint8,
    uint16  : np.uint16,
    uint32  : np.uint32,
    uint64  : np.uint64,
    float16 : np.float16,
    float32 : np.float32,
    float64 : np.float64,
}

if not ON_A1:
    dtype2np[bfloat16] = np_bfloat16

py2dtype = {
    bool_alias: bool,
    float_alias: float,
    int_alias: int64
}

mantissa_bits_map = {
    int64: 63,
    float32: 23,   # FP32, 单精度[6](@ref)
    float64: 52,   # FP64, 双精度
    float16: 10,   # FP16, 半精度[1,7](@ref)
    bfloat16: 7,   # BF16, Brain浮点16位[7](@ref)
}
