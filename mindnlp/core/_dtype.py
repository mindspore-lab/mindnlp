import warnings
import numpy as np
from mindspore.common.dtype import *
from mindspore._c_expression import typing
from mindspore._c_expression.typing import Type

from .configs import ON_A1

if ON_A1:
    warnings.warn('910A do not support bfloat16, use float16 instead.')
    bfloat16 = float16

dtype = Type

def is_floating_point(self):
    return isinstance(self, (typing.Float, typing.BFloat16))

def is_complex(self):
    return isinstance(self, typing.Complex)

Type.is_floating_point = is_floating_point
Type.is_complex = is_complex
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