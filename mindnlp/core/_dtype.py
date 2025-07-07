import numpy as np
from mindspore.common.dtype import *
from mindspore._c_expression import typing
from mindspore._c_expression.typing import Type

dtype = Type

def is_floating_point(self):
    return isinstance(self, (typing.Float, typing.BFloat16))

Type.is_floating_point = is_floating_point

half = float16
float = float32
double = float64

long = int64
int = int32
bool = bool_

float8_e4m3fn = None # TODO: not support fp8 for now

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