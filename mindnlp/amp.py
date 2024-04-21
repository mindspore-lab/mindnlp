"""Operator level amp"""
import functools
from typing import Any

import mindspore

CELL_WHITE_LIST = [
    'Dense',
    'Conv1d',
    'Conv2d',
    'Conv3d',
]

OP_WHITE_LIST = [
    'MatMul',
    'BatchMatMul',
    'Dense',
    'Conv2D',
    'Conv2DTranspose',
    'Conv3D',
    'Conv3DTranspose',
    'LSTM',
    'CudnnGRU',
    'PReLU'
]

OP_BLACK_LIST = [
    'Asin',
    'Acos',
    'BCEWithLogitsLoss',
    'BinaryCrossEntropy',
    'Cosh',
    'Cdis',
    'CumProd',
    'CumSum',
    'Div',
    'Erfinv',
    'Exp',
    'Expm1',
    'KLDivLoss',
    'LayerNorm',
    'Log',
    'LogSoftmax',
    'Log10',
    'Log1p',
    'Log2',
    'MultilabelMarginLoss',
    'MultiMarginLoss',
    'NLLLoss',
    'LpNorm',
    'L2Normalize',
    'Pdist',
    'Pow',
    'RealDiv',
    'ReduceProd',
    'Reciprocal',
    'Rsqrt',
    'Renorm',
    'Sinh',
    'Sum',
    'Softplus',
    'Softmax',
    'Softmin',
    'SoftMarginLoss',
    'SoftmaxCrossEntropyWithLogits',
    'SparseSoftmaxCrossEntropyWithLogits',
    'SmoothL1Loss',
    'Tan',
    'TripletMarginLoss'
]

GLOBAL_AMP = False
GLOBAL_AMP_DTYPE = mindspore.float32

def _set_amp(mode, dtype):
    global GLOBAL_AMP
    global GLOBAL_AMP_DTYPE
    GLOBAL_AMP = mode
    GLOBAL_AMP_DTYPE = dtype

def get_global_amp():
    return GLOBAL_AMP, GLOBAL_AMP_DTYPE


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    return decorate_autocast

class autocast:
    def __init__(
        self,
        enabled: bool = True,
        dtype = mindspore.float16,
    ):
        self.enabled = enabled
        self.dtype = dtype
        self.old_dtype = GLOBAL_AMP_DTYPE

    def __enter__(self):
        print('amp', self.enabled)
        _set_amp(self.enabled, self.dtype)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        _set_amp(self.enabled, self.old_dtype)
        return False

    def __call__(self, func):
        return autocast_decorator(self, func)
