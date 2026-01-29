"""Neural network module for mindtorch_v2."""

from .parameter import Parameter
from .module import Module
from .modules import (
    Linear, Identity,
    ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax,
    LeakyReLU, ELU, CELU, SELU, PReLU, Mish, Hardswish, Hardsigmoid, Softplus, Hardtanh, ReLU6,
    Embedding,
    Dropout,
    LayerNorm, GroupNorm,
    BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm,
    Sequential, ModuleList, ModuleDict,
    Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    MultiheadAttention,
    ZeroPad1d, ZeroPad2d, ZeroPad3d,
    ConstantPad1d, ConstantPad2d, ConstantPad3d,
    ReflectionPad1d, ReflectionPad2d, ReflectionPad3d,
    ReplicationPad1d, ReplicationPad2d, ReplicationPad3d,
    MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool2d,
    Flatten,
)
from .modules.parallel import DataParallel
from .modules.loss import (
    MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, NLLLoss,
    L1Loss, SmoothL1Loss, HuberLoss, KLDivLoss, BCELoss, CosineEmbeddingLoss
)
from . import functional
from . import init
from . import parallel
from . import utils

__all__ = [
    'Parameter', 'Module',
    'Linear', 'Identity',
    'ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
    'LeakyReLU', 'ELU', 'PReLU', 'Mish', 'Hardswish', 'Hardsigmoid', 'Softplus', 'Hardtanh', 'ReLU6',
    'Embedding',
    'Dropout',
    'LayerNorm',
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'SyncBatchNorm',
    'Sequential', 'ModuleList', 'ModuleDict',
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'MultiheadAttention',
    'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d',
    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d',
    'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d',
    'DataParallel',
    'MSELoss', 'CrossEntropyLoss', 'BCEWithLogitsLoss', 'NLLLoss',
    'L1Loss', 'SmoothL1Loss', 'HuberLoss', 'KLDivLoss', 'BCELoss', 'CosineEmbeddingLoss',
    'functional', 'init', 'parallel',
]
