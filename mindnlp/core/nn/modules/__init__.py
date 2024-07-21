"""new nn modules"""
from .module import Module
from .container import ModuleList, ParameterList, Sequential
from .linear import Linear, Identity
from .sparse import Embedding
from .normalization import LayerNorm, GroupNorm
from .dropout import Dropout
from .activation import *
from .conv import Conv2d, Conv1d, ConvTranspose2d
from .padding import ZeroPad2d, ConstantPad2d, ConstantPad1d, ConstantPad3d
from .batchnorm import BatchNorm2d
from .pooling import AdaptiveAvgPool2d, AvgPool1d, MaxPool2d
