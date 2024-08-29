"""new nn modules"""
from .module import Module
from .container import ModuleList, ParameterList, Sequential, ParameterDict, ModuleDict
from .linear import Linear, Identity
from .sparse import Embedding
from .normalization import LayerNorm, GroupNorm
from .dropout import Dropout, Dropout2d
from .activation import *
from .conv import Conv3d, Conv2d, Conv1d, ConvTranspose2d, ConvTranspose1d
from .padding import ZeroPad2d, ConstantPad2d, ConstantPad1d, ConstantPad3d
from .batchnorm import BatchNorm2d, BatchNorm1d
from .pooling import AdaptiveAvgPool2d, AvgPool1d, MaxPool2d, MaxPool1d, AdaptiveAvgPool1d, AvgPool2d
from .flatten import Unflatten, Flatten
from .rnn_cell import RNNCell, GRUCell, LSTMCell
from .rnn import RNN, LSTM, GRU
from .fold import Unfold, Fold
from .pixelshuffle import PixelUnshuffle, PixelShuffle
from .upsampling import Upsample, UpsamplingBilinear2d, UpsamplingNearest2d
from .loss import *
from .distance import *
from .adaptive import AdaptiveLogSoftmaxWithLoss
