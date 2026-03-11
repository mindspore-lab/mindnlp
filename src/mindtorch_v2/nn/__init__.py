from .module import Module
from .parameter import Parameter
from . import functional
from . import init

# Containers
from .modules.container import Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict

# Linear
from .modules.linear import Linear, Bilinear, Identity, Flatten, Unflatten

# Activations
from .modules.activation import (
    ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax,
    LeakyReLU, ELU, Mish, PReLU, ReLU6, Hardtanh, LogSigmoid,
    Hardswish, Hardsigmoid, SELU, CELU, Softplus, Softsign,
    Threshold, GLU, Softmax2d, Softmin, Tanhshrink,
    Softshrink, Hardshrink, RReLU,
)

# Normalization
from .modules.normalization import LayerNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, RMSNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d

# Embedding
from .modules.sparse import Embedding, EmbeddingBag

# Dropout
from .modules.dropout import Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout, FeatureAlphaDropout

# Convolution
from .modules.conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d

# Pooling
from .modules.pooling import (
    MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d,
    AdaptiveMaxPool1d, AdaptiveMaxPool2d,
)

# Loss
from .modules.loss import (
    CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss,
    NLLLoss, L1Loss, SmoothL1Loss, KLDivLoss,
    HuberLoss, CosineEmbeddingLoss, MarginRankingLoss, TripletMarginLoss,
    HingeEmbeddingLoss, SoftMarginLoss, MultiMarginLoss, MultiLabelSoftMarginLoss,
    PoissonNLLLoss, CTCLoss, GaussianNLLLoss,
)

# RNN
from .modules.rnn import RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell

# Attention
from .modules.attention import MultiheadAttention

# Padding
from .modules.padding import (
    ZeroPad1d, ZeroPad2d, ConstantPad1d, ConstantPad2d,
    ReflectionPad1d, ReflectionPad2d, ReplicationPad1d, ReplicationPad2d,
)

# Upsampling
from .modules.upsampling import Upsample, UpsamplingNearest2d, UpsamplingBilinear2d

# PixelShuffle / ChannelShuffle
from .modules.pixelshuffle import PixelShuffle, PixelUnshuffle, ChannelShuffle

# Distance
from .modules.distance import CosineSimilarity, PairwiseDistance

# Fold / Unfold
from .modules.fold import Fold, Unfold

# Transformer
from .modules.transformer import (
    Transformer, TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer,
)

# Utils
from . import utils

# Parallel
from . import parallel
from .parallel import DistributedDataParallel, DataParallel
