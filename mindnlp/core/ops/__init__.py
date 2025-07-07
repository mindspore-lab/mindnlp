"""core ops like torch funcional api"""
from . import array, blas, comparison, pointwise, creation, random, reduction, other, \
    tensor, _inner, optim
from .array import *
from .blas import *
from .comparison import *
from .pointwise import *
from .creation import *
from .random import *
from .reduction import *
from .other import *
from .tensor import *
# from .fft_op import *
# from .spectral import *
from ._inner import *
from .optim import *

def load_library(lib_path):
    raise ImportError('not support import any ops for now.')

aten = None

__all__ = []
__all__.extend(_inner.__all__)
__all__.extend(array.__all__)
__all__.extend(blas.__all__)
__all__.extend(comparison.__all__)
__all__.extend(creation.__all__)
# __all__.extend(fft_op.__all__)
__all__.extend(pointwise.__all__)
__all__.extend(random.__all__)
__all__.extend(reduction.__all__)
# __all__.extend(spectral.__all__)
__all__.extend(tensor.__all__)
__all__.extend(other.__all__)
__all__.extend(optim.__all__)
