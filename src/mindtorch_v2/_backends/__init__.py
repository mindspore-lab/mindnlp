from . import cpu
from . import meta
from . import cuda
from . import npu
from . import autograd

import sys
if sys.platform == "darwin":
    try:
        from . import mps
    except ImportError:
        pass

__all__ = ["cpu", "meta", "cuda", "npu", "autograd"]
