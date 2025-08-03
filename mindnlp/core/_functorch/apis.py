import functools
from typing import Callable

import mindspore
from mindspore import nn
from mindnlp import core

from .vmap import (
    _check_randomness_arg,
    vmap_impl,
)

class VmapFn(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def construct(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def vmap(
    func: Callable,
    in_dims = 0,
    out_dims = 0,
    randomness: str = "error",
    *,
    chunk_size=None,
) -> Callable:
    return mindspore.vmap(func, in_dims, out_dims)
