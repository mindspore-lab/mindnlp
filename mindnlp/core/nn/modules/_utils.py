"""utils"""
import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))
