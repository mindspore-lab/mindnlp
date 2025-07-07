import collections
from itertools import repeat

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")
