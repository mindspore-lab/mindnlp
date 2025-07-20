from collections import namedtuple
from mindspore import ops, mint
from mindspore.ops._primitive_cache import _get_cache_prim

from mindnlp import core

linalg_cholesky_ex = namedtuple('linalg_cholesky_ex', ['L', 'info'])

def cholesky(A, *, upper=False, out=None):
    cholesky_op = _get_cache_prim(ops.Cholesky)(upper=upper).set_device('CPU')
    return cholesky_op(A)

def cholesky_ex(A, *, upper=False, check_errors=False, out=None):
    try:
        out = cholesky(A, upper=upper, out=out)
        out + 1
        info = core.Tensor(0)
    except:
        info = core.Tensor(1)
        out = A
    return linalg_cholesky_ex(out, info)


def norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    return mint.norm(A, 2 if ord is None else ord, dim, keepdim, dtype=dtype)
