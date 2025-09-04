from collections import namedtuple
import numpy as np

from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim

from mindnlp import core
from ..executor import execute

linalg_cholesky_ex = namedtuple('linalg_cholesky_ex', ['L', 'info'])

def cholesky(A, *, upper=False, out=None):
    cholesky_op = _get_cache_prim(ops.Cholesky)(upper=upper).set_device('CPU')
    return cholesky_op(A)

def cholesky_ex(A, *, upper=False, check_errors=False, out=None):
    try:
        out = cholesky(A, upper=upper, out=out)
        out + 1
        info = core.tensor(0, device=A.device)
    except:
        info = core.tensor(1, device=A.device)
        out = A
    return linalg_cholesky_ex(out, info)


def norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    return core.norm(A, 2 if ord is None else ord, dim, keepdim, dtype=dtype)

def vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    return execute('linalg_vector_norm', x, ord, dim, keepdim, dtype=dtype)

def solve(A, B, *, left=True, out=None):
    return core.tensor(np.linalg.solve(A.numpy(), B.numpy()))

def qr(A, mode='reduced'):
    return execute('linalg_qr', A, mode)
