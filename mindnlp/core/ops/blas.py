"""blas op"""
import mindspore

from mindspore import ops
from ..configs import use_pyboost, ON_ORANGE_PI
from ._inner import call_ms_func

# addbmm
has_addbmm = hasattr(mindspore.mint, 'addbmm')
def addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    if use_pyboost() and has_addbmm:
        return call_ms_func(mindspore.mint.addbmm, input, batch1, batch2, beta=beta, alpha=alpha, out=out)
    return call_ms_func(ops.addbmm, input, batch1, batch2, beta=beta, alpha=alpha, out=out)

# addmm
has_addmm = hasattr(mindspore.mint, 'addmm')
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    if use_pyboost() and has_addmm:
        return mindspore.mint.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    return ops.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

# addmv


# addr


# baddbmm
has_baddbmm = hasattr(mindspore.mint, 'baddbmm')
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    if use_pyboost() and has_baddbmm:
        return call_ms_func(mindspore.mint.baddbmm, input, batch1, batch2, beta=beta, alpha=alpha, out=out)
    return call_ms_func(ops.baddbmm, input, batch1, batch2, beta=beta, alpha=alpha, out=out)

# bmm
has_bmm = hasattr(mindspore.mint, 'bmm')
def bmm(input, other, *, out=None):
    if ON_ORANGE_PI:
        input = input.to(mindspore.float16)
        other = input.to(mindspore.float16)
    if use_pyboost() and has_bmm:
        return call_ms_func(mindspore.mint.bmm, input, other, out=out)
    return call_ms_func(ops.bmm, input, other, out=out)

# chain_matmul


# cholesky

# cholesky_inverse

# cholesky_solve

# dot
has_dot = hasattr(mindspore.mint, 'dot')
def dot(input, other):
    if use_pyboost() and has_dot:
        return mindspore.mint.dot(input, other)
    return (input * other).sum()

# geqrf

# ger

# inner

# inverse

# det

# logdet

# slogdet

# lu

# lu_solve


# lu_unpack

# matmul
has_matmul = hasattr(mindspore.mint, 'matmul')
def matmul(input, other, *, out=None):
    if ON_ORANGE_PI:
        input = input.to(mindspore.float16)
        other = other.to(mindspore.float16)
    if use_pyboost() and has_matmul:
        return call_ms_func(mindspore.mint.matmul, input, other, out=out)
    return call_ms_func(ops.matmul, input, other, out=out)

# matrix_power

# matrix_exp

# mm
has_mm = hasattr(mindspore.mint, 'mm')
def mm(input, other):
    return matmul(input, other)

# mv


# orgqr

# ormqr

# outer
has_outer = hasattr(mindspore.mint, 'outer')
def outer(input, vec2, *, out=None):
    if use_pyboost() and has_outer:
        return call_ms_func(mindspore.mint.outer, input, vec2, out=out)
    return call_ms_func(ops.outer, input, vec2, out=out)

# pinverse


# qr

# svd

# svd_lowrank

# pca_lowrank


# lobpcg


# trapz


# trapezoid


# cumulative_trapezoid


# triangular_solve


# vdot

__all__ = [
    'addbmm',
    'addmm',
    # addmv
    # addr
    'baddbmm',
    'bmm',
    # chain_matmul
    # cholesky
    # cholesky_inverse
    # cholesky_solve
    'dot',
    # geqrf
    # ger
    # inner
    # inverse
    # det
    # logdet
    # slogdet
    # lu
    # lu_solve
    # lu_unpack
    'matmul',
    # matrix_power
    # matrix_exp
    'mm',
    # mv
    # orgqr
    # ormqr
    'outer',
    # pinverse
    # qr
    # svd
    # svd_lowrank
    # pca_lowrank
    # lobpcg
    # trapz
    # trapezoid
    # cumulative_trapezoid
    # triangular_solve
    # vdot
]
