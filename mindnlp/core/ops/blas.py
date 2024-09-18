"""blas op"""
import mindspore

from mindspore import ops
from mindnlp.configs import use_pyboost, ON_ORANGE_PI

# addbmm
def addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return ops.addbmm(input, batch1, batch2, beta=beta, alpha=alpha)

# addmm
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    return ops.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

# addmv


# addr


# baddbmm
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return ops.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)

# bmm
def bmm(input, other):
    if ON_ORANGE_PI:
        input = input.to(mindspore.float16)
        other = input.to(mindspore.float16)
    if use_pyboost():
        return mindspore.mint.bmm(input, other)
    return ops.bmm(input, other)

# chain_matmul


# cholesky

# cholesky_inverse

# cholesky_solve

# dot
def dot(input, other):
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
def matmul(input, other):
    if ON_ORANGE_PI:
        input = input.to(mindspore.float16)
        other = other.to(mindspore.float16)
    if use_pyboost():
        return mindspore.mint.matmul(input, other)
    return ops.matmul(input, other)

# matrix_power

# matrix_exp

# mm
def mm(input, other):
    return matmul(input, other)

# mv


# orgqr

# ormqr

# outer
def outer(input, vec2):
    return ops.outer(input, vec2)

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
