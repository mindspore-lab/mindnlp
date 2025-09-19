"""blas op"""
from mindtorch.executor import execute

# addbmm
def addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return execute('addbmm', input, batch1, batch2, beta, alpha)

# addmm
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    return execute('addmm', input, mat1, mat2, beta, alpha)

# addmv
def addmv(input, mat, vec, *, beta=1, alpha=1, out=None):
    return execute('addmv', input, mat, vec, beta, alpha)

# addr


# baddbmm
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return execute('baddbmm', input, batch1, batch2, beta, alpha)

# bmm
def bmm(input, other):
    return execute('bmm', input, other)

# chain_matmul


# cholesky
def cholesky(input, upper=False, *, out=None):
    return execute('cholesky', input, upper)

# cholesky_inverse

# cholesky_solve

# dot
def dot(input, other):
    return execute('dot', input, other)

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
    return execute('matmul', input, other)

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
    return execute('outer', input, vec2)

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
