"""other op"""
import copy
import mindspore
from mindspore import ops
from mindnlp.configs import USE_PYBOOST


# atleast_2d


# atleast_3d


# bincount
def bincount(input, weights=None, minlength=0):
    return ops.bincount(input, weights, minlength)

# block_diag


# broadcast_tensors


# broadcast_to
def broadcast_to(input, shape):
    if USE_PYBOOST:
        return mindspore.mint.broadcast_to(input, shape)
    return ops.broadcast_to(input, shape)

# broadcast_shapes


# bucketize

# cartesian_prod


# cdist

# clone
def clone(input):
    return copy.deepcopy(input)

# combinations


# corrcoef


# cov


# cross

# cummax

# cummin

# cumprod

# cumsum
def cumsum(input, dim, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.cumsum(input, dim, dtype)
    return ops.cumsum(input, dim, dtype)

# diag

# diag_embed


# diagflat


# diagonal

# diff


# einsum

# flatten
def flatten(input, start_dim=1, end_dim=-1):
    """Flattens the input. Does not affect the batch size."""
    if end_dim < 0:
        end_dim = input.ndim + end_dim
    new_shape = input.shape[:start_dim] + (-1,) + input.shape[end_dim + 1:]
    return ops.reshape(input, new_shape)

# flip


# fliplr


# flipud


# kron


# rot90


# gcd


# histc


# histogram


# histogramdd


# meshgrid
def meshgrid(*tensors, indexing=None):
    return ops.meshgrid(*tensors, indexing)

# lcm


# logcumsumexp

# ravel


# renorm


# repeat_interleave

# roll


# searchsorted
def searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None):
    if USE_PYBOOST:
        return mindspore.mint.searchsorted(sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter)
    return ops.searchsorted(sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter)

# tensordot

# trace

# tril

# tril_indices

# triu

# triu_indices

# unflatten
def unflatten(x, dim, sizes):
    new_shape = x.shape[:dim] + sizes
    return ops.reshape(x, new_shape)

# vander


# view_as_real

# view_as_complex


# resolve_conj


# resolve_neg
