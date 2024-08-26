"""array op"""
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim

from mindnlp.configs import USE_PYBOOST

# adjoint

# argwhere
def argwhere(input):
    if USE_PYBOOST:
        return mindspore.mint.nonzero(input)
    return ops.argwhere(input)

# cat
def cat(tensors, dim=0):
    if USE_PYBOOST:
        return mindspore.mint.cat(tensors, dim)
    return ops.cat(tensors, dim)

# concat
def concat(tensors, dim=0):
    return cat(tensors, dim)

# concatenate
def concatenate(tensors, dim=0):
    return cat(tensors, dim)

# conj
def conj(input):
    return ops.conj(input)

# chunk
def chunk(input, chunks, dim=0):
    return ops.chunk(input, chunks, dim)

# dsplit


# column_stack


# dstack


# gather
def gather(input, dim, index):
    if USE_PYBOOST:
        return mindspore.mint.gather(input, dim, index)
    return ops.gather_elements(input, dim, index)

def gather_nd(input, indices):
    return ops.gather_nd(input, indices)

# hsplit


# hstack
def hstack(tensors):
    return ops.hstack(tensors)


# index_fill
def index_fill(input, dim, index, value):
    return ops.index_fill(input, dim, index, value)

# index_add
def index_add(input, dim, index, source, *, alpha=1):
    if USE_PYBOOST:
        return mindspore.ops.auto_generate.gen_ops_prim.index_add_ext_op(input, index, source, dim, alpha)
    return ops.index_add(input, index, source, dim)

def inplace_index_add(input, dim, index, source):
    _inplace = _get_cache_prim(ops.InplaceIndexAdd)(dim)
    return _inplace(input, index, source)

# index_copy


# index_reduce


# index_select
def index_select(input, dim, index):
    if USE_PYBOOST:
        return mindspore.mint.index_select(input, dim, index)
    return ops.index_select(input, dim, index)

# masked_select
def masked_select(input, mask):
    return ops.masked_select(input, mask)

# movedim


# moveaxis


# narrow
def narrow(input, dim, start, length):
    if USE_PYBOOST:
        return mindspore.mint.narrow(input, dim, start, length)
    return ops.narrow(input, dim, start, length)

# narrow_copy


# nonzero
def nonzero(input, *, as_tuple=False):
    if USE_PYBOOST:
        return mindspore.mint.nonzero(input, as_tuple)
    _nonzero = _get_cache_prim(ops.NonZero)()
    out = _nonzero(input)
    if as_tuple:
        if 0 in out.shape:
            return (out, out)
        return unbind(out, 1)
    return out

# permute
def permute(input, dims):
    if USE_PYBOOST:
        return mindspore.mint.permute(input, dims)
    return ops.permute(input, dims)

# reshape
def reshape(input, shape):
    return ops.reshape(input, shape)

def view(input, *shape):
    # if USE_PYBOOST:
    #     return mindspore.ops.auto_generate.gen_ops_prim.view_op(input, shape)
    return ops.reshape(input, shape)

# row_stack

# select
def select(input, dim, index):
    slices = ()
    for _ in range(dim):
        slices += (slice(None, None, None),)
    slices += (index,)
    return input[slices]

# scatter
def scatter(input, dim, index, src):
    if USE_PYBOOST:
        return mindspore.ops.auto_generate.gen_ops_prim.scatter_op(input, dim, index, src, 0)
    if not isinstance(src, mindspore.Tensor):
        src = ops.full(index.shape, src, dtype=input.dtype)
    return ops.tensor_scatter_elements(input, index, src, dim)

# diagonal_scatter


# select_scatter


# slice_scatter


# scatter_add
def scatter_add(input, dim, index, src):
    if USE_PYBOOST:
        return mindspore.mint.scatter_add(input, dim, index, src)
    return ops.tensor_scatter_elements(input, index, src, dim, 'add')

# scatter_reduce


# scatter_nd_update
def scatter_nd_update(input, indices, update):
    return ops.scatter_nd_update(input, indices, update)


def scatter_update(input, indices, updates):
    return ops.scatter_update(input, indices, updates)

# split
def split(tensor, split_size_or_sections, dim=0):
    if USE_PYBOOST:
        return mindspore.mint.split(tensor, split_size_or_sections, dim)
    return ops.split(tensor, split_size_or_sections, dim)

# squeeze
def squeeze(input, dim=None):
    return ops.squeeze(input, dim)

# stack
def stack(tensors, dim=0):
    if USE_PYBOOST:
        return mindspore.mint.stack(tensors, dim)
    return ops.stack(tensors, dim)

# swapaxes
def swapaxes(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# swapdims
def swapdims(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# take


# take_along_dim


# tensor_split


# tile
def tile(input, dims):
    if USE_PYBOOST:
        return mindspore.mint.tile(input, dims)
    return ops.tile(input, dims)

# transpose
def transpose(input, dim0, dim1):
    ranks = list(range(input.ndim))
    rank0 = ranks[dim0]
    rank1 = ranks[dim1]
    ranks[dim0] = rank1
    ranks[dim1] = rank0
    if USE_PYBOOST:
        return mindspore.ops.auto_generate.gen_ops_prim.transpose_op(input, tuple(ranks))
    return permute(input, tuple(ranks))

# unbind
def unbind(input, dim=0):
    return ops.unbind(input, dim)

# unravel_index

# unsqueeze
def unsqueeze(input, dim):
    return ops.expand_dims(input, dim)

# vsplit

# vstack
def vstack(input):
    return ops.vstack(input)


# where
def where(condition, input, other):
    if USE_PYBOOST:
        return mindspore.mint.where(condition, input, other)
    return ops.where(condition, input, other)

def _slice_helper(slice_spec):
    if not isinstance(slice_spec, (list, tuple)):
        slice_spec = [slice_spec]

    begin, end, strides = (), (), ()
    index = 0

    new_axis_mask, shrink_axis_mask = 0, 0
    begin_mask, end_mask = 0, 0
    ellipsis_mask = 0

    for s in slice_spec:
        if isinstance(s, slice):
            if s.start is not None:
                begin += (s.start,)
            else:
                begin += (0,)
                begin_mask |= (1 << index)

            if s.stop is not None:
                end += (s.stop,)
            else:
                end += (0,)
                end_mask |= (1 << index)

            if s.step is not None:
                strides += (s.step,)
            else:
                strides += (1,)
        elif s is Ellipsis:
            begin += (0,)
            end += (0,)
            strides += (1,)
            ellipsis_mask |= (1 << index)
        elif s is None:
            # begin += (0,)
            # end += (0,)
            # strides += (1,)
            new_axis_mask |= (1 << index)
        else:
            begin += (s,)
            end += (s + 1,)
            strides += (1,)
            shrink_axis_mask |= (1 << index)
        index += 1

    return begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask

def getitem(tensor, slice):
    slices = _slice_helper(slice)
    # input_x, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0
    return ops.strided_slice(tensor, *slices)

def tensor_scatter_add(input, indeices, updates):
    return ops.tensor_scatter_add(input, indeices, updates)

def tensor_scatter_max(input, indeices, updates):
    return ops.tensor_scatter_max(input, indeices, updates)

def tensor_scatter_min(input, indeices, updates):
    return ops.tensor_scatter_min(input, indeices, updates)
