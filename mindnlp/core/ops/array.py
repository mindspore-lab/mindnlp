"""array op"""
import mindspore
from mindspore import ops

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

# hsplit


# hstack


# index_add


# index_copy


# index_reduce


# index_select


# masked_select


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
    return ops.nonzero(input, as_tuple)

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
        reduce = mindspore.ops.auto_generate.gen_arg_handler.str_to_enum('Scatter', 'reduce', "none")
        return mindspore.ops.auto_generate.gen_ops_prim.scatter_op(input, dim, index, src, reduce)
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
