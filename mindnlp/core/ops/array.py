"""array op"""
import numbers
import numpy as np
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations._grad_ops import StridedSliceGrad
from mindspore.ops.auto_generate.gen_ops_prim import inplace_scatter_src_reduce_op

from ..configs import use_pyboost, ON_ORANGE_PI
from .other import broadcast_tensors, finfo
from ._inner import call_ms_func

# adjoint

# argwhere
def argwhere(input):
    if use_pyboost():
        return mindspore.mint.nonzero(input)
    return ops.argwhere(input)

# cat
has_cat = hasattr(mindspore.mint, 'cat')
def cat(tensors, dim=0, *, out=None, **kwargs):
    axis = kwargs.get('axis', None)
    if axis is not None:
        dim = axis
    max_dtype = max([x.dtype for x in tensors])
    tensors = [x.to(max_dtype) for x in tensors]
    if use_pyboost() and has_cat:
        return call_ms_func(mindspore.mint.cat, tensors, dim, out=out)
    return call_ms_func(ops.cat, tensors, dim, out=out)

# concat
has_concat = hasattr(mindspore.mint, 'concat')
def concat(tensors, dim=0, *, out=None, **kwargs):
    return cat(tensors, dim, out=out, **kwargs)

# concatenate
def concatenate(tensors, dim=0, out=None, **kwargs):
    return cat(tensors, dim, out=out, **kwargs)

# conj
def conj(input):
    return ops.conj(input)

# chunk
has_chunk = hasattr(mindspore.mint, 'chunk')
def chunk(input, chunks, dim=0):
    if use_pyboost() and has_chunk:
        return mindspore.mint.chunk(input, chunks, dim)
    return ops.chunk(input, chunks, dim)

# dsplit


# column_stack


# dstack


# gather
has_gather = hasattr(mindspore.mint, 'gather')
def gather(input, dim, index):
    is_complex = input.dtype == mindspore.complex64
    if is_complex:
        real_part = mindspore.mint.gather(input.real, dim, index)
        imag_part = mindspore.mint.gather(input.imag, dim, index)
        _complex = _get_cache_prim(ops.Complex)()
        return _complex(real_part, imag_part)

    if use_pyboost() and has_gather:
        return mindspore.mint.gather(input, dim, index)

    index = ops.where(index < input.shape[dim], index, index - input.shape[dim])
    return ops.gather_elements(input, dim, index)

def gather_nd(input, indices):
    return ops.gather_nd(input, indices)

def tf_gather(input, indices, axis, batch_dims=0):
    return ops.gather(input, indices, axis, batch_dims)

# hsplit


# hstack
def hstack(tensors):
    return ops.hstack(tensors)


# index_fill
def index_fill(input, dim, index, value):
    return ops.index_fill(input, dim, index, value)

# index_add
def index_add(input, dim, index, source, *, alpha=1):
    if use_pyboost():
        return mindspore.mint.index_add(input, dim, index, source, alpha=alpha)
    return ops.index_add(input, index, source, dim)

# index_copy


# index_reduce


# index_select
has_index_select = hasattr(mindspore.mint, 'index_select')
def index_select(input, dim, index, *, out=None):
    if use_pyboost() and has_index_select:
        return call_ms_func(mindspore.mint.index_select, input, dim, index, out=out)
    return call_ms_func(ops.index_select, input, dim, index, out=out)

# masked_select
has_masked_select = hasattr(mindspore.mint, 'masked_select')
def masked_select(input, mask, *, out=None):
    if use_pyboost() and has_masked_select:
        return call_ms_func(mindspore.mint.masked_select, input, mask, out=out)
    return call_ms_func(ops.masked_select, input, mask, out=out)

# movedim


# moveaxis


# narrow
has_narrow = hasattr(mindspore.mint, 'narrow')
def narrow(input, dim, start, length):
    length = length.item() if isinstance(length, mindspore.Tensor) else length
    if use_pyboost() and has_narrow:
        return mindspore.mint.narrow(input, dim, start, length)
    return ops.narrow(input, dim, start, length)

# narrow_copy


# nonzero
has_nonzero = hasattr(mindspore.mint, 'nonzero')
def nonzero(input, *, as_tuple=False):
    if use_pyboost() and has_nonzero:
        return mindspore.mint.nonzero(input, as_tuple=as_tuple)
    _nonzero = _get_cache_prim(ops.NonZero)()
    out = _nonzero(input)
    if as_tuple:
        if 0 in out.shape:
            return (out, out)
        return unbind(out, 1)
    return out

# permute
has_permute = hasattr(mindspore.mint, 'permute')
def permute(input, dims):
    if use_pyboost() and has_permute:
        return mindspore.mint.permute(input, dims)
    return ops.permute(input, dims)

# reshape
has_reshape = hasattr(mindspore.mint, 'reshape')
def reshape(input, *shape, **kwargs):
    shape = kwargs.pop('shape', shape)
    if isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    new_shape = ()
    for s in shape:
        if not isinstance(s, int):
            s = s.item()
        new_shape += (s,)
    if use_pyboost() and has_reshape:
        return mindspore.mint.reshape(input, new_shape)
    return ops.reshape(input, new_shape)

def view(input, *shape):
    # if use_pyboost():
    #     return mindspore.ops.auto_generate.gen_ops_prim.view_op(input, shape)
    return reshape(input, shape)

# row_stack

# select
has_select = hasattr(mindspore.mint, 'select')
def select(input, dim, index):
    if use_pyboost() and has_select:
        return mindspore.mint.select(input, dim, index)
    slices = ()
    for _ in range(dim):
        slices += (slice(None, None, None),)
    slices += (index,)
    return input[slices]

# scatter
has_scatter = hasattr(mindspore.mint, 'scatter')
def scatter(input, dim, index, src):
    if use_pyboost() and has_scatter:
        return mindspore.mint.scatter(input, dim, index, src)
    if not isinstance(src, mindspore.Tensor):
        src = ops.full(index.shape, src, dtype=input.dtype)
    return ops.tensor_scatter_elements(input, index, src, dim)

def tf_scatter_nd_update(input, indices, updates):
    return ops.scatter_nd_update(input, indices, updates)

def tf_scatter_nd(indices, updates, shape):
    return ops.scatter_nd(indices, updates, shape)

# diagonal_scatter


# select_scatter


# slice_scatter


# scatter_add
has_scatter_add = hasattr(mindspore.mint, 'scatter_add')
def scatter_add(input, dim, index, src):
    if use_pyboost() and has_scatter_add:
        return mindspore.mint.scatter_add(input, dim, index, src)
    return ops.tensor_scatter_elements(input, index, src, dim, 'add')

scatter_reduce_dict = {
    'sum': 'add',
    'amax': 'max',
    'amin': 'min',
    'mean': 'mean'
}
# scatter_reduce
def scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
    return inplace_scatter_src_reduce_op(input.clone(), dim, index, src, scatter_reduce_dict[reduce])

# scatter_nd_update
def scatter_nd_update(input, indices, update):
    return ops.scatter_nd_update(input, indices, update)


def scatter_update(input, indices, updates):
    return ops.scatter_update(input, indices, updates)

# split
has_split = hasattr(mindspore.mint, 'split')
def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, (tuple, list)):
        new_split_size_or_sections = ()
        for s in split_size_or_sections:
            if not isinstance(s, int):
                s = s.item()
            new_split_size_or_sections += (s,)
        split_size_or_sections = new_split_size_or_sections
    if use_pyboost() and has_split:
        return mindspore.mint.split(tensor, split_size_or_sections, dim)
    return ops.split(tensor, split_size_or_sections, dim)

def split_with_sizes(input, split_sizes, dim=0):
    assert input.dim() != 0, "split expects at least a 1-dimensional tensor"
    dim_size = input.size(dim)
    num_splits = len(split_sizes)
    start_idx = 0

    splits = []
    for i in range(num_splits):
        length = split_sizes[i]
        assert length >= 0, f"split_with_sizes expects split_sizes have only non-negative entries, but got split_sizes={split_sizes}"
        splits.append(
            narrow(input, dim, start_idx, length)
        )
        start_idx += length

    return splits


# squeeze
has_squeeze = hasattr(mindspore.mint, 'squeeze')
def squeeze(input, *dim, **kwargs):
    dim = kwargs.get('dim', dim)
    if use_pyboost() and has_squeeze:
        return mindspore.mint.squeeze(input, dim)
    return ops.squeeze(input, dim)

# stack
has_stack = hasattr(mindspore.mint, 'stack')
def stack(tensors, dim=0, *, out=None, **kwargs):
    dim = kwargs.pop('axis', dim)
    if use_pyboost() and has_stack:
        return call_ms_func(mindspore.mint.stack, tensors, dim, out=out)
    return call_ms_func(ops.stack, tensors, dim, out=out)

# swapaxes
has_swapaxes = hasattr(mindspore.mint, 'swapaxes')
def swapaxes(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# swapdims
def swapdims(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# take
def take(input, index):
    input = input.view(-1)
    index_shape = index.shape
    index = index.view(-1)
    if ON_ORANGE_PI:
        return tf_gather(input, index, 0).view(index_shape)
    if index_shape == ():
        return gather(input, 0, index)[0]
    return gather(input, 0, index).view(index_shape)

def infer_size_impl(a, b):
    lenA = len(a)
    lenB = len(b)
    ndim = max(lenA, lenB)
    expanded_sizes = [0] * ndim

    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = lenA - 1 - offset
        dimB = lenB - 1 - offset
        
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # 检查维度兼容性
        if not (sizeA == sizeB or sizeA == 1 or sizeB == 1):
            raise RuntimeError(
                f"The size of tensor a ({sizeA}) must match the size of tensor b ({sizeB}) "
                f"at non-singleton dimension {i}"
            )

        # 应用广播规则：优先选择非1的维度大小
        expanded_sizes[i] = sizeB if sizeA == 1 else sizeA

    return expanded_sizes


def _take_along_dim_helper(self, indices, dim):
    assert self.dim() == indices.dim(), f"torch.take_along_dim(): input and indices should have the same number of dimensions, " \
        f"but got {self.dim()} dimensions for input, and {indices.dim()} dimensions for indices"
    dim = self.dim() + dim if dim < 0 else dim
    self_sizes = list(self.shape)
    self_sizes[dim] = indices.size(dim)
    broadcast_shape = infer_size_impl(self_sizes, indices.shape)
    indices_broadcasted = indices.broadcast_to(broadcast_shape)

    indices_sizes = list(indices.shape)
    indices_sizes[dim] = self.size(dim)
    broadcast_shape = infer_size_impl(indices_sizes, self.shape)
    self_broadcasted = self.broadcast_to(broadcast_shape)

    return self_broadcasted, indices_broadcasted, dim

# take_along_dim
def take_along_dim(input, indices, dim=None, *, out=None):
    if dim:
        self_broadcasted, indices_broadcasted, dim = _take_along_dim_helper(input, indices, dim)
        return self_broadcasted.gather(dim, indices_broadcasted)
    return input.view(-1).gather(0, indices.view(-1))

# tensor_split
def tensor_split(input, indices_or_sections, dim=0):
    return ops.tensor_split(input, indices_or_sections, dim)

# tile
has_tile = hasattr(mindspore.mint, 'tile')
def tile(input, *dims):
    if isinstance(dims[0], tuple):
        dims = dims[0]
    if use_pyboost() and has_tile:
        return mindspore.mint.tile(input, dims)
    return ops.tile(input, dims)

# transpose
has_transpose = hasattr(mindspore.mint, 'transpose')
def transpose(input, dim0, dim1):
    if use_pyboost() and has_transpose:
        return mindspore.mint.transpose(input, dim0, dim1)
    ranks = list(range(input.ndim))
    rank0 = ranks[dim0]
    rank1 = ranks[dim1]
    ranks[dim0] = rank1
    ranks[dim1] = rank0
    return permute(input, tuple(ranks))

def t(input):
    assert input.ndim <= 2, 'Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.'
    if input.ndim == 1:
        return input
    return transpose(input, 0, 1)

# unbind
has_unbind = hasattr(mindspore.mint, 'unbind')
def unbind(input, dim=0):
    if use_pyboost() and has_unbind:
        return mindspore.mint.unbind(input, dim)
    return ops.unbind(input, dim)

# unravel_index

# unsqueeze
has_unsqueeze = hasattr(mindspore.mint, 'unsqueeze')
def unsqueeze(input, dim=None):
    if use_pyboost() and has_unsqueeze:
        return mindspore.mint.unsqueeze(input, dim)
    return ops.expand_dims(input, dim)

# vsplit

# vstack
def vstack(input):
    return ops.vstack(input)

_SLICE_ERROR = (
    'only integers, slices (`:`), ellipsis (`...`), '
    'newaxis (`None`) and integer or boolean arrays are valid indices'
)

# where
def where(condition, *args, out=None):
    if len(args) == 0:
        return nonzero(condition, as_tuple=True)
    assert len(args) == 2
    input, other = args

    if isinstance(input, float) and input == -float("inf"):
        input = finfo(other.dtype).min
    if isinstance(other, float) and other == -float("inf"):
        if isinstance(input, numbers.Number):
            input = mindspore.tensor(input, dtype=mindspore.float32)
        other = finfo(input.dtype).min

    output = mindspore.mint.where(condition, input, other)
    if out is not None:
        out.assign_value(output)
    return output

def _as_index(idx, need_scalar=True):
    """Helper function to parse idx as an index.
    """
    if isinstance(idx, numbers.Integral):
        return idx, True

    idx = mindspore.Tensor(idx)
    if need_scalar and idx.ndim not in (None, 0):
        raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))

    if idx.ndim == 0:
        return idx.item(), True
    return idx, False


def cumprod(x, axis=0, exclusive=False, reverse=False):
    x = np.array(x)
    if reverse:
        x = np.flip(x, axis=axis)

    if exclusive:
        shifted_x = np.ones_like(x)
        if axis == 0:
            shifted_x[1:] = x[:-1]
        else:
            shifted_x[:, 1:] = x[:, :-1]
        result = np.cumprod(shifted_x, axis=axis)
    else:
        result = np.cumprod(x, axis=axis)

    if reverse:
        result = np.flip(result, axis=axis)

    return result

def moveaxis(a, source, destination):
    """Raises ValueError if source, destination not in (-ndim(a), ndim(a))."""
    if not source and not destination:
        return a

    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    if len(source) != len(destination):
        raise ValueError('The lengths of source and destination must equal')

    a_rank = a.ndim

    def _correct_axis(axis, rank):
        if axis < 0:
            return axis + rank
        return axis

    source = tuple(_correct_axis(axis, a_rank) for axis in source)
    destination = tuple(_correct_axis(axis, a_rank) for axis in destination)

    if a.ndim is not None:
        perm = [i for i in range(a_rank) if i not in source]
        for dest, src in sorted(zip(destination, source)):
            assert dest <= len(perm)
            perm.insert(dest, src)
    else:
        r = ops.range(0, a_rank, 1)

        def _remove_indices(a, b):
            """Remove indices (`b`) from `a`."""
            items = ops.unstack(
                ops.sort(ops.stack(b))
            )

            i = 0
            result = []

            for item in items:
                result.append(a[i:item])
                i = item + 1

            result.append(a[i:])

            return ops.concat(result, 0)

        minus_sources = _remove_indices(r, source)
        minus_dest = _remove_indices(r, destination)

        perm = ops.scatter_nd(
            ops.expand_dims(minus_dest, 1), minus_sources, [a_rank]
        )
        perm = ops.tensor_scatter_update(
            perm, ops.expand_dims(destination, 1), source
        )
    a = ops.transpose(a, tuple(perm))

    return a

def _slice_helper(tensor, slice_spec, do_update=False, updates=None):
    """Helper function for __getitem__ and _with_index_update_helper.
    """
    begin, end, strides = [], [], []
    new_axis_mask, shrink_axis_mask = 0, 0
    begin_mask, end_mask = 0, 0
    ellipsis_mask = 0
    advanced_indices = []
    shrink_indices = []
    for index, s in enumerate(slice_spec):
        if isinstance(s, slice):
            if s.start is not None:
                begin.append(s.start)
            else:
                begin.append(0)
                begin_mask |= (1 << index)
            if s.stop is not None:
                end.append(s.stop)
            else:
                end.append(0)
                end_mask |= (1 << index)
            if s.step is not None:
                strides.append(s.step)
            else:
                strides.append(1)
        elif s is Ellipsis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            ellipsis_mask |= (1 << index)
        elif s is None:
            # begin.append(0)
            # end.append(0)
            # strides.append(1)
            new_axis_mask |= (1 << index)
        else:
            s, is_scalar = _as_index(s, False)
            if is_scalar:
                begin.append(s)
                end.append(s + 1)
                strides.append(1)
                shrink_axis_mask |= (1 << index)
                shrink_indices.append(index)
            else:
                begin.append(0)
                end.append(0)
                strides.append(1)
                begin_mask |= (1 << index)
                end_mask |= (1 << index)
                advanced_indices.append((index, s, ellipsis_mask != 0))

    if do_update and not advanced_indices:
        return strided_slice_update(
            tensor,
            begin,
            end,
            strides,
            updates,
            begin_mask=begin_mask,
            end_mask=end_mask,
            shrink_axis_mask=shrink_axis_mask,
            new_axis_mask=new_axis_mask,
            ellipsis_mask=ellipsis_mask,
        )
    else:
        if updates is not None:
            original_tensor = tensor
        tensor = ops.strided_slice(
            tensor,
            begin,
            end,
            strides,
            begin_mask=begin_mask,
            end_mask=end_mask,
            shrink_axis_mask=shrink_axis_mask,
            new_axis_mask=new_axis_mask,
            ellipsis_mask=ellipsis_mask,
        )

    if not advanced_indices:
        return tensor

    advanced_indices_map = {}
    for index, data, had_ellipsis in advanced_indices:
        if had_ellipsis:
            num_shrink = len([x for x in shrink_indices if x > index])
            dim = index - len(slice_spec) + num_shrink
        else:
            num_shrink = len([x for x in shrink_indices if x < index])
            dim = index - num_shrink
        advanced_indices_map[dim] = data
    dims = sorted(advanced_indices_map.keys())
    dims_contiguous = True
    if len(dims) > 1:
        if dims[0] < 0 and dims[-1] >= 0:  # not all same sign
            dims_contiguous = False
        else:
            for i in range(len(dims) - 1):
                if dims[i] + 1 != dims[i + 1]:
                    dims_contiguous = False
                    break
    indices = [advanced_indices_map[x] for x in dims]
    indices = broadcast_tensors(*indices)
    stacked_indices = ops.stack(indices, axis=-1)
    # Skip the contiguous-dims optimization for update because there is no
    # tf.*scatter* op that supports the `axis` argument.
    if not dims_contiguous or updates is not None:
        if range(len(dims)) != dims:
            tensor = moveaxis(tensor, dims, range(len(dims)))
        tensor_shape_prefix = mindspore.Tensor(tensor.shape[: len(dims)])
        stacked_indices = where(
            stacked_indices < 0,
            stacked_indices + tensor_shape_prefix,
            stacked_indices,
        )
        if updates is None:
            return ops.gather_nd(tensor, stacked_indices)
        else:
            # We only need to move-axis `updates` in the contiguous case becausce
            # only in this case the result dimensions of advanced indexing are in
            # the middle of `updates`. In the non-contiguous case, those dimensions
            # are always at the front.
            if dims_contiguous:
                batch_size = stacked_indices.ndim - 1
                batch_start = dims[0]
                if batch_start < 0:
                    batch_start += len(dims) - batch_size

                def range_(start, length):
                    return range(start, start + length)

                updates = moveaxis(
                    updates, range_(batch_start, batch_size), range(batch_size)
                )
            tensor = ops.tensor_scatter_update(tensor, stacked_indices, updates)
            if range(len(dims)) != dims:
                tensor = moveaxis(tensor, range(len(dims)), dims)
            return strided_slice_update(
                original_tensor,
                begin,
                end,
                strides,
                tensor,
                begin_mask=begin_mask,
                end_mask=end_mask,
                shrink_axis_mask=shrink_axis_mask,
                new_axis_mask=new_axis_mask,
                ellipsis_mask=ellipsis_mask,
            )

    # Note that gather_nd does not support gathering from inside the array.
    # To avoid shuffling data back and forth, we transform the indices and
    # do a gather instead.
    rank = tensor.ndim
    dims = [(x + rank if x < 0 else x) for x in dims]
    shape_tensor = tensor.shape
    dim_sizes = np.take_along_axis(np.array(shape_tensor), np.array(dims), axis=0)
    if len(dims) == 1:
        stacked_indices = indices[0]
    stacked_indices = ops.cast(stacked_indices, mindspore.int32)
    stacked_indices = where(
        stacked_indices < 0, stacked_indices + mindspore.Tensor(dim_sizes), stacked_indices
    )
    axis = dims[0]
    if len(dims) > 1:
        index_scaling = cumprod(dim_sizes, reverse=True, exclusive=True)

        def _tensordot(a, b):
            # TODO(b/168657656): This function should be replaced by
            # tensordot(axis=1) once MatMul has int32 XLA kernel.
            b = ops.broadcast_to(b, a.shape)
            return ops.sum(a * b, dim=-1)

        stacked_indices = _tensordot(stacked_indices, mindspore.Tensor(index_scaling))
        flat_shape = shape_tensor[:axis] + (-1,) + shape_tensor[axis + len(dims) :]
        tensor = ops.reshape(tensor, flat_shape)

    return ops.gather(tensor, stacked_indices, axis=axis)

def _as_spec_tuple(slice_spec):
    """Convert slice_spec to tuple."""
    if isinstance(slice_spec, (list, tuple)):
        is_index = True
        for s in slice_spec:
            if s is None or s is Ellipsis or isinstance(s, (list, tuple, slice)):
                is_index = False
                break
        if not is_index:
            return tuple(slice_spec)
    return (slice_spec,)

def getitem(self, slice_spec):
    if (
        isinstance(slice_spec, bool)
        or (
            isinstance(slice_spec, mindspore.Tensor)
            and slice_spec.dtype == mindspore.bool_
        )
    ):
        return ops.boolean_mask(tensor=self, mask=slice_spec)

    if not isinstance(slice_spec, tuple):
        slice_spec = _as_spec_tuple(slice_spec)

    result_t = _slice_helper(self, slice_spec)
    return result_t

def setitem(a, slice_spec, updates):
    """Implementation of ndarray._with_index_*."""
    if (
        isinstance(slice_spec, bool)
        or (
            isinstance(slice_spec, mindspore.Tensor)
            and slice_spec.dtype == mindspore.bool_
        )
    ):
        slice_spec = nonzero(slice_spec)

    if not isinstance(slice_spec, tuple):
        slice_spec = _as_spec_tuple(slice_spec)

    a_dtype = a.dtype
    result_t = _slice_helper(a, slice_spec, True, updates)
    return result_t.astype(a_dtype)

def tensor_scatter_add(input, indeices, updates):
    return ops.tensor_scatter_add(input, indeices, updates)

def tensor_scatter_max(input, indeices, updates):
    return ops.tensor_scatter_max(input, indeices, updates)

def tensor_scatter_min(input, indeices, updates):
    return ops.tensor_scatter_min(input, indeices, updates)

def strided_slice_update(input, begin, end, strides, update, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    strided_slice_grad = _get_cache_prim(StridedSliceGrad)(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    updated_tensor = strided_slice_grad(update, input.shape, begin, end, strides)
    return ops.assign(input, where(updated_tensor != 0, updated_tensor, input))

__all__ = [
    # adjoint,
    'argwhere',
    'cat',
    'concat',
    'concatenate',
    'conj',
    'chunk',
    # dsplit,
    # column_stack
    # dstack
    'gather',
    'gather_nd',
    'tf_gather',
    # hsplit
    'hstack',
    'index_fill',
    'index_add',
    # index_copy
    # index_reduce
    'index_select',
    'masked_select',
    # movedim
    # moveaxis
    'narrow',
    # narrow_copy
    'nonzero',
    'permute',
    'reshape',
    'view',
    # row_stack
    'select',
    'scatter',
    'tf_scatter_nd_update',
    'tf_scatter_nd',
    # diagonal_scatter
    # select_scatter
    # slice_scatter
    'scatter_add',
    'scatter_reduce',
    'scatter_nd_update',
    'scatter_update',
    'split',
    'split_with_sizes',
    'squeeze',
    'stack',
    'swapaxes',
    'swapdims',
    'take',
    'take_along_dim',
    'tensor_split',
    'tile',
    'transpose',
    't',
    'unbind',
    # unravel_index
    'unsqueeze',
    # vsplit
    'vstack',
    'where',
    'getitem',
    'setitem',
    'tensor_scatter_add',
    'tensor_scatter_max',
    'tensor_scatter_min',
    'strided_slice_update'
]
