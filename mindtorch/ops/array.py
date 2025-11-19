"""array op"""
import numbers
import operator
import builtins
import numpy as np

import mindspore
from mindspore import ops, mint
import mindtorch
from mindtorch.executor import execute
from .other import broadcast_tensors, broadcast_to


def t(input):
    assert input.ndim <= 2
    if input.ndim == 2:
        return transpose(input, 0, 1)
    return input

# adjoint


# argwhere
def argwhere(input):
    return execute("non_zero", input)

def infer_dtype(dtypes):
    is_float_dtypes = [d.is_floating_point for d in dtypes]
    float_dtypes = [d for d in dtypes if d.is_floating_point]
    if any(is_float_dtypes):
        return max(float_dtypes)
    else:
        return max(dtypes)

# cat
def cat(tensors, dim=0, **kwargs):
    dim = kwargs.pop('axis', dim)
    # dtype = infer_dtype([t.dtype for t in tensors])
    # tensors = [t.to(dtype) for t in tensors if t.shape != (0,)]
    # tensors = [t for t in tensors if t.numel() != 0]
    return execute("concat", tensors, dim, device_from_list=True)


# concat
def concat(tensors, dim=0, **kwargs):
    dim = kwargs.pop('axis', dim)
    return cat(tensors, dim)


# concatenate
def concatenate(tensors, dim=0, **kwargs):
    dim = kwargs.pop('axis', dim)
    return cat(tensors, dim)


# conj
def conj(input):
    return execute("conj", input)


# chunk
def chunk(input, chunks, dim=0):
    return execute("chunk", input, chunks, dim)


# dsplit


# column_stack


# dstack


# gather
def gather(input, dim, index):
    # if ON_ORANGE_PI:
    #     return gather_with_index_select(input, dim, index)
    return execute("gather_d", input, dim, index)

def gather_with_index_select(x, dim, index):
    # 获取所有维度的索引
    idx = mindtorch.meshgrid(*[mindtorch.arange(s) for s in index.shape], indexing='ij')
    
    # 替换目标维度的索引
    new_idx = ()
    for ix, i in enumerate(idx):
        if ix == dim:
            new_idx += (index,)
        else:
            new_idx += (i,)
    
    # 使用高级索引提取数据
    return x[new_idx]

def gather_nd(input, indices):
    return execute("gather_nd", input, indices)


# hsplit


# hstack

# index_fill


# index_add
def index_add(input, dim, index, source, *, alpha=1):
    return execute("index_add_ext", input, dim, index, source, alpha)


# index_copy


# index_reduce


# index_select
def index_select(input, dim, index):
    return execute("index_select", input, dim, index)

# masked_select
def masked_select(input, mask):
    return execute("masked_select", input, mask)


# movedim
def movedim(x, source, destination):
    """
    Swap two dimensions of the input tensor.

    Args:
        x (Tensor): The input tensor.
        source (Union[int, sequence[int]]): Original dimensions.
        destination (Union[int, sequence[int]]): Destination positions for each of the original dimensions.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case1 : moving single axis
        >>> import mindspore
        >>> x = mindspore.tensor(mindspore.ops.zeros((3, 4, 5)))
        >>> output = mindspore.ops.movedim(x, 0, -1)
        >>> print(output.shape)
        (4, 5, 3)
        >>> # case 2 : moving multiple axes
        >>> x = mindspore.tensor(mindspore.ops.zeros((3, 4, 5)))
        >>> output = mindspore.ops.movedim(x, (0, 2), (1, 2))
        >>> print(output.shape)
        (4, 3, 5)
    """
    return movedim(x, source, destination)

# moveaxis
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
        r = mindtorch.range(0, a_rank, 1)

        def _remove_indices(a, b):
            """Remove indices (`b`) from `a`."""
            items = mindtorch.unbind(
                mindtorch.sort(mindtorch.stack(b))
            )

            i = 0
            result = []

            for item in items:
                result.append(a[i:item])
                i = item + 1

            result.append(a[i:])

            return mindtorch.concat(result, 0)

        minus_sources = _remove_indices(r, source)
        minus_dest = _remove_indices(r, destination)

        perm = execute('scatter_nd', 
            mindtorch.unsqueeze(minus_dest, 1), minus_sources, [a_rank]
        )
        perm = execute('tensor_scatter_update',
            perm, mindtorch.unsqueeze(destination, 1), source
        )
    a = mindtorch.permute(a, tuple(perm))

    return a

# narrow
def narrow(input, dim, start, length):
    length = length.item() if not isinstance(length, int) else length
    start = start.item() if not isinstance(start, int) else start
    return execute("narrow", input, dim, start, length)


# narrow_copy


# nonzero
def nonzero(input, *, as_tuple=False):
    if as_tuple:
        return execute("non_zero_ext", input)
    return execute("non_zero", input)


# permute
def permute(input, dims):
    assert isinstance(dims, tuple)
    return execute("permute", input, dims)


# reshape
def reshape(input, *shape):
    if isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    new_shape = ()
    for s in shape:
        if not isinstance(s, numbers.Number) or isinstance(s, np.int64):
            s = s.item()
        new_shape += (s,)
    return execute("reshape", input, new_shape)


def view(input, *shape):
    return reshape(input, shape)


# row_stack


# select
def select(input, dim, index):
    return execute("select_ext", input, dim, index)


# scatter
def scatter(input, dim, index, src):
    if input.dtype == mindspore.bool_:
        return execute("scatter", input.int(), dim, index, src.int()).bool()

    if not isinstance(src, mindtorch.Tensor):
        return execute("scatter_value", input, dim, index, src)
    return execute("scatter", input, dim, index, src)


# diagonal_scatter


# select_scatter


# slice_scatter


# scatter_add
def scatter_add(input, dim, index, src):
    return execute("scatter_add_ext", input, dim, index, src)


# scatter_reduce
def scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
    if reduce == 'sum':
        return scatter_add(input, dim, index, src)
    else:
        raise ValueError(f'do not support reduce: {reduce}')


# split
def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        res = execute("split_tensor", tensor, split_size_or_sections, dim)
    elif isinstance(split_size_or_sections, (list, tuple)):
        split_size_or_sections = tuple(s.item() if isinstance(s, mindtorch.Tensor) else s for s in split_size_or_sections)
        res = execute("split_with_size", tensor, split_size_or_sections, dim)
    else:
        raise TypeError(
            f"Type of Argument `split_size_or_sections` should be integer, tuple(int) or list(int), "
            f"but got {type(split_size_or_sections)}"
        )
    return res

def split_with_sizes(input, split_sizes, dim=0):
    return execute("split_with_size", input, split_sizes, dim)

# squeeze
def squeeze(input, dim=None):
    if dim is None:
        dim = ()
    return execute("squeeze", input, dim)

# stack


def stack(tensors, dim=0):
    return execute("stack", tensors, dim, device_from_list=True)


# swapaxes
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
    return gather(input, 0, index).view(index_shape)


# take_along_dim
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
    input = input.clone() # input wiil be modified on CPU
    if dim:
        self_broadcasted, indices_broadcasted, dim = _take_along_dim_helper(input, indices, dim)
        return gather(self_broadcasted, dim, indices_broadcasted)
    return input.view(-1).gather(0, indices.view(-1))

# tensor_split
def tensor_split(input, indices_or_sections, dim=0):
    if isinstance(indices_or_sections, int):
        # 分割成大致相等的部分
        dim_size = input.size(dim)
        if dim_size == 0:
            return [input] * indices_or_sections
        split_size = (dim_size + indices_or_sections - 1) // indices_or_sections
        return split(input, split_size, dim=dim)
    elif isinstance(indices_or_sections, (list, tuple, mindtorch.Tensor)):
        # 按照给定的索引分割
        dim_size = input.size(dim)
        indices = [0] + list(indices_or_sections) + [dim_size]
        split_sizes = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        return split(input, split_sizes, dim=dim)
    else:
        raise ValueError("indices_or_sections must be int or list/tuple of indices")

# tile
def tile(input, dims):
    if isinstance(dims[0], (tuple, list)):
        dims = dims[0]

    new_dims = ()
    for d in dims:
        if not isinstance(d, int):
            d = d.item()
        new_dims += (d,)
    return execute("tile", input, tuple(new_dims))


# transpose
def transpose(input, dim0, dim1):
    return execute("transpose_view", input, dim0, dim1)


# unbind
def unbind(input, dim=0):
    return execute("unstack_view", input, dim)


# unravel_index


# unsqueeze
def unsqueeze(input, dim):
    return execute("expand_dims", input, dim)


# vsplit

# vstack


# where
def where(condition, input=None, other=None):
    if input is None and other is None:
        return nonzero(condition, as_tuple=True)
    return execute("select", condition, input, other)


__all__ = [
    # adjoint,
    "argwhere",
    "cat",
    "concat",
    "concatenate",
    "conj",
    "chunk",
    # dsplit,
    # column_stack
    # dstack
    "gather",
    "gather_nd",
    # hsplit
    "index_add",
    # index_copy
    # index_reduce
    "index_select",
    "masked_select",
    "movedim",
    "moveaxis",
    "narrow",
    # narrow_copy
    "nonzero",
    "permute",
    "reshape",
    "view",
    # row_stack
    "select",
    "scatter",
    # diagonal_scatter
    # select_scatter
    # slice_scatter
    "scatter_add",
    "scatter_reduce",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "swapdims",
    "take",
    "take_along_dim",
    "tensor_split",
    "tile",
    "transpose",
    "unbind",
    # unravel_index
    "unsqueeze",
    # vsplit
    "where",
    't',
    'split_with_sizes',
]
