"""Meta device backend for shape inference without actual computation.

Meta tensors only track shape, dtype, and device - no actual data storage.
This is useful for:
1. Model initialization without memory allocation
2. Shape inference during tracing
3. Memory estimation before execution
"""

from typing import Tuple, List, Union


def _infer_output_shape_binary(a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Infer output shape for binary operations with broadcasting.

    Follows NumPy/PyTorch broadcasting rules:
    1. Align shapes from the right
    2. For each dimension, output is max(a_dim, b_dim)
    3. Dimensions of size 1 can broadcast to any size

    Args:
        a_shape: Shape of first operand
        b_shape: Shape of second operand

    Returns:
        Output shape after broadcasting

    Raises:
        ValueError: If shapes are not broadcast-compatible
    """
    # Handle scalars
    if len(a_shape) == 0:
        return b_shape
    if len(b_shape) == 0:
        return a_shape

    # Pad shorter shape with 1s on the left
    max_ndim = max(len(a_shape), len(b_shape))
    a_padded = (1,) * (max_ndim - len(a_shape)) + a_shape
    b_padded = (1,) * (max_ndim - len(b_shape)) + b_shape

    result = []
    for a_dim, b_dim in zip(a_padded, b_padded):
        if a_dim == b_dim:
            result.append(a_dim)
        elif a_dim == 1:
            result.append(b_dim)
        elif b_dim == 1:
            result.append(a_dim)
        else:
            raise ValueError(
                f"Shapes {a_shape} and {b_shape} are not broadcast-compatible"
            )

    return tuple(result)


def _infer_matmul_shape(a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Infer output shape for matrix multiplication.

    Handles:
    - 1D @ 1D: () (dot product)
    - 1D @ 2D: (n,)
    - 2D @ 1D: (m,)
    - 2D @ 2D: (m, p)
    - Batched: (..., m, n) @ (..., n, p) -> (..., m, p)

    Args:
        a_shape: Shape of first operand
        b_shape: Shape of second operand

    Returns:
        Output shape after matmul

    Raises:
        ValueError: If shapes are not matmul-compatible
    """
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)

    # 1D @ 1D: dot product
    if a_ndim == 1 and b_ndim == 1:
        if a_shape[0] != b_shape[0]:
            raise ValueError(f"1D matmul requires same size, got {a_shape[0]} and {b_shape[0]}")
        return ()

    # 1D @ 2D: (n,) @ (n, p) -> (p,)
    if a_ndim == 1 and b_ndim == 2:
        if a_shape[0] != b_shape[0]:
            raise ValueError(f"matmul: {a_shape} and {b_shape} are not compatible")
        return (b_shape[1],)

    # 2D @ 1D: (m, n) @ (n,) -> (m,)
    if a_ndim == 2 and b_ndim == 1:
        if a_shape[1] != b_shape[0]:
            raise ValueError(f"matmul: {a_shape} and {b_shape} are not compatible")
        return (a_shape[0],)

    # 2D @ 2D: (m, n) @ (n, p) -> (m, p)
    if a_ndim == 2 and b_ndim == 2:
        if a_shape[1] != b_shape[0]:
            raise ValueError(f"matmul: {a_shape} and {b_shape} are not compatible (inner dims {a_shape[1]} vs {b_shape[0]})")
        return (a_shape[0], b_shape[1])

    # Batched matmul: (..., m, n) @ (..., n, p) -> (..., m, p)
    # Broadcast batch dimensions
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]

    # Get matrix dimensions
    a_m, a_n = a_shape[-2], a_shape[-1]
    b_n, b_p = b_shape[-2], b_shape[-1]

    if a_n != b_n:
        raise ValueError(f"matmul: {a_shape} and {b_shape} are not compatible (inner dims {a_n} vs {b_n})")

    # Broadcast batch dimensions
    batch_shape = _infer_output_shape_binary(a_batch, b_batch)

    return batch_shape + (a_m, b_p)


def _create_meta_tensor(shape: Tuple[int, ...], dtype, device):
    """Create a meta tensor with given shape, dtype, and device.

    Args:
        shape: Output tensor shape
        dtype: Output tensor dtype
        device: Device (should be meta device)

    Returns:
        Meta tensor with the specified properties
    """
    from .._tensor import Tensor, _compute_strides
    from .._storage import TypedStorage
    from .._device import device as device_cls

    # Create storage with no actual data
    storage = TypedStorage.__new__(TypedStorage)
    storage._ms_tensor = None  # No actual data
    storage._size = 0
    storage._dtype = dtype
    storage._device = device_cls("meta")

    # Create tensor with the storage
    result = Tensor.__new__(Tensor)
    result._storage = storage
    result._shape = tuple(shape)
    result._stride = _compute_strides(shape)
    result._storage_offset = 0
    result._dtype = dtype
    result._device = device_cls("meta")
    result._requires_grad = False
    result._grad_fn = None
    result._grad = None
    result._version = 0
    result._hooks = {}
    result._hook_counter = 0

    return result


# --- Binary arithmetic ops ---

def add_meta(a, b):
    """Meta add: compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    # Use dtype from first operand (simplified - real impl would do type promotion)
    return _create_meta_tensor(output_shape, a.dtype, a.device)


def sub_meta(a, b):
    """Meta sub: compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a.dtype, a.device)


def mul_meta(a, b):
    """Meta mul: compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a.dtype, a.device)


def div_meta(a, b):
    """Meta div: compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a.dtype, a.device)


# --- Matrix ops ---

def matmul_meta(a, b):
    """Meta matmul: compute output shape only."""
    output_shape = _infer_matmul_shape(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a.dtype, a.device)


def bmm_meta(a, b):
    """Meta batched matmul: compute output shape only."""
    # bmm requires 3D tensors: (batch, n, m) @ (batch, m, p) -> (batch, n, p)
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise ValueError(f"bmm requires 3D tensors, got {a.shape} and {b.shape}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"bmm batch sizes must match: {a.shape[0]} vs {b.shape[0]}")
    if a.shape[2] != b.shape[1]:
        raise ValueError(f"bmm inner dims must match: {a.shape[2]} vs {b.shape[1]}")

    output_shape = (a.shape[0], a.shape[1], b.shape[2])
    return _create_meta_tensor(output_shape, a.dtype, a.device)


# --- Shape manipulation ops ---

def reshape_meta(a, shape):
    """Meta reshape: compute output shape only."""
    # Handle -1 in shape
    numel = 1
    for s in a.shape:
        numel *= s

    neg_one_idx = None
    known_product = 1
    new_shape = list(shape)

    for i, s in enumerate(new_shape):
        if s == -1:
            if neg_one_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            neg_one_idx = i
        else:
            known_product *= s

    if neg_one_idx is not None:
        new_shape[neg_one_idx] = numel // known_product

    return _create_meta_tensor(tuple(new_shape), a.dtype, a.device)


def transpose_meta(a, dim0, dim1):
    """Meta transpose: compute output shape only."""
    ndim = len(a.shape)

    # Normalize negative dims
    if dim0 < 0:
        dim0 += ndim
    if dim1 < 0:
        dim1 += ndim

    # Swap dimensions
    new_shape = list(a.shape)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

    return _create_meta_tensor(tuple(new_shape), a.dtype, a.device)


def permute_meta(a, dims):
    """Meta permute: compute output shape only."""
    new_shape = tuple(a.shape[d] for d in dims)
    return _create_meta_tensor(new_shape, a.dtype, a.device)


def view_meta(a, *shape):
    """Meta view: compute output shape only."""
    return reshape_meta(a, shape)


# --- Unary ops (shape doesn't change) ---

def neg_meta(a):
    """Meta neg: shape stays the same."""
    return _create_meta_tensor(a.shape, a.dtype, a.device)


def exp_meta(a):
    """Meta exp: shape stays the same."""
    return _create_meta_tensor(a.shape, a.dtype, a.device)


def log_meta(a):
    """Meta log: shape stays the same."""
    return _create_meta_tensor(a.shape, a.dtype, a.device)


def sqrt_meta(a):
    """Meta sqrt: shape stays the same."""
    return _create_meta_tensor(a.shape, a.dtype, a.device)


def abs_meta(a):
    """Meta abs: shape stays the same."""
    return _create_meta_tensor(a.shape, a.dtype, a.device)


def clone_meta(a):
    """Meta clone: shape stays the same."""
    return _create_meta_tensor(a.shape, a.dtype, a.device)


# --- In-place ops (return input tensor, no actual modification) ---

def zero_meta(a):
    """Meta zero_: in-place op returns input tensor."""
    return a


def fill_meta(a, value):
    """Meta fill_: in-place op returns input tensor."""
    return a


def ones_meta(a):
    """Meta ones_: in-place op returns input tensor."""
    return a


# --- Embedding and other common ops ---

def embedding_meta(indices, weight, **kwargs):
    """Meta embedding: output shape is indices.shape + (embedding_dim,)."""
    embedding_dim = weight.shape[1]
    output_shape = indices.shape + (embedding_dim,)
    return _create_meta_tensor(output_shape, weight.dtype, weight.device)


def layer_norm_meta(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Meta layer_norm: shape stays the same."""
    return _create_meta_tensor(input.shape, input.dtype, input.device)


def dropout_meta(input, p=0.5, training=True, inplace=False, **kwargs):
    """Meta dropout: shape stays the same."""
    return _create_meta_tensor(input.shape, input.dtype, input.device)


def sum_meta(input, dim=None, keepdim=False, dtype=None):
    """Meta sum: compute output shape based on reduction."""
    if dim is None:
        return _create_meta_tensor((), dtype or input.dtype, input.device)
    if isinstance(dim, int):
        dim = (dim,)
    shape = list(input.shape)
    for d in sorted(dim, reverse=True):
        if keepdim:
            shape[d] = 1
        else:
            shape.pop(d)
    return _create_meta_tensor(tuple(shape), dtype or input.dtype, input.device)


def mean_meta(input, dim=None, keepdim=False, dtype=None):
    """Meta mean: compute output shape based on reduction."""
    return sum_meta(input, dim=dim, keepdim=keepdim, dtype=dtype)


def softmax_meta(input, dim=-1, **kwargs):
    """Meta softmax: shape stays the same."""
    return _create_meta_tensor(input.shape, input.dtype, input.device)


def relu_meta(input):
    """Meta relu: shape stays the same."""
    return _create_meta_tensor(input.shape, input.dtype, input.device)


def gelu_meta(input, **kwargs):
    """Meta gelu: shape stays the same."""
    return _create_meta_tensor(input.shape, input.dtype, input.device)


def tanh_meta(input):
    """Meta tanh: shape stays the same."""
    return _create_meta_tensor(input.shape, input.dtype, input.device)


def contiguous_meta(input):
    """Meta contiguous: shape stays the same."""
    return _create_meta_tensor(input.shape, input.dtype, input.device)


def all_meta(input, dim=None, keepdim=False):
    """Meta all: compute output shape based on reduction."""
    from .. import bool as torch_bool
    if dim is None:
        return _create_meta_tensor((), torch_bool, input.device)
    if isinstance(dim, int):
        dim = (dim,)
    shape = list(input.shape)
    for d in sorted(dim, reverse=True):
        if keepdim:
            shape[d] = 1
        else:
            shape.pop(d)
    return _create_meta_tensor(tuple(shape), torch_bool, input.device)


def any_meta(input, dim=None, keepdim=False):
    """Meta any: compute output shape based on reduction."""
    return all_meta(input, dim=dim, keepdim=keepdim)


def _elementwise_meta(a, b=None):
    """Meta elementwise binary op: broadcast shapes."""
    if b is None or not hasattr(b, 'shape'):
        return _create_meta_tensor(a.shape, a.dtype, a.device)
    # Broadcast
    import numpy as np
    out_shape = np.broadcast_shapes(a.shape, b.shape)
    return _create_meta_tensor(out_shape, a.dtype, a.device)


def _comparison_meta(a, b=None):
    """Meta comparison op: broadcast shapes, return bool."""
    from .. import bool as torch_bool
    if b is None or not hasattr(b, 'shape'):
        return _create_meta_tensor(a.shape, torch_bool, a.device)
    import numpy as np
    out_shape = np.broadcast_shapes(a.shape, b.shape)
    return _create_meta_tensor(out_shape, torch_bool, a.device)


def eq_meta(a, b=None):
    return _comparison_meta(a, b)

def ne_meta(a, b=None):
    return _comparison_meta(a, b)

def gt_meta(a, b=None):
    return _comparison_meta(a, b)

def lt_meta(a, b=None):
    return _comparison_meta(a, b)

def ge_meta(a, b=None):
    return _comparison_meta(a, b)

def le_meta(a, b=None):
    return _comparison_meta(a, b)


def pow_meta(a, exponent=None):
    return _elementwise_meta(a, exponent)

def silu_meta(a):
    return _create_meta_tensor(a.shape, a.dtype, a.device)

def sigmoid_meta(a):
    return _create_meta_tensor(a.shape, a.dtype, a.device)

def rsqrt_meta(a):
    return _create_meta_tensor(a.shape, a.dtype, a.device)

def sin_meta(a):
    return _create_meta_tensor(a.shape, a.dtype, a.device)

def cos_meta(a):
    return _create_meta_tensor(a.shape, a.dtype, a.device)

def maximum_meta(a, b):
    return _elementwise_meta(a, b)

def minimum_meta(a, b):
    return _elementwise_meta(a, b)

def cat_meta(tensors, dim=0):
    """Meta cat: concatenate along dim."""
    shapes = [t.shape for t in tensors]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(s[dim] for s in shapes)
    return _create_meta_tensor(tuple(out_shape), tensors[0].dtype, tensors[0].device)

def stack_meta(tensors, dim=0):
    """Meta stack: stack along new dim."""
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    return _create_meta_tensor(tuple(shape), tensors[0].dtype, tensors[0].device)

def unsqueeze_meta(input, dim):
    shape = list(input.shape)
    if dim < 0:
        dim = len(shape) + 1 + dim
    shape.insert(dim, 1)
    return _create_meta_tensor(tuple(shape), input.dtype, input.device)

def squeeze_meta(input, dim=None):
    shape = list(input.shape)
    if dim is not None:
        if isinstance(dim, int):
            dim = [dim]
        for d in sorted(dim, reverse=True):
            if shape[d] == 1:
                shape.pop(d)
    else:
        shape = [s for s in shape if s != 1]
    return _create_meta_tensor(tuple(shape), input.dtype, input.device)

def where_meta(condition, x=None, y=None):
    if x is None:
        return _create_meta_tensor(condition.shape, condition.dtype, condition.device)
    import numpy as np
    out_shape = np.broadcast_shapes(condition.shape, x.shape, y.shape)
    return _create_meta_tensor(out_shape, x.dtype, x.device)

def index_select_meta(input, dim, index):
    shape = list(input.shape)
    shape[dim] = index.shape[0]
    return _create_meta_tensor(tuple(shape), input.dtype, input.device)

def masked_fill_meta(input, mask, value):
    return _create_meta_tensor(input.shape, input.dtype, input.device)

def expand_meta(input, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = sizes[0]
    shape = list(input.shape)
    # Expand -1 means keep original size
    out = []
    for i, s in enumerate(sizes):
        if s == -1:
            out.append(shape[i])
        else:
            out.append(s)
    return _create_meta_tensor(tuple(out), input.dtype, input.device)

def repeat_meta(input, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = sizes[0]
    shape = list(input.shape)
    out = [s * r for s, r in zip(shape, sizes)]
    return _create_meta_tensor(tuple(out), input.dtype, input.device)

def triu_meta(input, diagonal=0):
    return _create_meta_tensor(input.shape, input.dtype, input.device)

def tril_meta(input, diagonal=0):
    return _create_meta_tensor(input.shape, input.dtype, input.device)

def arange_meta(*args, **kwargs):
    """Meta arange - compute length from args."""
    from .. import float32
    if len(args) == 1:
        length = args[0]
    elif len(args) == 2:
        length = args[1] - args[0]
    elif len(args) == 3:
        import math
        length = math.ceil((args[1] - args[0]) / args[2])
    else:
        length = 0
    dtype = kwargs.get('dtype', float32)
    return _create_meta_tensor((int(length),), dtype, 'meta')


# --- Dispatch table ---

META_OPS = {
    'add': add_meta,
    'sub': sub_meta,
    'mul': mul_meta,
    'div': div_meta,
    'matmul': matmul_meta,
    'bmm': bmm_meta,
    'reshape': reshape_meta,
    'transpose': transpose_meta,
    'permute': permute_meta,
    'view': view_meta,
    'neg': neg_meta,
    'exp': exp_meta,
    'log': log_meta,
    'sqrt': sqrt_meta,
    'abs': abs_meta,
    'clone': clone_meta,
    # In-place ops
    'zero_': zero_meta,
    'fill_': fill_meta,
    'ones_': ones_meta,
    # Model execution ops
    'embedding': embedding_meta,
    'layer_norm': layer_norm_meta,
    'dropout': dropout_meta,
    'sum': sum_meta,
    'mean': mean_meta,
    'softmax': softmax_meta,
    'relu': relu_meta,
    'gelu': gelu_meta,
    'tanh': tanh_meta,
    'contiguous': contiguous_meta,
    'all': all_meta,
    'any': any_meta,
    # Comparison ops
    'eq': eq_meta,
    'ne': ne_meta,
    'gt': gt_meta,
    'lt': lt_meta,
    'ge': ge_meta,
    'le': le_meta,
    # More elementwise ops
    'pow': pow_meta,
    'silu': silu_meta,
    'sigmoid': sigmoid_meta,
    'rsqrt': rsqrt_meta,
    'sin': sin_meta,
    'cos': cos_meta,
    'maximum': maximum_meta,
    'minimum': minimum_meta,
    # Shape ops
    'cat': cat_meta,
    'stack': stack_meta,
    'unsqueeze': unsqueeze_meta,
    'squeeze': squeeze_meta,
    'where': where_meta,
    'index_select': index_select_meta,
    'masked_fill': masked_fill_meta,
    'expand': expand_meta,
    'repeat': repeat_meta,
    'triu': triu_meta,
    'tril': tril_meta,
    'arange': arange_meta,
}


def dispatch_meta(op_name: str, *args, **kwargs):
    """Dispatch an operation to its meta implementation.

    Args:
        op_name: Name of the operation
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Meta tensor with correct output shape

    Raises:
        NotImplementedError: If op is not supported for meta tensors
    """
    if op_name not in META_OPS:
        raise NotImplementedError(f"Meta op '{op_name}' not implemented")

    return META_OPS[op_name](*args, **kwargs)
