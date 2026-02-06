"""Functional API for tensor operations."""

from ._dispatch import dispatch


def add(input, other, *, alpha=1, out=None):
    """Add tensors: input + alpha * other."""
    if alpha != 1:
        other = mul(other, alpha)
    return dispatch("add", input, other)


def sub(input, other, *, alpha=1, out=None):
    """Subtract tensors: input - alpha * other."""
    if alpha != 1:
        other = mul(other, alpha)
    return dispatch("sub", input, other)


def mul(input, other, *, out=None):
    """Multiply tensors element-wise."""
    return dispatch("mul", input, other)


def div(input, other, *, rounding_mode=None, out=None):
    """Divide tensors element-wise."""
    result = dispatch("div", input, other)
    if rounding_mode == "trunc":
        result = trunc(result)
    elif rounding_mode == "floor":
        result = floor(result)
    return result


def neg(input, *, out=None):
    """Negate tensor element-wise."""
    return dispatch("neg", input)


def abs(input, *, out=None):
    """Absolute value element-wise."""
    return dispatch("abs", input)


def pow(input, exponent, *, out=None):
    """Power element-wise."""
    return dispatch("pow", input, exponent)


def exp(input, *, out=None):
    """Exponential element-wise."""
    return dispatch("exp", input)


def log(input, *, out=None):
    """Natural logarithm element-wise."""
    return dispatch("log", input)


def sqrt(input, *, out=None):
    """Square root element-wise."""
    return dispatch("sqrt", input)


def square(input, *, out=None):
    """Square each element of the tensor."""
    return mul(input, input)


def sin(input, *, out=None):
    """Sine element-wise."""
    return dispatch("sin", input)


def cos(input, *, out=None):
    """Cosine element-wise."""
    return dispatch("cos", input)


def tanh(input, *, out=None):
    """Hyperbolic tangent element-wise."""
    return dispatch("tanh", input)


def matmul(input, other, *, out=None):
    """Matrix multiplication."""
    return dispatch("matmul", input, other)


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """Performs a matrix multiplication of mat1 and mat2, adds input, and returns the result.

    out = beta * input + alpha * (mat1 @ mat2)
    """
    result = matmul(mat1, mat2)
    if alpha != 1:
        result = mul(result, alpha)
    if beta != 1:
        input = mul(input, beta)
    return add(input, result)


# --- Reduction ops ---

def sum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    """Sum of tensor elements."""
    return dispatch("sum", input, dim=dim, keepdim=keepdim)


def mean(input, dim=None, keepdim=False, *, dtype=None, out=None, axis=None):
    """Mean of tensor elements. Accepts both dim and axis for numpy compatibility."""
    if axis is not None and dim is None:
        dim = axis
    return dispatch("mean", input, dim=dim, keepdim=keepdim)


def max(input, dim=None, keepdim=False, *, out=None):
    """Max of tensor elements."""
    return dispatch("max", input, dim=dim, keepdim=keepdim)


def min(input, dim=None, keepdim=False, *, out=None):
    """Min of tensor elements."""
    return dispatch("min", input, dim=dim, keepdim=keepdim)


def prod(input, dim=None, keepdim=False, *, dtype=None, out=None):
    """Product of tensor elements."""
    return dispatch("prod", input, dim=dim, keepdim=keepdim)


def argmax(input, dim=None, keepdim=False):
    """Index of max element."""
    return dispatch("argmax", input, dim=dim, keepdim=keepdim)


def argmin(input, dim=None, keepdim=False):
    """Index of min element."""
    return dispatch("argmin", input, dim=dim, keepdim=keepdim)


# --- Comparison ops ---

def eq(input, other, *, out=None):
    """Element-wise equality comparison."""
    return dispatch("eq", input, other)


def ne(input, other, *, out=None):
    """Element-wise not equal comparison."""
    return dispatch("ne", input, other)


def gt(input, other, *, out=None):
    """Element-wise greater than comparison."""
    return dispatch("gt", input, other)


def lt(input, other, *, out=None):
    """Element-wise less than comparison."""
    return dispatch("lt", input, other)


def ge(input, other, *, out=None):
    """Element-wise greater than or equal comparison."""
    return dispatch("ge", input, other)


def le(input, other, *, out=None):
    """Element-wise less than or equal comparison."""
    return dispatch("le", input, other)


# --- Tensor manipulation ops ---

def cat(tensors, dim=0, *, out=None, axis=None):
    """Concatenate tensors along a dimension."""
    # Support axis as alias for dim (numpy compatibility)
    if axis is not None and dim == 0:
        dim = axis
    return dispatch("cat", tensors, dim=dim)


def stack(tensors, dim=0, *, out=None):
    """Stack tensors along a new dimension."""
    return dispatch("stack", tensors, dim=dim)


def split(tensor, split_size_or_sections, dim=0):
    """Split tensor into chunks."""
    return dispatch("split", tensor, split_size_or_sections, dim=dim)


def chunk(input, chunks, dim=0):
    """Split tensor into specified number of chunks."""
    return dispatch("chunk", input, chunks, dim=dim)


def clone(input, *, memory_format=None):
    """Create a copy of tensor with new storage."""
    return dispatch("clone", input)


def where(condition, input, other):
    """Select elements based on condition."""
    return dispatch("where", condition, input, other)


# --- Additional math ops ---

def var(input, dim=None, *, correction=1, keepdim=False, out=None, axis=None):
    """Variance of tensor elements. Accepts both dim and axis for numpy compatibility."""
    if axis is not None and dim is None:
        dim = axis
    return dispatch("var", input, dim=dim, correction=correction, keepdim=keepdim)


def std(input, dim=None, *, correction=1, keepdim=False, out=None, axis=None):
    """Standard deviation of tensor elements. Accepts both dim and axis for numpy compatibility."""
    if axis is not None and dim is None:
        dim = axis
    return dispatch("std", input, dim=dim, correction=correction, keepdim=keepdim)


def clamp(input, min=None, max=None, *, out=None):
    """Clamp values to range [min, max]."""
    return dispatch("clamp", input, min=min, max=max)


def rsqrt(input, *, out=None):
    """Reciprocal square root element-wise."""
    return dispatch("rsqrt", input)


def reciprocal(input, *, out=None):
    """Reciprocal (1/x) element-wise."""
    return dispatch("reciprocal", input)


def bmm(input, mat2, *, out=None):
    """Batched matrix multiplication."""
    return dispatch("bmm", input, mat2)


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    """Batched matrix multiply with add: beta*input + alpha*(batch1 @ batch2)."""
    return dispatch("baddbmm", input, batch1, batch2, beta=beta, alpha=alpha)


def all(input, dim=None, keepdim=False, *, out=None, axis=None, keepdims=None):
    """Test if all elements evaluate to True."""
    # Support axis as alias for dim (numpy compatibility)
    if axis is not None and dim is None:
        dim = axis
    # Support keepdims as alias for keepdim (numpy compatibility)
    if keepdims is not None:
        keepdim = keepdims
    return dispatch("all", input, dim=dim, keepdim=keepdim)


def any(input, dim=None, keepdim=False, *, out=None, axis=None, keepdims=None):
    """Test if any element evaluates to True."""
    # Support axis as alias for dim (numpy compatibility)
    if axis is not None and dim is None:
        dim = axis
    # Support keepdims as alias for keepdim (numpy compatibility)
    if keepdims is not None:
        keepdim = keepdims
    return dispatch("any", input, dim=dim, keepdim=keepdim)


def isin(elements, test_elements, *, assume_unique=False, invert=False):
    """Test if each element of elements is in test_elements."""
    return dispatch("isin", elements, test_elements, assume_unique=assume_unique, invert=invert)


def topk(input, k, dim=-1, largest=True, sorted=True, *, out=None):
    """Return the k largest/smallest elements along a dimension."""
    return dispatch("topk", input, k, dim=dim, largest=largest, sorted=sorted)


def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    """Draw samples from a multinomial distribution."""
    return dispatch("multinomial", input, num_samples, replacement=replacement, generator=generator)


def cumsum(input, dim, *, dtype=None, out=None):
    """Cumulative sum of elements along a dimension."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    result = np.cumsum(arr, axis=dim)
    if dtype is not None:
        result = result.astype(dtype)
    return Tensor(result)


def cumprod(input, dim, *, dtype=None, out=None):
    """Cumulative product of elements along a dimension."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    result = np.cumprod(arr, axis=dim)
    if dtype is not None:
        result = result.astype(dtype)
    return Tensor(result)


def floor(input, *, out=None):
    """Floor element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.floor(input.numpy())
    return Tensor(result.astype(input.numpy().dtype))


def trunc(input, *, out=None):
    """Truncate element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.trunc(input.numpy())
    return Tensor(result.astype(input.numpy().dtype))


def ceil(input, *, out=None):
    """Ceiling element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.ceil(input.numpy())
    return Tensor(result.astype(input.numpy().dtype))


def round(input, *, decimals=0, out=None):
    """Round element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.round(input.numpy(), decimals=decimals)
    return Tensor(result.astype(input.numpy().dtype))


def sign(input, *, out=None):
    """Sign of elements (-1, 0, or 1)."""
    import numpy as np
    from ._tensor import Tensor
    result = np.sign(input.numpy())
    return Tensor(result.astype(input.numpy().dtype))


def fmod(input, other, *, out=None):
    """Element-wise remainder of division."""
    import numpy as np
    from ._tensor import Tensor
    if hasattr(other, 'numpy'):
        other = other.numpy()
    result = np.fmod(input.numpy(), other)
    return Tensor(result.astype(input.numpy().dtype))


def remainder(input, other, *, out=None):
    """Computes Python's modulus operation."""
    import numpy as np
    from ._tensor import Tensor
    if hasattr(other, 'numpy'):
        other = other.numpy()
    result = np.remainder(input.numpy(), other)
    return Tensor(result.astype(input.numpy().dtype))


def log10(input, *, out=None):
    """Base-10 logarithm element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.log10(input.numpy())
    return Tensor(result.astype(np.float32))


def log2(input, *, out=None):
    """Base-2 logarithm element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.log2(input.numpy())
    return Tensor(result.astype(np.float32))


def log1p(input, *, out=None):
    """Natural log of (1 + x) element-wise."""
    from ._dispatch import dispatch
    return dispatch("log1p", input)


def expm1(input, *, out=None):
    """Exp(x) - 1 element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.expm1(input.numpy())
    return Tensor(result.astype(np.float32))


def acos(input, *, out=None):
    """Arc cosine element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.arccos(input.numpy())
    return Tensor(result.astype(np.float32))


def asin(input, *, out=None):
    """Arc sine element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.arcsin(input.numpy())
    return Tensor(result.astype(np.float32))


def atan(input, *, out=None):
    """Arc tangent element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.arctan(input.numpy())
    return Tensor(result.astype(np.float32))


def atan2(input, other, *, out=None):
    """Arc tangent of input/other element-wise."""
    import numpy as np
    from ._tensor import Tensor
    if hasattr(other, 'numpy'):
        other = other.numpy()
    result = np.arctan2(input.numpy(), other)
    return Tensor(result.astype(np.float32))


def cosh(input, *, out=None):
    """Hyperbolic cosine element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.cosh(input.numpy())
    return Tensor(result.astype(np.float32))


def sinh(input, *, out=None):
    """Hyperbolic sine element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.sinh(input.numpy())
    return Tensor(result.astype(np.float32))


def acosh(input, *, out=None):
    """Inverse hyperbolic cosine element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.arccosh(input.numpy())
    return Tensor(result.astype(np.float32))


def asinh(input, *, out=None):
    """Inverse hyperbolic sine element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.arcsinh(input.numpy())
    return Tensor(result.astype(np.float32))


def atanh(input, *, out=None):
    """Inverse hyperbolic tangent element-wise."""
    import numpy as np
    from ._tensor import Tensor
    result = np.arctanh(input.numpy())
    return Tensor(result.astype(np.float32))


# Activation functions at module level
def relu(input, inplace=False):
    """Applies ReLU element-wise."""
    return dispatch("relu", input)


def sigmoid(input):
    """Applies sigmoid element-wise."""
    return dispatch("sigmoid", input)


# Tensor manipulation ops
def squeeze(input, dim=None):
    """Remove dimensions of size 1."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    if dim is None:
        result = np.squeeze(arr)
    else:
        result = np.squeeze(arr, axis=dim)
    return Tensor(result)


def unsqueeze(input, dim):
    """Add a dimension of size 1 at the specified position."""
    import numpy as np
    from ._tensor import Tensor
    result = np.expand_dims(input.numpy(), axis=dim)
    return Tensor(result)


def flip(input, dims):
    """Reverse the order of elements along given dimensions."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    for d in dims:
        arr = np.flip(arr, axis=d)
    return Tensor(arr.copy())


def roll(input, shifts, dims=None):
    """Roll the tensor along the given dimensions."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    if dims is None:
        result = np.roll(arr, shifts)
    elif isinstance(shifts, (list, tuple)):
        result = arr
        for s, d in zip(shifts, dims):
            result = np.roll(result, s, axis=d)
    else:
        result = np.roll(arr, shifts, axis=dims)
    return Tensor(result)


def gather(input, dim, index):
    """Gather values along an axis specified by dim."""
    import numpy as np
    from ._tensor import Tensor
    src = input.numpy()
    idx = index.numpy().astype(np.int64)  # Ensure indices are integers
    out = np.take_along_axis(src, idx, axis=dim)
    return Tensor(out)


def index_select(input, dim, index):
    """Select values along given dimension using indices."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    idx = index.numpy()
    result = np.take(arr, idx, axis=dim)
    return Tensor(result)


def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    """Repeat elements of a tensor."""
    import numpy as np
    from ._tensor import Tensor
    if isinstance(input, Tensor):
        arr = input.numpy()
    else:
        arr = np.asarray(input)
    if hasattr(repeats, 'numpy'):
        repeats = repeats.numpy()
    result = np.repeat(arr, repeats, axis=dim)
    return Tensor(result)


def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    """Eliminates all but the first element from consecutive groups of equivalent elements."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    if dim is not None:
        raise NotImplementedError("unique_consecutive with dim is not supported")
    flat = arr.flatten()
    mask = np.concatenate([[True], flat[1:] != flat[:-1]])
    output = flat[mask]
    ret = (Tensor(output),)
    if return_inverse:
        inverse = np.zeros(len(flat), dtype=np.int64)
        idx = 0
        for i in range(len(flat)):
            if mask[i]:
                idx = np.sum(mask[:i+1]) - 1
            inverse[i] = idx
        ret = ret + (Tensor(inverse),)
    if return_counts:
        indices = np.where(mask)[0]
        counts = np.diff(np.append(indices, len(flat)))
        ret = ret + (Tensor(counts),)
    if len(ret) == 1:
        return ret[0]
    return ret


def einsum(equation, *operands):
    """Evaluate Einstein summation convention on operands."""
    import numpy as np
    from ._tensor import Tensor
    np_ops = [op.numpy() if hasattr(op, 'numpy') else np.asarray(op) for op in operands]
    result = np.einsum(equation, *np_ops)
    return Tensor(result)


def sort(input, dim=-1, descending=False, stable=False):
    """Sort the elements of the input tensor along a given dimension."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    if descending:
        indices = np.argsort(-arr, axis=dim, kind='stable' if stable else 'quicksort')
    else:
        indices = np.argsort(arr, axis=dim, kind='stable' if stable else 'quicksort')
    values = np.take_along_axis(arr, indices, axis=dim)
    from collections import namedtuple
    SortResult = namedtuple('SortResult', ['values', 'indices'])
    return SortResult(Tensor(values), Tensor(indices))


def reshape(input, shape):
    """Reshape a tensor."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(input.numpy().reshape(shape))


def permute(input, dims):
    """Permute the dimensions of the input tensor."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(np.transpose(input.numpy(), dims))


def transpose(input, dim0, dim1):
    """Transpose two dimensions of a tensor."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    axes = list(range(arr.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Tensor(np.transpose(arr, axes))


def narrow(input, dim, start, length):
    """Return a narrowed version of input tensor."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    slices = [slice(None)] * arr.ndim
    slices[dim] = slice(start, start + length)
    return Tensor(arr[tuple(slices)])


def masked_fill(input, mask, value):
    """Fill elements of input where mask is True with value."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy().copy()
    m = mask.numpy() if hasattr(mask, 'numpy') else np.asarray(mask)
    arr[m] = value
    return Tensor(arr)


def norm(input, p=2, dim=None, keepdim=False):
    """Return the matrix norm or vector norm of a given tensor."""
    import numpy as np
    from ._tensor import Tensor
    arr = input.numpy()
    if dim is None:
        result = np.linalg.norm(arr, ord=p)
        if keepdim:
            result = np.array(result).reshape([1] * arr.ndim)
    else:
        result = np.linalg.norm(arr, ord=p, axis=dim, keepdims=keepdim)
    return Tensor(result.astype(np.float32))


def isnan(input):
    """Return a boolean tensor where True means NaN."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(np.isnan(input.numpy()))


def isinf(input):
    """Return a boolean tensor where True means inf/-inf."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(np.isinf(input.numpy()))


def isfinite(input):
    """Return a boolean tensor where True means finite."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(np.isfinite(input.numpy()))


def logical_not(input):
    """Compute element-wise logical NOT."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(np.logical_not(input.numpy()))


def logical_and(input, other):
    """Compute element-wise logical AND."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(np.logical_and(input.numpy(), other.numpy() if hasattr(other, 'numpy') else other))


def logical_or(input, other):
    """Compute element-wise logical OR."""
    import numpy as np
    from ._tensor import Tensor
    return Tensor(np.logical_or(input.numpy(), other.numpy() if hasattr(other, 'numpy') else other))


def equal(input, other):
    """Check if two tensors are equal (same size and elements).

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        True if the two tensors have the same size and elements, False otherwise.
    """
    import numpy as np
    a = input.numpy()
    b = other.numpy() if hasattr(other, 'numpy') else np.asarray(other)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


def svd_lowrank(A, q=6, niter=2, M=None):
    """Compute low-rank SVD approximation of a matrix.

    Args:
        A: Input tensor of shape (*, m, n)
        q: Rank of the approximation
        niter: Number of subspace iterations
        M: Optional matrix for centering

    Returns:
        U, S, V: Low-rank SVD components
    """
    import numpy as np
    from ._tensor import Tensor

    a = A.numpy()
    if M is not None:
        a = a - M.numpy()

    # Use full SVD and truncate
    U, S, Vt = np.linalg.svd(a, full_matrices=False)

    # Truncate to rank q
    U = U[..., :q]
    S = S[..., :q]
    V = Vt[..., :q, :].swapaxes(-2, -1)

    return Tensor(U), Tensor(S), Tensor(V)


def diff(input, n=1, dim=-1, prepend=None, append=None):
    """Computes the n-th forward difference along the given dimension.

    Args:
        input: Input tensor
        n: Number of times to recursively compute the difference
        dim: Dimension to compute the difference along
        prepend: Values to prepend before computing the difference
        append: Values to append before computing the difference

    Returns:
        Tensor of differences
    """
    import numpy as np
    from ._tensor import Tensor

    arr = input.numpy()

    if prepend is not None:
        prepend_arr = prepend.numpy() if hasattr(prepend, 'numpy') else np.asarray(prepend)
        arr = np.concatenate([prepend_arr, arr], axis=dim)
    if append is not None:
        append_arr = append.numpy() if hasattr(append, 'numpy') else np.asarray(append)
        arr = np.concatenate([arr, append_arr], axis=dim)

    result = np.diff(arr, n=n, axis=dim)
    return Tensor(result)


def maximum(input, other, *, out=None):
    """Compute element-wise maximum of input and other.

    Args:
        input: First tensor
        other: Second tensor or scalar
        out: Optional output tensor (ignored)

    Returns:
        Element-wise maximum
    """
    from ._dispatch import dispatch
    return dispatch("maximum", input, other)


def minimum(input, other, *, out=None):
    """Compute element-wise minimum of input and other.

    Args:
        input: First tensor
        other: Second tensor or scalar
        out: Optional output tensor (ignored)

    Returns:
        Element-wise minimum
    """
    from ._dispatch import dispatch
    return dispatch("minimum", input, other)


def take_along_dim(input, indices, dim):
    """Select values from input at positions given by indices along dim.

    Args:
        input: Input tensor
        indices: Index tensor (same dims as input, or broadcastable)
        dim: Dimension to index along

    Returns:
        Tensor with values gathered from input
    """
    import numpy as np
    from ._tensor import Tensor

    arr = input.numpy()
    idx = indices.numpy() if hasattr(indices, 'numpy') else np.asarray(indices)
    result = np.take_along_axis(arr, idx, axis=dim)
    return Tensor(result)


def conj(input):
    """Return the complex conjugate of input tensor.

    Args:
        input: Input tensor (can be complex or real)

    Returns:
        Complex conjugate of input
    """
    from ._dispatch import dispatch
    return dispatch("conj", input)


def conj_physical(input):
    """Return the complex conjugate of input tensor (physical conjugate).

    Args:
        input: Input tensor

    Returns:
        Complex conjugate of input
    """
    return conj(input)


def bucketize(input, boundaries, *, out_int32=False, right=False, out=None):
    """Return the indices of the buckets to which each value in input belongs.

    Args:
        input: Input tensor of values to bucketize
        boundaries: 1-D tensor of sorted boundaries
        out_int32: If True, return indices as int32 instead of int64
        right: If False, return the bucket i where boundaries[i-1] < x <= boundaries[i].
               If True, return bucket i where boundaries[i-1] <= x < boundaries[i].
        out: Optional output tensor

    Returns:
        Tensor of bucket indices
    """
    import numpy as np
    from ._tensor import Tensor

    arr = input.numpy()
    bounds = boundaries.numpy() if hasattr(boundaries, 'numpy') else np.asarray(boundaries)

    # numpy.searchsorted is similar to bucketize but with opposite 'side' semantics
    # when right=False in torch, we want side='right' in numpy (find first bound > x)
    # when right=True in torch, we want side='left' in numpy (find first bound >= x)
    side = 'left' if right else 'right'
    result = np.searchsorted(bounds, arr, side=side)

    dtype = np.int32 if out_int32 else np.int64
    return Tensor(result.astype(dtype))


def histc(input, bins=100, min=0, max=0, *, out=None):
    """Compute the histogram of a tensor.

    Args:
        input: Input tensor
        bins: Number of histogram bins
        min: Lower bound of the histogram range
        max: Upper bound of the histogram range
        out: Optional output tensor

    Returns:
        Tensor containing the histogram
    """
    import numpy as np
    from ._tensor import Tensor

    arr = input.numpy().flatten()

    # Handle default min/max (0, 0 means use data range)
    if min == 0 and max == 0:
        min_val = float(arr.min())
        max_val = float(arr.max())
    else:
        min_val = float(min)
        max_val = float(max)

    # Compute histogram
    hist, _ = np.histogram(arr, bins=bins, range=(min_val, max_val))
    return Tensor(hist.astype(np.float32))


def nonzero(input, *, as_tuple=False):
    """Return indices of non-zero elements.

    Args:
        input: Input tensor
        as_tuple: If False (default), returns 2-D tensor of shape (N, ndim).
                  If True, returns tuple of 1-D tensors.

    Returns:
        Tensor or tuple of tensors containing indices of non-zero elements
    """
    import numpy as np
    from ._tensor import Tensor

    arr = input.numpy()
    indices = np.nonzero(arr)

    if as_tuple:
        return tuple(Tensor(idx) for idx in indices)
    else:
        # Stack indices into (N, ndim) array
        if len(indices[0]) == 0:
            # No non-zero elements
            return Tensor(np.empty((0, arr.ndim), dtype=np.int64))
        stacked = np.stack(indices, axis=1)
        return Tensor(stacked)


def argsort(input, dim=-1, descending=False, stable=False):
    """Return the indices that would sort a tensor.

    Args:
        input: Input tensor
        dim: Dimension to sort along (default: -1, last dim)
        descending: If True, sort in descending order
        stable: If True, use stable sorting algorithm

    Returns:
        Tensor of indices that would sort the input tensor
    """
    import numpy as np
    from ._tensor import Tensor

    arr = input.numpy()
    if descending:
        indices = np.argsort(-arr, axis=dim, kind='stable' if stable else 'quicksort')
    else:
        indices = np.argsort(arr, axis=dim, kind='stable' if stable else 'quicksort')
    return Tensor(indices)
