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


# --- Reduction ops ---

def sum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    """Sum of tensor elements."""
    return dispatch("sum", input, dim=dim, keepdim=keepdim)


def mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    """Mean of tensor elements."""
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

def cat(tensors, dim=0, *, out=None):
    """Concatenate tensors along a dimension."""
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

def var(input, dim=None, *, correction=1, keepdim=False, out=None):
    """Variance of tensor elements."""
    return dispatch("var", input, dim=dim, correction=correction, keepdim=keepdim)


def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    """Standard deviation of tensor elements."""
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


def all(input, dim=None, keepdim=False, *, out=None):
    """Test if all elements evaluate to True."""
    return dispatch("all", input, dim=dim, keepdim=keepdim)


def any(input, dim=None, keepdim=False, *, out=None):
    """Test if any element evaluates to True."""
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
