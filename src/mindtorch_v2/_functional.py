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
