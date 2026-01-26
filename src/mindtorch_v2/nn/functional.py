"""Functional operations for neural networks."""

from .._dispatch import dispatch
from .._tensor import Tensor


def relu(input, inplace=False):
    """Apply ReLU activation."""
    return dispatch("relu", input)


def gelu(input, approximate='none'):
    """Apply GELU activation."""
    return dispatch("gelu", input, approximate=approximate)


def silu(input, inplace=False):
    """Apply SiLU/Swish activation."""
    return dispatch("silu", input)


def sigmoid(input):
    """Apply sigmoid activation."""
    return dispatch("sigmoid", input)


def tanh(input):
    """Apply tanh activation."""
    return dispatch("tanh", input)


def softmax(input, dim=None, dtype=None):
    """Apply softmax."""
    return dispatch("softmax", input, dim=dim)


def log_softmax(input, dim=None, dtype=None):
    """Apply log softmax."""
    return dispatch("log_softmax", input, dim=dim)


def linear(input, weight, bias=None):
    """Apply linear transformation: y = xW^T + b."""
    output = dispatch("matmul", input, weight.t())
    if bias is not None:
        output = dispatch("add", output, bias)
    return output


def embedding(input, weight, padding_idx=None, max_norm=None,
              norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Look up embeddings."""
    return dispatch("embedding", input, weight)


def dropout(input, p=0.5, training=True, inplace=False):
    """Apply dropout."""
    if not training or p == 0:
        return input
    return dispatch("dropout", input, p=p, training=training)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Apply layer normalization."""
    return dispatch("layer_norm", input, normalized_shape, weight, bias, eps)
