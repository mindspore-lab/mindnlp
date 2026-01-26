"""Sparse modules like Embedding."""

import numpy as np
from ..module import Module
from ..parameter import Parameter
from .. import functional as F
import mindtorch_v2 as torch


class Embedding(Module):
    """A lookup table that stores embeddings of a fixed dictionary and size.

    Args:
        num_embeddings: size of the dictionary
        embedding_dim: size of each embedding vector
        padding_idx: If specified, entries at padding_idx do not contribute to gradient
        max_norm: If given, embeddings are renormalized to have max_norm
        norm_type: The p of the p-norm for max_norm
        scale_grad_by_freq: If True, scale gradients by frequency
        sparse: If True, gradient is sparse
    """

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    max_norm: float
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None,
                 max_norm: float = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, sparse: bool = False,
                 _weight: torch.Tensor = None, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        if _weight is None:
            # Initialize with normal distribution
            weight_np = np.random.normal(0, 1, (num_embeddings, embedding_dim)).astype(np.float32)
            self.weight = Parameter(torch.tensor(weight_np))
        else:
            self.weight = Parameter(_weight)

        if padding_idx is not None:
            # Zero out padding embedding
            # Note: this is a simplified version - full implementation would need no_grad
            pass

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                          self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        if self.max_norm is not None:
            s += f', max_norm={self.max_norm}'
        if self.norm_type != 2.0:
            s += f', norm_type={self.norm_type}'
        if self.scale_grad_by_freq:
            s += ', scale_grad_by_freq=True'
        if self.sparse:
            s += ', sparse=True'
        return s

    def __repr__(self):
        return f'Embedding({self.extra_repr()})'
