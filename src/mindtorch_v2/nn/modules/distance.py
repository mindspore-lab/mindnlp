from ..module import Module
from .. import functional as F


class CosineSimilarity(Module):
    """Returns cosine similarity between x1 and x2, computed along dim.

    Args:
        dim (int): dimension of calculation. Default: 1.
        eps (float): small value to avoid division by zero. Default: 1e-8.
    """

    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, self.dim, self.eps)

    def extra_repr(self):
        return f'dim={self.dim}, eps={self.eps}'


class PairwiseDistance(Module):
    """Computes the pairwise distance between input vectors, or between columns of input matrices.

    Args:
        p (float): the norm degree. Default: 2.
        eps (float): small value to avoid numerical issues. Default: 1e-6.
        keepdim (bool): whether to keep the last dimension. Default: False.
    """

    def __init__(self, p=2.0, eps=1e-6, keepdim=False):
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1, x2):
        return F.pairwise_distance(x1, x2, self.p, self.eps, self.keepdim)

    def extra_repr(self):
        return f'p={self.p}, eps={self.eps}, keepdim={self.keepdim}'
