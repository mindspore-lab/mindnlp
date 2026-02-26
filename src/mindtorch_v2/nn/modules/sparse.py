from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor
from .. import functional as F


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 _freeze=False, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if _weight is not None:
            self.weight = Parameter(_weight) if not isinstance(_weight, Parameter) else _weight
        else:
            self.weight = Parameter(tensor([[0.0] * embedding_dim for _ in range(num_embeddings)]))
        if _freeze:
            self.weight.requires_grad = False

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None, max_norm=None,
                        norm_type=2.0, scale_grad_by_freq=False, sparse=False):
        rows, cols = embeddings.shape
        return cls(rows, cols, padding_idx=padding_idx, max_norm=max_norm,
                   norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse, _weight=embeddings, _freeze=freeze)

    def extra_repr(self):
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        return s


class EmbeddingBag(Module):
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None,
                 include_last_offset=False, padding_idx=None, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx
        if _weight is not None:
            self.weight = Parameter(_weight)
        else:
            self.weight = Parameter(tensor([[0.0] * embedding_dim for _ in range(num_embeddings)]))

    def forward(self, input, offsets=None, per_sample_weights=None):
        raise NotImplementedError("EmbeddingBag forward is not yet implemented")

    def extra_repr(self):
        return f'{self.num_embeddings}, {self.embedding_dim}, mode={repr(self.mode)}'
