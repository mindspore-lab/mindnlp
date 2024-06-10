

def _no_grad_embedding_renorm_(weight: Tensor, input: Tensor, max_norm: float, norm_type: float) -> Tuple[Tensor, Tensor]:
    torch.embedding_renorm_(weight.detach(), input, max_norm, norm_type)


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    r"""Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    .. note::
        Note that the analytical gradients of this function with respect to
        entries in :attr:`weight` at the row specified by :attr:`padding_idx`
        are expected to differ from the numerical ones.

    .. note::
        Note that `:class:`torch.nn.Embedding` differs from this function in
        that it initializes the row of :attr:`weight` specified by
        :attr:`padding_idx` to all zeros on construction.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad".
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
          where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = torch.tensor([[0, 2, 0, 5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    """
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)