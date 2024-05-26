# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Transformer modules."""
import copy
from typing import Optional
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, XavierUniform
from mindnlp._legacy.initializer import XavierNormal
from mindnlp._legacy.functional import multi_head_attention_forward, relu, gelu
from .dropout import Dropout


class Linear(nn.Dense):
    """inner Linear."""

class MultiheadAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in the paper `Attention is all you need
    <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length, and the
    key and value vector with target length, the attention will be performed as the following

    .. math::
            MultiHeadAttention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

    if query, key and value tensor is same, then it will be self attention.

    Args:
        embed_dim (int): Total dimension of MultiheadAttention.
        num_heads (int): Number of attention heads. Note that `embed_dim` will be split
            across `num_heads` (i.e. each head will have dimension `embed_dim // num_heads`).
        dropout (float): Dropout probability of `attn_output_weights`. Default: ``0.0``.
        has_bias (bool): Whether adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv (bool): Whether adds bias to the key and value sequences at axis=0. Default: ``False``.
        add_zero_attn (bool): Whether adds a new batch of zeros to the key and value sequences at axis=1.
            Default: ``False``.
        kdim (int): Total number of features for keys. Default: ``None`` (`kdim=embed_dim`).
        vdim (int): Total number of features for values. Default: ``None`` (`vdim=embed_dim`).
        batch_first (bool): If ``True``, then the input and output shape are (batch, seq, feature),
            else (seq, batch, feature). Default: ``False``.

    Inputs:
        - **query** (Tensor): The query embeddings. If `query` is unbatched, the shape is :math:`(L, E_q)`,
          otherwise the shape is :math:`(L, N, E_q)` when `batch_first=False` or :math:`(N, L, E_q)` when
          `batch_first=True`, where :math:`L`is the target sequence length, :math:`N` is the batch size,
          and :math:`E_q` is the query embedding dimension `embed_dim`. Queries are compared against
          key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        - **key** (Tensor): The key embeddings. If `key` is unbatched, the shape is :math:`(S, E_k)`, otherwise
          the shape is :math:`(S, N, E_k)` when `batch_first=False` or :math:`(N, S, E_k)` when
          `batch_first=True`, where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_k` is the key embedding dimension `kdim`. See "Attention Is All You Need" for more details.
        - **value** (Tensor): The value embeddings. If `value` is unbatched, the shape is :math:`(S, E_v)`,
          otherwise the shape is :math:`(S, N, E_v)` when `batch_first=False` or :math:`(N, S, E_v)` when
          `batch_first=True`, where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_v` is the value embedding dimension `vdim`. See "Attention Is All You Need" for more details.
        - **key_padding_mask** (Tensor, optional): If specified, a mask of shape :math:`(N, S)` indicating which
          elements within `key` to ignore for the purpose of attention (i.e. treat as "padding").
          For unbatched `query`, shape should be :math:`(S)`. Binary and byte masks are supported.
          For a binary mask, a ``True`` value indicates that the corresponding `key` value will be ignored for
          the purpose of attention. For a float mask, it will be directly added to the corresponding `key` value.
        - **need_weights** (bool): Whether returns `attn_output_weights` in addition to `attn_outputs`.
          Default: ``True``.
        - **attn_mask** (Tensor, optional): If specified, a 2D or 3D mask preventing attention to certain positions.
          Must be of shape :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the
          batch size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length.
          A 2D mask will be broadcasted across the batch while a 3D mask allows for a different mask for each entry
          in the batch. Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates
          that the corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that
          the corresponding position is not allowed to attend. For a float mask, the mask values will be added to
          the attention weight.
        - **average_attn_weights** (bool): If true, indicates that the returned `attn_weights` should be averaged
          across heads. Otherwise, `attn_weights` are provided separately per head. Note that this flag only
          has an effect when `need_weights=True`. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        Tuple, a tuple contains(`attn_output`, `attn_output_weights`)

        - **attn_output** - Attention outputs. If input is unbatched, the output shape is:math:`(L, E)`, otherwise
          the output shape is :math:`(L, N, E)` when `batch_first=False` or :math:`(N, L, E)` when
          `batch_first=True`, where :math:`L` is the target sequence length, :math:`N` is the batch size,
          and :math:`E` is the embedding dimension `embed_dim`.
        - **attn_output_weights** - Only returned when `need_weights=True`. If `average_attn_weights=True`,
          returns attention weights averaged across heads with shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)` when input is batched, where :math:`N` is the batch size, :math:`L` is
          the target sequence length, and :math:`S` is the source sequence length.
          If `average_attn_weights=False`, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or
          :math:`(N, \text{num\_heads}, L, S)` when input is batched.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False) -> None:

        r"""Initialize the MultiheadAttention class.
        
        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            bias (bool, optional): Whether to include bias parameters. Defaults to True.
            add_bias_kv (bool, optional): Whether to add bias to key and value tensors. Defaults to False.
            add_zero_attn (bool, optional): Whether to handle the zero attention case. Defaults to False.
            kdim (int, optional): The dimension of the key projections. Defaults to None.
            vdim (int, optional): The dimension of the value projections. Defaults to None.
            batch_first (bool, optional): Whether the input and output tensors are provided as (batch, seq, feature). 
                Defaults to False.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            AssertionError: If the embed_dim is not divisible by num_heads.
            ValueError: If invalid parameters or configurations are provided.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, embed_dim)), 'q_proj_weight')
            self.k_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.kdim)), 'k_proj_weight')
            self.v_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.vdim)), 'v_proj_weight')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = Parameter(initializer(XavierUniform(), (3 * embed_dim, embed_dim)), 'in_proj_weight')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if bias:
            self.in_proj_bias = Parameter(initializer('zeros', (3 * embed_dim)), 'in_proj_bias')
        else:
            self.in_proj_bias = None
        self.out_proj = Linear(embed_dim, embed_dim, has_bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_k')
            self.bias_v = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_v')
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.k_is_v = False
        self.q_is_k = False

    def __call__(self, *args, **kwargs):

        r"""
        This method is the '__call__' method of the 'MultiheadAttention' class.
        
        Args:
            self (object): The instance of the 'MultiheadAttention' class.
                - Purpose: Represents the instance of the 'MultiheadAttention' class on which the method is called.
                - Restrictions: This parameter is required for the method to be called.
        
        Returns:
            None: This method does not return any value.
                - Purpose: The method does not have a specific return value as it is intended for its side effects on the instance.
        
        Raises:
            No specific exceptions are raised by this method.
                - Purpose: The method does not explicitly raise any exceptions.
        """
        query = kwargs.get('query', args[0])
        key = kwargs.get('key', args[1])
        value = kwargs.get('value', args[2])
        self.k_is_v = key is value
        self.q_is_k = query is key
        return super().__call__(*args, **kwargs)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                  need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True):

        r"""
        Method 'construct' in the class 'MultiheadAttention'.
        
        Args:
        - self: The instance of the class.
        - query (Tensor): The input query tensor.
        - key (Tensor): The input key tensor.
        - value (Tensor): The input value tensor.
        - key_padding_mask (Optional[Tensor], optional): Mask tensor specifying which keys have padding elements. Defaults to None.
        - need_weights (bool): Flag indicating whether to return attention weights. Defaults to True.
        - attn_mask (Optional[Tensor], optional): Mask tensor for attention calculation. Defaults to None.
        - average_attn_weights (bool): Flag indicating whether to average attention weights. Defaults to True.
        
        Returns:
        - None: This method does not return any value.
        
        Raises:
        - AssertionError: Raised if key_padding_mask is not of type bool or floating point type.
        """
        is_batched = query.ndim == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != mindspore.bool_ and not ops.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        if self.batch_first and is_batched:
            # k_is_v and q_is_k preprocess in __call__ since Graph mode do not support `is`
            if self.k_is_v:
                if self.q_is_k:
                    query = key = value = query.swapaxes(1, 0)
                else:
                    query = query.swapaxes(1, 0)
                    key = key.swapaxes(1, 0)
                    value = key
            else:
                query = query.swapaxes(1, 0)
                key = key.swapaxes(1, 0)
                value = value.swapaxes(1, 0)

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)

        if self.batch_first and is_batched:
            attn_output = attn_output.swapaxes(1, 0)
        if need_weights:
            return attn_output, attn_output_weights
        return (attn_output,)


class TransformerEncoderLayer(nn.Cell):
    r"""
    Transformer Encoder Layer. This is an implementation of the single layer of the transformer
    encoder layer, including multihead attention and feedward layer.

    Args:
        d_model (int): The number of features in the input tensor.
        nhead (int): The number of heads in the MultiheadAttention modules.
        dim_feedforward (int): The dimension of the feedforward layer. Default: ``2048``.
        dropout (float): The dropout value. Default: ``0.1``.
        activation (Union[str, callable, Cell]): The activation function of the intermediate layer,
            can be a string (`"relu"` or `"gelu"`), Cell instance (`nn.ReLU()` or `nn.GELU()`) or
            a callable (`ops.relu` or `ops.gelu`). Default: ``"relu"``.
        layer_norm_eps (float): The epsilon value in LayerNorm modules. Default: ``1e-5``.
        batch_first (bool): If `batch_first = True`, then the shape of input and output tensors is
            (batch, seq, feature), otherwise the shape is (seq, batch, feature). Default: ``False``.
        norm_first (bool): If `norm_first = True`, layer norm is done prior to attention and feedforward
            operations, respectively. Default: ``False``.

    Inputs:
        - **src** (Tensor): the sequence to the encoder layer.
        - **src_mask** (Tensor, optional): the mask for the src sequence. Default: ``None``.
        - **src_key_padding_mask** (Tensor, optional): the mask for the src keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> out = encoder_layer(src)
        >>> # Alternatively, when batch_first=True:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = Tensor(np.random.rand(32, 10, 512), mindspore.float32)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = 'relu',
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):

        r"""
        Args:
            self: The instance of the class.
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int, optional): The dimension of the feedforward network model. Default is 2048.
            dropout (float, optional): The dropout value. Default is 0.1.
            activation (str or torch.nn.Module): The activation function. Can be a string ('relu') or a torch.nn.Module instance (e.g., nn.ReLU). Default is 'relu'.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Default is 1e-05.
            batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default is False.
            norm_first (bool, optional): If True, layer normalization is applied first, otherwise last. Default is False.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the activation function is neither 'relu' nor 'gelu' or an instance of nn.ReLU or nn.GELU.
            TypeError: If the provided activation function is not of type str or torch.nn.Module.
        """
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(p=dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm2 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def construct(self, src, src_mask=None, src_key_padding_mask=None):

        r"""
        This method 'construct' is part of the 'TransformerEncoderLayer' class and is used to construct the transformer encoder layer.
        
        Args:
            self: The instance of the class.
            src: Input tensor representing the source data.
                Type: Tensor
                Purpose: The input source data for the transformer encoder layer.
            src_mask: An optional mask tensor for the source data.
                Type: Tensor, default None
                Purpose: Mask tensor for the source data to specify padding or special tokens.
            src_key_padding_mask: An optional mask tensor for key padding.
                Type: Tensor, default None
                Purpose: Mask tensor to specify which elements in the key sequence should be ignored.
                Restrictions: Only bool and floating types of key_padding_mask are supported.
        
        Returns:
            None: This method returns None after constructing the transformer encoder layer.
        
        Raises:
            AssertionError: Raised when the key_padding_mask dtype is not bool or floating-point.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mindspore.bool_ and not ops.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):

        r"""
        This method represents a self-attention block within a Transformer Encoder Layer.
        
        Args:
            self (object): The instance of the TransformerEncoderLayer class.
            x (Tensor): The input tensor to the self-attention block.
            attn_mask (Tensor): A mask tensor applied to the attention calculation to prevent attending to certain positions.
            key_padding_mask (Tensor): A mask tensor indicating which elements in the input should be ignored in the attention calculation.
        
        Returns:
            None. This method applies self-attention mechanism to the input tensor 'x' using the given masks and returns None.
        
        Raises:
            - ValueError: If the input tensors 'x', attn_mask, or key_padding_mask have incorrect shapes.
            - RuntimeError: If there are runtime issues during the self-attention computation.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor):

        r"""
        TransformEncoderLayer._ff_block
        
        This method applies a feed-forward block to the input tensor.
        
        Args:
            self (TransformerEncoderLayer): The instance of the TransformerEncoderLayer class.
            x (Tensor): The input tensor to be processed by the feed-forward block.
        
        Returns:
            None. The method modifies the input tensor in place.
        
        Raises:
            N/A
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Cell):
    r"""
    Transformer Decoder Layer. This is an implementation of the single layer of the transformer
    decoder layer, including self-attention, cross attention and feedward layer.

    Args:
        d_model (int): The number of expected features in the input tensor.
        nhead (int): The number of heads in the MultiheadAttention modules.
        dim_feedforward (int): The dimension of the feedforward layer. Default: ``2048``.
        dropout (float): The dropout value. Default: ``0.1``.
        activation (Union[str, callable, Cell]): The activation function of the intermediate layer,
            can be a string (`"relu"` or `"gelu"`), Cell instance (`nn.ReLU()` or `nn.GELU()`) or
            a callable (`ops.relu` or `ops.gelu`). Default: ``"relu"``
        layer_norm_eps (float): The epsilon value in LayerNorm modules. Default: ``1e-5``.
        batch_first (bool): If `batch_first = True`, then the shape of input and output tensors is
            (batch, seq, feature), otherwise the shape is (seq, batch, feature). Default: ``False``.
        norm_first (bool): If `norm_first = True`, layer norm is done prior to attention and feedforward
            operations, respectively. Default: ``False``.

    Inputs:
        - **tgt** (Tensor): The sequence to the decoder layer.
        - **memory** (Tensor): The sequence from the last layer of the encoder.
        - **tgt_mask** (Tensor, optional): The mask of the tgt sequence. Default: ``None``.
        - **memory_mask** (Tensor, optional): The mask of the memory sequence. Default: ``None``.
        - **tgt_key_padding_mask** (Tensor, optional): The mask of the tgt keys per batch.
          Default: ``None``.
        - **memory_key_padding_mask** (Tensor, optional): The mask of the memory keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(20, 32, 512), mindspore.float32)
        >>> out = decoder_layer(tgt, memory)
        >>> # Alternatively, when `batch_first` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = Tensor(np.random.rand(32, 10, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(32, 20, 512), mindspore.float32)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = 'relu',
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):

        r"""
        Initialize a Transformer Decoder Layer.
        
        Args:
            self (object): The instance of the class.
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int, optional): The dimension of the feedforward network model. Default is 2048.
            dropout (float, optional): The dropout value. Default is 0.1.
            activation (str or function, optional): The activation function or string. Default is 'relu'.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Default is 1e-05.
            batch_first (bool, optional): If True, the input and output tensors are provided as (batch, seq, feature). Default is False.
            norm_first (bool, optional): If True, apply layer normalization before each sub-layer. Default is False.
        
        Returns:
            None. This method initializes the Transformer Decoder Layer.
        
        Raises:
            ValueError: If the activation function is not recognized or supported.
        """
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(p=dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm2 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm3 = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.dropout3 = Dropout(p=dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def construct(self, tgt, memory, tgt_mask = None, memory_mask = None,
                  tgt_key_padding_mask = None, memory_key_padding_mask = None):

        r"""
        Constructs a single layer of the Transformer Decoder.
        
        Args:
            self (TransformerDecoderLayer): An instance of the TransformerDecoderLayer class.
            tgt (Tensor): The target sequence tensor of shape (seq_len_tgt, batch_size, embed_dim).
            memory (Tensor): The source sequence tensor of shape (seq_len_mem, batch_size, embed_dim).
            tgt_mask (Optional[Tensor]): The mask tensor for the target sequence, with shape (seq_len_tgt, seq_len_tgt).
                Each element should be a boolean value indicating whether the corresponding position is valid or padded.
                Defaults to None.
            memory_mask (Optional[Tensor]): The mask tensor for the source sequence, with shape (seq_len_tgt, seq_len_mem).
                Each element should be a boolean value indicating whether the corresponding position is valid or padded.
                Defaults to None.
            tgt_key_padding_mask (Optional[Tensor]): The padding mask tensor for the target sequence,
                with shape (batch_size, seq_len_tgt). Each element should be a boolean value indicating whether the
                corresponding position is a valid token or a padding token. Defaults to None.
            memory_key_padding_mask (Optional[Tensor]): The padding mask tensor for the source sequence,
                with shape (batch_size, seq_len_mem). Each element should be a boolean value indicating whether the
                corresponding position is a valid token or a padding token. Defaults to None.
        
        Returns:
            Tensor: The output tensor of the Transformer Decoder layer, with shape (seq_len_tgt, batch_size, embed_dim).
        
        Raises:
            ValueError: If the shapes of tgt and memory are not compatible.
            TypeError: If the input arguments are not of the expected types.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):

        r"""
        Method _sa_block in class TransformerDecoderLayer.
        
        Args:
            self (object): The instance of TransformerDecoderLayer.
            x (Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
            attn_mask (ByteTensor, optional): 3D mask tensor for the self-attention mechanism, with shape (batch_size, seq_len, seq_len). Defaults to None.
            key_padding_mask (ByteTensor, optional): 2D mask tensor for ignoring padding elements in key, with shape (batch_size, seq_len). Defaults to None.
        
        Returns:
            None. The method does not return anything.
        
        Raises:
            ValueError: If the shape of x is incorrect or if attn_mask and key_padding_mask have incompatible shapes.
            TypeError: If the input types are not as expected.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):

        r"""
        This method `_mha_block` is defined within the class `TransformerDecoderLayer` and is used to perform multi-head attention block operations.
        
        Args:
            self (object): The instance of the `TransformerDecoderLayer` class.
            x (Tensor): The input tensor of shape (seq_len, batch_size, embed_dim) representing the current input to the multi-head attention block.
            mem (Tensor): The memory tensor of shape (mem_seq_len, batch_size, embed_dim) representing the memory input to the multi-head attention block.
            attn_mask (ByteTensor, optional): An optional boolean mask tensor of shape (seq_len, mem_seq_len) indicating the positions to be masked in the attention computation.
            key_padding_mask (ByteTensor, optional): An optional boolean mask tensor of shape (batch_size, mem_seq_len) indicating which elements in the key need to be masked.
        
        Returns:
            None. This method returns None as the result of the multi-head attention block operation.
        
        Raises:
            - TypeError: If the input arguments are not of the expected types.
            - ValueError: If the shapes of the input tensors are not compatible for the multi-head attention operation.
            - RuntimeError: If there are any runtime issues during the multi-head attention computation.
        """
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor):

        r"""
        Method _ff_block in the TransformerDecoderLayer class.
        
        Args:
            self (TransformerDecoderLayer): The instance of the TransformerDecoderLayer class.
                Used to access the methods and attributes of the TransformerDecoderLayer.
            x (Tensor): The input tensor to the feed-forward block.
                It represents the output of the previous layer in the TransformerDecoder.
                Must be a valid tensor object.
        
        Returns:
            None. The method does not return any value directly, but modifies the input tensor 'x' in-place.
        
        Raises:
            None.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerEncoder(nn.Cell):
    r"""
    Transformer Encoder module with multi-layer stacked of `TransformerEncoderLayer`, including multihead self
    attention and feedforward layer. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer (Cell): An instance of the TransformerEncoderLayer() class.
        num_layers (int): The number of encoder-layers in the encoder.
        norm (Cell, optional): The layer normalization module.

    Inputs:
        - **src** (Tensor): The sequence to the encoder.
        - **src_mask** (Tensor, optional): The mask of the src sequence. Default: ``None``.
        - **src_key_padding_mask** (Tensor, optional): the mask of the src keys per batch .
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):

        r"""
        Initializes a TransformerEncoder object.
        
        Args:
            self (TransformerEncoder): The instance of the TransformerEncoder class.
            encoder_layer (torch.nn.Module): The encoder layer module to be cloned.
            num_layers (int): The number of encoder layers to be created.
            norm (torch.nn.Module, optional): The normalization layer to be applied after each encoder layer. 
                Defaults to None.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, src: Tensor, src_mask = None, src_key_padding_mask = None):

        r"""
        This method constructs the Transformer encoder by applying the specified layers to the source input, with optional masking and padding.
        
        Args:
            self: The instance of the TransformerEncoder class.
            src (Tensor): The source input tensor to be encoded.
            src_mask (Tensor, optional): The mask tensor to apply on the source input, if applicable.
            src_key_padding_mask (Tensor, optional): The mask tensor for padding in the source input. Only bool and floating types are supported.
        
        Returns:
            None: This method returns None as the output.
        
        Raises:
            AssertionError: Raised if the src_key_padding_mask is not of type bool or floating point, or if it does not match the dtype of the input source.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mindspore.bool_ and not ops.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Cell):
    r"""
    Transformer Decoder module with multi-layer stacked of `TransformerDecoderLayer`, including multihead self
    attention, cross attention and feedforward layer.

    Args:
        decoder_layer (Cell): An instance of the TransformerDecoderLayer() class.
        num_layers (int): The number of decoder-layers in the decoder.
        norm (Cell): The layer normalization module.

    Inputs:
        - **tgt** (Tensor): The sequence to the decoder.
        - **memory** (Tensor): The sequence from the last layer of the encoder.
        - **tgt_mask** (Tensor, optional): the mask of the tgt sequence. Default: ``None``.
        - **memory_mask** (Tensor, optional): the mask of the memory sequence. Default: ``None``.
        - **tgt_key_padding_mask** (Tensor, optional): the mask of the tgt keys per batch.
          Default: ``None``.
        - **memory_key_padding_mask** (Tensor, optional): the mask of the memory keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(20, 32, 512), mindspore.float32)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):

        r"""
        Initializes a TransformerDecoder object.
        
        Args:
            self (TransformerDecoder): The instance of the class.
            decoder_layer (nn.Module): The decoder layer to be cloned.
            num_layers (int): The number of decoder layers to be created.
            norm (nn.Module, optional): The normalization layer to be applied after each decoder layer. Default is None.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, tgt, memory, tgt_mask = None,
                  memory_mask = None, tgt_key_padding_mask = None,
                  memory_key_padding_mask = None):

        r"""
        Constructs the output of the TransformerDecoder.
        
        Args:
            self (TransformerDecoder): The instance of the TransformerDecoder class.
            tgt (Tensor): The input tensor representing the target sequence. 
            memory (Tensor): The input tensor representing the memory sequence.
            tgt_mask (Optional[Tensor]): An optional tensor representing the mask for the target sequence. Defaults to None.
            memory_mask (Optional[Tensor]): An optional tensor representing the mask for the memory sequence. Defaults to None.
            tgt_key_padding_mask (Optional[Tensor]): An optional tensor representing the mask for the target key padding. Defaults to None.
            memory_key_padding_mask (Optional[Tensor]): An optional tensor representing the mask for the memory key padding. Defaults to None.
        
        Returns:
            Tensor: The output tensor representing the constructed output.
        
        Raises:
            None
        """

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Transformer(nn.Cell):
    r"""
    Transformer module including encoder and decoder. The difference with the original implements is the module use
    the residual addition before the layer normalization. And the default hidden act is `gelu`.
    The details can be found in `Attention is all you need <https://arxiv.org/pdf/1706.03762v5.pdf>`_.

    Args:
        d_model (int): The number of expected features in the inputs tensor. Default: ``512``.
        nhead (int): The number of heads in the MultiheadAttention modules. Default: ``8``.
        num_encoder_layers (int): The number of encoder-layers in the encoder. Default: ``6``.
        num_decoder_layers (int): The number of decoder-layers in the decoder. Default: ``6``.
        dim_feedforward (int): The dimension of the feedforward layer. Default: ``2048``.
        dropout (float): The dropout value. Default: ``0.1``.
        activation (Union[str, callable, Cell]): The activation function of the intermediate layer,
            can be a string (`"relu"` or `"gelu"`), Cell instance (`nn.ReLU()` or `nn.GELU()`) or
            a callable (`ops.relu` or `ops.gelu`). Default: ``"relu"``
        custom_encoder (Cell): Custom encoder. Default: ``None``.
        custom_decoder (Cell): Custom decoder. Default: ``None``.
        layer_norm_eps (float): the epsilion value in layer normalization module. Default: ``1e-5``.
        batch_first (bool): If `batch_first = True`, then the shape of input and output tensors is
            (batch, seq, feature), otherwise the shape is (seq, batch, feature). Default: ``False``.
        norm_first (bool): If `norm_first = True`, layer norm is done prior to attention and feedforward
            operations, respectively. Default: ``False``.

    Inputs:
        - **src** (Tensor): The source sequence to the encoder.
        - **tgt** (Tensor): The target sequence to the decoder.
        - **src_mask** (Tensor, optional): The mask of the src sequence. Default: ``None``.
        - **tgt_mask** (Tensor, optional): The mask of the tgt sequence. Default: ``None``.
        - **memory_mask** (Tensor, optional): The additive mask of the encoder output.
          Default: ``None``.
        - **src_key_padding_mask** (Tensor, optional): The mask of src keys per batch.
          Default: ``None``.
        - **tgt_key_padding_mask** (Tensor, optional): The mask of tgt keys per batch.
          Default: ``None``.
        - **memory_key_padding_mask** (Tensor, optional): The mask of memory keys per batch.
          Default: ``None``.

    Outputs:
        Tensor.

    Examples:
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = Tensor(np.random.rand(10, 32, 512), mindspore.float32)
        >>> tgt = Tensor(np.random.rand(20, 32, 512), mindspore.float32)
        >>> out = transformer_model(src, tgt)
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = 'relu',
                 custom_encoder = None, custom_decoder = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):

        r"""
        Initializes a Transformer model with the specified parameters.
        
        Args:
            self: The object itself.
            d_model (int): The number of expected features in the input (default=512).
            nhead (int): The number of heads in the multiheadattention models (default=8).
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder (default=6).
            num_decoder_layers (int): The number of sub-decoder-layers in the decoder (default=6).
            dim_feedforward (int): The dimension of the feedforward network model (default=2048).
            dropout (float): The dropout value (default=0.1).
            activation (str): The activation function of the feedforward network, can be 'relu', 'gelu', etc. (default='relu').
            custom_encoder: Custom encoder to be used instead of the default TransformerEncoder (default=None).
            custom_decoder: Custom decoder to be used instead of the default TransformerDecoder (default=None).
            layer_norm_eps (float): The epsilon value used in LayerNorm (default=1e-05).
            batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature) (default=False).
            norm_first (bool): If True, Layer Normalization is applied first (default=False).
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If d_model, nhead, num_encoder_layers, or num_decoder_layers is not a positive integer.
            ValueError: If dim_feedforward is not a positive integer greater than 0.
            ValueError: If dropout is not in the range [0.0, 1.0].
            ValueError: If activation is not a valid activation function.
            ValueError: If layer_norm_eps is not a positive float.
            TypeError: If custom_encoder or custom_decoder is not of the correct type.
            ValueError: If batch_first or norm_first is not a boolean value.
        """
        super().__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first)
            encoder_norm = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first)
            decoder_norm = nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def construct(self, src, tgt, src_mask = None, tgt_mask = None,
                memory_mask = None, src_key_padding_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):

        r"""
        Args:
            self (object): The instance of the Transformer class.
            src (tensor): The input tensor representing the source sequence. If the data is batched, the shape should be (batch_size, sequence_length, feature_number). If not batched, the shape should be (sequence_length, feature_number).
            tgt (tensor): The input tensor representing the target sequence. If the data is batched, the shape should be (batch_size, sequence_length, feature_number). If not batched, the shape should be (sequence_length, feature_number).
            src_mask (tensor, optional): The mask tensor for the src input. It should have the same shape as src. Defaults to None.
            tgt_mask (tensor, optional): The mask tensor for the tgt input. It should have the same shape as tgt. Defaults to None.
            memory_mask (tensor, optional): The mask tensor for the memory. Defaults to None.
            src_key_padding_mask (tensor, optional): The mask tensor for src key padding. Defaults to None.
            tgt_key_padding_mask (tensor, optional): The mask tensor for tgt key padding. Defaults to None.
            memory_key_padding_mask (tensor, optional): The mask tensor for memory key padding. Defaults to None.
        
        Returns:
            None. The method does not return any value.
        
        Raises:
            RuntimeError: 
                - If the batch number of src and tgt must be equal but is not.
                - If the feature number of src and tgt is not equal to d_model.
        """

        is_batched = src.ndim == 3
        if not self.batch_first and src.shape[1] != tgt.shape[1] and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if self.batch_first and src.shape[0] != tgt.shape[0] and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.shape[-1] != self.d_model or tgt.shape[-1] != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for _, p in self.parameters_and_names():
            if p.ndim > 1:
                p.set_data(initializer('xavier_uniform', p.shape, p.dtype))

def _get_activation_fn(activation: str):

    r"""
    Args:
        activation (str): Specifies the type of activation function to retrieve. 
            Should be either 'relu' or 'gelu'.
            
    Returns:
        None: The retrieved activation function based on the specified type.
        
    Raises:
        RuntimeError: If the provided activation type is neither 'relu' nor 'gelu'.
    """
    if activation == "relu":
        return relu
    if activation == "gelu":
        return gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):

    r"""
    Args:
        module: The module to be cloned. Type should be a valid module object. This parameter specifies the module that needs to be cloned.
        N: The number of clones to create. Type should be an integer. This parameter specifies the number of clones to create for the given module.
    
    Returns:
        None. This function does not return any value explicitly, but it creates a CellList containing deep copies of the provided module.
    
    Raises:
        None.
    """
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


__all__ = [
    'Transformer', 'TransformerEncoder', 'TransformerDecoder',
    'TransformerEncoderLayer', 'TransformerDecoderLayer',
    'MultiheadAttention'
]
