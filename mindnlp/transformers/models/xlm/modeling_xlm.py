# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 PyTorch XLM model.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import (
    ModelOutput,
    logging,
)
from .configuration_xlm import XLMConfig
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary, SQuADHead
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer


logger = logging.get_logger(__name__)


XLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-clm-enfr-1024",
    "xlm-clm-ende-1024",
    "xlm-mlm-17-1280",
    "xlm-mlm-100-1280",
    # See all XLM models at https://hf-mirror.com/models?filter=xlm
]


def create_sinusoidal_embeddings(n_pos, dim, out):
    """
    Creates sinusoidal embeddings for positional encoding.
    
    Args:
        n_pos (int): The number of positions to be encoded.
        dim (int): The dimension of the embeddings.
        out (Tensor): The output tensor to store the sinusoidal embeddings.
    
    Returns:
        None.
    
    Raises:
        None.
    """
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = mindspore.Tensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = mindspore.Tensor(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = ops.arange(slen, dtype=mindspore.int64)
    if padding_mask is not None:
        mask = padding_mask
    else:
        # assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = lengths.shape[0]
    if causal:
        attn_mask = alen[None, None, :].tile((bs, slen, 1)) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.shape == (bs, slen)
    assert causal is False or attn_mask.shape == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    """
    A class representing a multi-head attention mechanism for neural networks.
    
    This class implements multi-head attention by dividing the input into multiple heads and processing them in parallel. 
    It includes methods for initializing the attention mechanism, pruning heads based on specific criteria, and 
    forwarding the attention output based on input, masks, and key-value pairs.

    Attributes:
        layer_id: An identifier for the attention layer.
        dim: The dimensionality of the input.
        n_heads: The number of attention heads.
        dropout: The dropout rate for attention weights.
        q_lin: Linear transformation for query vectors.
        k_lin: Linear transformation for key vectors.
        v_lin: Linear transformation for value vectors.
        out_lin: Linear transformation for the final output.
        pruned_heads: A set containing indices of pruned attention heads.

    Methods:
        __init__: Initializes the multi-head attention mechanism.
        prune_heads: Prunes specified attention heads based on given criteria.
        forward: Constructs the attention output based on input, masks, and key-value pairs.

    Note:
        This class inherits from nn.Module and is designed for neural network architectures that require multi-head 
        attention mechanisms.
    """
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        """Initialize a MultiHeadAttention object.

        Args:
            self: The MultiHeadAttention object.
            n_heads (int): The number of attention heads.
            dim (int): The dimension of the input.
            config (object): The configuration object containing the attention dropout.

        Returns:
            None

        Raises:
            AssertionError: If the dimension is not divisible by the number of attention heads.

        """
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes the attention heads in a MultiHeadAttention layer.

        Args:
            self (MultiHeadAttention): The instance of the MultiHeadAttention class.
            heads (List[int]): A list of integers representing the indices of the attention heads to be pruned.

        Returns:
            None

        Raises:
            None

        This method prunes the specified attention heads in a MultiHeadAttention layer. 
        The attention heads are pruned based on the given indices. The method performs the following steps:

        1. Calculates the attention_head_size by dividing the dimension (self.dim) by the number of heads (self.n_heads).
        2. If the list of heads is empty, the method returns without performing any pruning.
        3. Calls the 'find_pruneable_heads_and_indices' function to find the pruneable heads and their corresponding 
        indices based on the given parameters (heads, self.n_heads, attention_head_size, self.pruned_heads).
        4. Prunes the linear layers q_lin, k_lin, v_lin, and out_lin using the 'prune_linear_layer' function, passing 
        the calculated indices (index) as a parameter.
        5. Updates the number of heads (self.n_heads) by subtracting the length of the pruneable heads list.
        6. Updates the dimension (self.dim) by multiplying the attention_head_size with the updated number of heads.
        7. Updates the set of pruned heads (self.pruned_heads) by adding the pruneable heads.

        Note:
            Pruning attention heads reduces the computational complexity of the MultiHeadAttention layer.
        """
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, _ = input.shape
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.shape[1]
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).swapaxes(1, 2)

        def unshape(x):
            """compute context"""
            return x.swapaxes(1, 2).view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = ops.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = ops.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = ops.matmul(q, k.swapaxes(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        scores = scores.masked_fill(mask, float(ops.finfo(scores.dtype).min))  # (bs, n_heads, qlen, klen)

        weights = ops.softmax(scores.float(), dim=-1).astype(scores.dtype)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = ops.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TransformerFFN(nn.Module):

    """
    TransformerFFN is a class that represents a feed-forward neural network component of a transformer model. 
    It inherits from nn.Module and includes methods for initializing the network and forwarding the forward pass.

    Attributes:
        in_dim (int): The input dimension of the network.
        dim_hidden (int): The dimension of the hidden layer in the network.
        out_dim (int): The output dimension of the network.
        config (object): The configuration object containing parameters for the network.

    Methods:
        __init__: Initializes the TransformerFFN instance with the specified input, hidden, and output dimensions, 
            as well as the configuration object.
        forward: Constructs the forward pass of the network using chunking for the specified input.
        ff_chunk: Implements the feed-forward chunk of the network, including linear transformations, 
            activation function, and dropout.

    Note:
        This class assumes the presence of nn, ops, and apply_chunking_to_forward functions and objects for neural 
        network and tensor operations.
    """
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        """
        Initializes an instance of the TransformerFFN class.

        Args:
            self (TransformerFFN): The instance of the TransformerFFN class.
            in_dim (int): The input dimension.
            dim_hidden (int): The dimension of the hidden layer.
            out_dim (int): The output dimension.
            config (object): The configuration object containing various settings.

        Returns:
            None.

        Raises:
            None.

        """
        super().__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = F.gelu if config.gelu_activation else F.relu
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(self, input):
        """
        Method 'forward' in the class 'TransformerFFN'.

        Args:
            self (object): The instance of the TransformerFFN class.
            input (any): The input data to be processed by the method.

        Returns:
            None.

        Raises:
            None.
        """
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        """
        Method 'ff_chunk' in the class 'TransformerFFN'.

        Args:
            self (object): The instance of the TransformerFFN class.
            input (tensor): The input tensor to the feedforward chunk.

        Returns:
            None. The method returns the processed input tensor after passing through the feedforward chunk layers.

        Raises:
            ValueError: If the input tensor is not in the expected format.
            RuntimeError: If an issue occurs during the dropout operation.
        """
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class XLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLMConfig
    load_tf_weights = None
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        """
        Generates dummy inputs for the XLMPreTrainedModel.

        Args:
            self: An instance of the XLMPreTrainedModel class.

        Returns:
            dict: A dictionary containing the dummy inputs for the model.
                The dictionary has the following keys:

                - 'input_ids': A tensor representing the input sequences. The shape of the tensor is
                (num_sequences, sequence_length), where num_sequences is the number of input sequences and
                sequence_length is the maximum length of any sequence.
                - 'attention_mask': A tensor representing the attention mask for the input sequences.
                The shape of the tensor is the same as 'input_ids' and contains 0s and 1s, where 0 indicates padding and
                1 indicates a valid token.
                - 'langs': A tensor representing the language embeddings for the input sequences.
                The shape of the tensor is the same as 'input_ids'. If the model is configured to use language
                embeddings and there are multiple languages,  the tensor contains language embeddings for each token.
                Otherwise, it is set to None.

        Raises:
            None.
        """
        inputs_list = mindspore.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = mindspore.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = mindspore.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, nn.Embedding):
            if self.config is not None and self.config.embed_init_std is not None:
                weight = np.random.normal(0.0, self.config.embed_init_std, cell.weight.shape)
                if cell.padding_idx:
                    weight[cell.padding_idx] = 0

                cell.weight.assign_value(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.Linear):
            if self.config is not None and self.config.init_std is not None:
                cell.weight.assign_value(initializer(Normal(self.config.init_std),
                                                        cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        if isinstance(cell, nn.LayerNorm):
            cell.weight.assign_value(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))

@dataclass
class XLMForQuestionAnsweringOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a `SquadHead`.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions`
            are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`mindspore.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if
            `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`mindspore.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if
            `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`mindspore.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`,
            *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`mindspore.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`,
            *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`mindspore.Tensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or
            `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """
    loss: Optional[mindspore.Tensor] = None
    start_top_log_probs: Optional[mindspore.Tensor] = None
    start_top_index: Optional[mindspore.Tensor] = None
    end_top_log_probs: Optional[mindspore.Tensor] = None
    end_top_index: Optional[mindspore.Tensor] = None
    cls_logits: Optional[mindspore.Tensor] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class XLMModel(XLMPreTrainedModel):

    """
    XLMModel is a class representing a transformer model for cross-lingual language model pre-training based on the
    XLM architecture.

    This class inherits from XLMPreTrainedModel and implements various methods for initializing the model, handling
    embeddings, pruning heads, and forwarding the model for inference.

    The __init__ method initializes the model with configuration parameters and sets up the model's architecture.
    It handles encoder-decoder setup, embeddings, attention mechanisms, layer normalization, and other model components.

    The get_input_embeddings method returns the input embeddings used in the model, while set_input_embeddings allows
    for updating the input embeddings.

    The _prune_heads method prunes specific attention heads in the model based on the provided dictionary of
    {layer_num: list of heads}.

    The forward method forwards the model for inference, taking input tensors for input_ids, attention_mask, langs,
    token_type_ids, position_ids, lengths, cache, head_mask, inputs_embeds, output settings, and returns the model
    output or a BaseModelOutput object depending on the return_dict setting.

    Overall, XLMModel provides a comprehensive implementation of the XLM transformer model for cross-lingual language
    tasks.
    """
    def __init__(self, config):
        """
        This method initializes an instance of the XLMModel class with the provided configuration.

        Args:
            self: The instance of the XLMModel class.
            config:
                An object containing configuration parameters for the XLMModel.

                - Type: object
                - Purpose: Specifies the configuration settings for the XLMModel.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            NotImplementedError: If the provided configuration indicates that the XLMModel is used as a decoder,
                since XLM can only be used as an encoder.
            AssertionError: If the transformer dimension is not a multiple of the number of heads.

        """
        super().__init__(config)

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError("Currently XLM can only be used as an encoder")
        # self.with_output = with_output
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        # self.dico = dico
        # self.id2lang = config.id2lang
        # self.lang2id = config.lang2id
        # assert len(self.dico) == self.n_words
        # assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = config.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = config.n_heads  # 8 by default
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        if config.sinusoidal_embeddings:
            create_sinusoidal_embeddings(config.max_position_embeddings, self.dim, out=self.position_embeddings.weight)
        if config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = nn.Embedding(self.n_langs, self.dim)
        self.embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm([self.dim], eps=config.layer_norm_eps)

        # transformer layers
        attentions = []
        layer_norm1 = []
        ffns = []
        layer_norm2 = []
        # if self.is_decoder:
        #     self.layer_norm15 = nn.ModuleList()
        #     self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            attentions.append(MultiHeadAttention(self.n_heads, self.dim, config=config))
            layer_norm1.append(nn.LayerNorm([self.dim], eps=config.layer_norm_eps))
            # if self.is_decoder:
            #     self.layer_norm15.append(nn.LayerNorm(self.dim, eps=config.layer_norm_eps))
            #     self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, config=config))
            layer_norm2.append(nn.LayerNorm([self.dim], eps=config.layer_norm_eps))

        self.attentions = nn.ModuleList(attentions)
        self.layer_norm1 = nn.ModuleList(layer_norm1)
        self.ffns = nn.ModuleList(ffns)
        self.layer_norm2 = nn.ModuleList(layer_norm2)

        if hasattr(config, "pruned_heads"):
            pruned_heads = config.pruned_heads.copy().items()
            config.pruned_heads = {}
            for layer, heads in pruned_heads:
                if self.attentions[int(layer)].n_heads == config.n_heads:
                    self.prune_heads({int(layer): list(map(int, heads))})

        # Initialize weights and apply final processing
        self.post_init()
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the XLMModel.

        Args:
            self (XLMModel): An instance of the XLMModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        """Set the input embeddings for the XLMModel.

        This method sets the input embeddings for the XLMModel using the given new_embeddings.

        Args:
            self (XLMModel): The instance of the XLMModel class.
            new_embeddings (Any): The new embeddings to set for the XLMModel. It can be of any type.

        Returns:
            None.

        Raises:
            None.

        """
        self.embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        langs: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        lengths: Optional[mindspore.Tensor] = None,
        cache: Optional[Dict[str, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        '''
        Constructs the XLM model.

        Args:
            self: The object itself.
            input_ids (Optional[mindspore.Tensor]): The input tensor of shape (batch_size, sequence_length).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape (batch_size, sequence_length).
            langs (Optional[mindspore.Tensor]): The language tensor of shape (batch_size, sequence_length).
            token_type_ids (Optional[mindspore.Tensor]): The token type tensor of shape (batch_size, sequence_length).
            position_ids (Optional[mindspore.Tensor]): The position tensor of shape (batch_size, sequence_length).
            lengths (Optional[mindspore.Tensor]): The lengths tensor of shape (batch_size,).
            cache (Optional[Dict[str, mindspore.Tensor]]): The cache tensor.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor.
            inputs_embeds (Optional[mindspore.Tensor]): The input embeddings tensor of shape
                (batch_size, sequence_length, embedding_size).
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary.

        Returns:
            Union[Tuple, BaseModelOutput]: The model output, which can be a tuple of tensors or a BaseModelOutput object.

        Raises:
            AssertionError: If the lengths tensor shape does not match the batch size or if the maximum length in the
                lengths tensor exceeds the sequence length.
            AssertionError: If the position_ids tensor shape does not match the input tensor shape.
            AssertionError: If the langs tensor shape does not match the input tensor shape.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            bs, slen = input_ids.shape
        else:
            bs, slen = inputs_embeds.shape[:-1]

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_index).sum(axis=1).long()
            else:
                lengths = mindspore.tensor([slen] * bs)
        # mask = input_ids != self.pad_index

        # check inputs
        assert lengths.shape[0] == bs
        # assert lengths.max().item() <= slen
        # input_ids = input_ids.swapaxes(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.shape[0] == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = ops.arange(src_len.max(), dtype=mindspore.int64) < src_len[:, None]

        # position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :slen]
        else:
            assert position_ids.shape == (bs, slen)  # (slen, bs)
            # position_ids = position_ids.swapaxes(0, 1)

        # langs
        if langs is not None:
            assert langs.shape == (bs, slen)  # (slen, bs)
            # langs = langs.swapaxes(0, 1)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layers)

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _slen = slen - cache["slen"]
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if langs is not None and self.use_lang_emb and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for i in range(self.n_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attentions[i](
                tensor,
                attn_mask,
                cache=cache,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = F.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["slen"] += tensor.shape[1]

        # move back sequence length to dimension 0
        # tensor = tensor.swapaxes(0, 1)

        if not return_dict:
            return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)


class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, config):
        """
        Initialize the XLMPredLayer class.

        Args:
            self: The instance of the XLMPredLayer class.
            config:
                A configuration object containing the following attributes:

                - asm (bool): Indicates whether to use Adaptive Softmax. If False, a Dense layer will be used.
                - n_words (int): Number of words in the vocabulary.
                - pad_index (int): Index of the padding token.
                - emb_dim (int): Dimension of the embedding.
                - asm_cutoffs (list of int): Cutoffs for Adaptive Softmax.
                - asm_div_value (float): Divisor value for Adaptive Softmax.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim

        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        if self.asm is False:
            scores = self.proj(x)
            outputs = (scores,) + outputs
            if y is not None:
                loss = F.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction="mean")
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs

        return outputs


class XLMWithLMHeadModel(XLMPreTrainedModel):

    """
    XLMWithLMHeadModel represents a transformer model with a language modeling head based on the XLM
    (Cross-lingual Language Model) architecture.

    This class inherits from XLMPreTrainedModel and provides methods for initializing the model, getting and setting
    output embeddings, preparing inputs for generation, and forwarding the model for language modeling tasks.

    Attributes:
        transformer (XLMModel): The XLMModel instance used for the transformer architecture.
        pred_layer (XLMPredLayer): The XLMPredLayer instance used for the language modeling head.

    Methods:
        __init__: Initializes the XLMWithLMHeadModel instance with the given configuration.
        get_output_embeddings: Returns the output embeddings from the language modeling head.
        set_output_embeddings: Sets new output embeddings for the language modeling head.
        prepare_inputs_for_generation: Prepares input tensors for language generation tasks.
        forward: Constructs the model for language modeling tasks and returns the masked language model output.

    Note:
        The forward method includes detailed documentation for its parameters and return value, including optional
        and shifted labels for language modeling.
    """
    _tied_weights_keys = ["pred_layer.proj.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the XLMWithLMHeadModel class.

        Args:
            self (XLMWithLMHeadModel): The current instance of the XLMWithLMHeadModel class.
            config: The configuration object for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.transformer = XLMModel(config)
        self.pred_layer = XLMPredLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the XLMWithLMHeadModel.

        Args:
            self (XLMWithLMHeadModel): The instance of the XLMWithLMHeadModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.pred_layer.proj

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the XLM model with a language modeling head.

        Args:
            self (XLMWithLMHeadModel): The instance of the XLMWithLMHeadModel class.
            new_embeddings (torch.nn.Embedding): The new embeddings to be set as the output embeddings.
                This parameter should be an instance of torch.nn.Embedding class representing the new embeddings.

        Returns:
            None:
                This method does not return any value explicitly but updates the output embeddings of the model in-place.

        Raises:
            TypeError: If the new_embeddings parameter is not an instance of torch.nn.Embedding.
            ValueError: If the shape or type of the new_embeddings parameter is not compatible with the model's
                requirements.
        """
        self.pred_layer.proj = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare the inputs for generation in XLMWithLMHeadModel.

        Args:
            self: The instance of the XLMWithLMHeadModel class.
            input_ids (Tensor): The input tensor containing token IDs. Shape (batch_size, sequence_length).

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation.

                - 'input_ids' (Tensor): The input tensor with additional mask token appended.
                Shape (batch_size, sequence_length + 1).
                - 'langs' (Tensor or None): The tensor specifying the language IDs for each token,
                or None if lang_id is not provided.

        Raises:
            ValueError: If the input_ids tensor is not valid or if an error occurs during tensor operations.
            TypeError: If the input_ids tensor is not of type Tensor.
        """
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        effective_batch_size = input_ids.shape[0]
        mask_token = ops.full((effective_batch_size, 1), mask_token_id, dtype=mindspore.int64)
        input_ids = ops.cat([input_ids, mask_token], dim=1)
        if lang_id is not None:
            langs = ops.full_like(input_ids, lang_id)
        else:
            langs = None
        return {"input_ids": input_ids, "langs": langs}

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        langs: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        lengths: Optional[mindspore.Tensor] = None,
        cache: Optional[Dict[str, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        if not return_dict:
            return outputs + transformer_outputs[1:]

        return MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLMForSequenceClassification(XLMPreTrainedModel):

    """
    XLMForSequenceClassification includes the logic to classify sequences using a transformer-based model.
    This class inherits from XLMPreTrainedModel and implements the specific logic for sequence classification
    using the XLM model.

    Attributes:
        num_labels (int): The number of labels for sequence classification.
        config (XLMConfig): The configuration for the XLM model.
        transformer (XLMModel): The transformer model used for sequence classification.
        sequence_summary (SequenceSummary): The sequence summarization layer.

    Args:
        config (XLMConfig): The configuration object for the XLMForSequenceClassification model.

    Methods:
        forward:
            This method forwards the sequence classification model and returns the sequence classifier output.

    Raises:
        ValueError: If the number of labels is invalid or the problem type is not recognized.

    Returns:
        Union[Tuple, SequenceClassifierOutput]:
            A tuple containing the loss and output if loss is not None, else the output.

    Raises:
        ValueError: If the number of labels is invalid or the problem type is not recognized.
    """
    def __init__(self, config):
        """
        Initializes an instance of the XLMForSequenceClassification class.

        Args:
            self (XLMForSequenceClassification): The current instance of the XLMForSequenceClassification class.
            config (XLMConfig): The configuration object containing settings for the model initialization.
                It must include the number of labels 'num_labels' and other necessary configurations.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type XLMConfig.
            ValueError: If the config object does not contain the required 'num_labels' attribute.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        langs: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        lengths: Optional[mindspore.Tensor] = None,
        cache: Optional[Dict[str, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLMForQuestionAnsweringSimple(XLMPreTrainedModel):

    """
    This class represents a simple XLM model for question answering. It inherits from XLMPreTrainedModel
    and includes methods for forwarding the model and handling question answering tasks.

    Attributes:
        transformer (XLMModel): The XLMModel instance for the transformer component of the model.
        qa_outputs (nn.Linear): The output layer for question answering predictions.

    Methods:
        forward: Construct the model for question answering tasks, with optional input parameters and return values.
            This method includes detailed descriptions of the input and output tensors, as well as
            the expected behavior of the model during inference.

    Note:
        This class is intended for use with the MindSpore framework.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the 'XLMForQuestionAnsweringSimple' class.

        Args:
            self: The object instance.
            config: An instance of the 'XLMConfig' class containing the configuration parameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        langs: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        lengths: Optional[mindspore.Tensor] = None,
        cache: Optional[Dict[str, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLMForQuestionAnswering(XLMPreTrainedModel):

    """
    The `XLMForQuestionAnswering` class is a model for question answering tasks using the XLM
    (Cross-lingual Language Model) architecture. It is designed to take input sequences and output the start and end
    positions of the answer within the sequence.

    This class inherits from `XLMPreTrainedModel`, which provides the base functionality for loading and using
    pre-trained XLM models.

    Attributes:
        `transformer`: An instance of the `XLMModel` class, which is responsible for encoding the input sequences.
        `qa_outputs`: An instance of the `SQuADHead` class, which is responsible for predicting the start and end
        positions of the answer.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, XLMForQuestionAnswering
        ...
        >>> tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-en-2048")
        >>> model = XLMForQuestionAnswering.from_pretrained("xlm-mlm-en-2048")
        ...
        >>> input_ids = mindspore.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        >>> start_positions = mindspore.tensor([1])
        >>> end_positions = mindspore.tensor([3])
        ...
        >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        ```

    """
    def __init__(self, config):
        """
        Initializes an instance of the XLMForQuestionAnswering class.

        Args:
            self (XLMForQuestionAnswering): The current instance of the XLMForQuestionAnswering class.
            config: The configuration object containing settings for the XLMForQuestionAnswering model.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not of the expected type.
            ValueError: If there are issues during the initialization process of the XLMForQuestionAnswering instance.
        """
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = SQuADHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        langs: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        lengths: Optional[mindspore.Tensor] = None,
        cache: Optional[Dict[str, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        is_impossible: Optional[mindspore.Tensor] = None,
        cls_index: Optional[mindspore.Tensor] = None,
        p_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, XLMForQuestionAnsweringOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            is_impossible (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels whether a question has an answer or no answer (SQuAD 2.0)
            cls_index (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the classification token to use as input for computing plausibility of the
                answer.
            p_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
                masked. 0.0 mean token is not masked.

        Returns:
            `Union[Tuple, XLMForQuestionAnsweringOutput]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, XLMForQuestionAnswering
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-en-2048")
            >>> model = XLMForQuestionAnswering.from_pretrained("xlm-mlm-en-2048")
            ...
            >>> input_ids = mindspore.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
            ...     0
            ... )  # Batch size 1
            >>> start_positions = mindspore.tensor([1])
            >>> end_positions = mindspore.tensor([3])
            ...
            >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
            >>> loss = outputs.loss
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]

        outputs = self.qa_outputs(
            output,
            start_positions=start_positions,
            end_positions=end_positions,
            cls_index=cls_index,
            is_impossible=is_impossible,
            p_mask=p_mask,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs + transformer_outputs[1:]

        return XLMForQuestionAnsweringOutput(
            loss=outputs.loss,
            start_top_log_probs=outputs.start_top_log_probs,
            start_top_index=outputs.start_top_index,
            end_top_log_probs=outputs.end_top_log_probs,
            end_top_index=outputs.end_top_index,
            cls_logits=outputs.cls_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLMForTokenClassification(XLMPreTrainedModel):

    """XLMForTokenClassification

    This class is a token classification model based on the XLM architecture. It is designed for token-level
    classification tasks, such as named entity recognition or part-of-speech tagging. The model takes input sequences
    and predicts a label for each token in the sequence.

    The XLMForTokenClassification class inherits from the XLMPreTrainedModel class, which provides the basic
    functionality for pre-training and fine-tuning XLM models.

    Attributes:
        num_labels (int): The number of labels for token classification.
        transformer (XLMModel): The XLMModel instance used for the transformer architecture.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Linear layer for classification.

    Methods:
        __init__: Initializes the XLMForTokenClassification instance.
        forward: Constructs the XLMForTokenClassification model and performs token classification.

    """
    def __init__(self, config):
        """Initialize the XLMForTokenClassification class.

            Args:
                config (XLMConfig): The configuration object for the model.

            Returns:
                None

            Raises:
                None
            """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLMModel(config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        langs: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        lengths: Optional[mindspore.Tensor] = None,
        cache: Optional[Dict[str, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLMForMultipleChoice(XLMPreTrainedModel):

    """
    XLMForMultipleChoice represents a XLM model for multiple choice tasks. It is a subclass of XLMPreTrainedModel
    and includes methods for building the model, processing input data, and computing multiple choice classification
    loss.

    Attributes:
        transformer: An instance of XLMModel for processing input data.
        sequence_summary: An instance of SequenceSummary for summarizing the transformer outputs.
        logits_proj: An instance of nn.Linear for projecting the sequence summary outputs.

    Args:
        config: The model configuration.
        *inputs: Variable length input for the model.
        **kwargs: Additional keyword arguments for the model.

    Methods:
        forward: Constructs the model and processes the input data for multiple choice tasks.

    Returns:
        Union[Tuple, MultipleChoiceModelOutput]: A tuple containing the loss and model outputs or an instance of
            MultipleChoiceModelOutput.

    Note:
        This class inherits from XLMPreTrainedModel and follows the implementation details specific to XLM
        multiple choice models.

    See Also:
        XLMPreTrainedModel: The base class for all XLM model implementations.
        XLMModel: The base transformer model used for processing input data.
        SequenceSummary: A class for summarizing transformer outputs.
        MultipleChoiceModelOutput: The output class for multiple choice model predictions.

    Raises:
        ValueError: If invalid input data or model configuration is provided.
        RuntimeError: If errors occur during model processing or loss computation.

    Example:
        ```python
        >>> # Initialize XLMForMultipleChoice model
        >>> model = XLMForMultipleChoice(config)
        ...
        >>> # Process input data and compute multiple choice classification loss
        >>> outputs = model.forward(input_ids, attention_mask, labels=labels)
        ...
        >>> # Access model outputs
        >>> logits = outputs.logits
        >>> hidden_states = outputs.hidden_states
        >>> attentions = outputs.attentions
        ```
    """
    def __init__(self, config, *inputs, **kwargs):
        """
        Initializes the XLMForMultipleChoice class.
        
        Args:
            self: The instance of the class.
            config: The configuration object containing various parameters for model initialization.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the provided config parameter is not of the correct type.
            ValueError: If the config parameter is missing required attributes.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config, *inputs, **kwargs)

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.num_labels, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        langs: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        lengths: Optional[mindspore.Tensor] = None,
        cache: Optional[Dict[str, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        langs = langs.view(-1, langs.shape[-1]) if langs is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the "
                "attention mask instead."
            )
            lengths = None

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

__all__ = [
        "XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMForMultipleChoice",
        "XLMForQuestionAnswering",
        "XLMForQuestionAnsweringSimple",
        "XLMForSequenceClassification",
        "XLMForTokenClassification",
        "XLMModel",
        "XLMPreTrainedModel",
        "XLMWithLMHeadModel",
    ]
