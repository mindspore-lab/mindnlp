# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# pylint: disable=too-many-instance-attributes
# pylint: disable=C0103
# pylint: disable=W0622
"""
xlm module
"""
import math
import itertools
import inspect
from typing import List, Set, Tuple, Callable, Optional
import mindspore
import numpy as np
from mindspore import ops, nn, Parameter
from mindspore.common.initializer import Normal, initializer
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mindnlp.models.utils.utils import SequenceSummary, SQuADHead
from mindnlp.abc import PreTrainedModel
from .xlm_config import XLMConfig
from ..utils import logging
from ..utils.activations import get_activation

logger = logging.get_logger(__name__)


def create_sinusoidal_embeddings(n_pos, dim, out):
    """
    create_sinusoidal_embeddings
    """
    position_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = mindspore.Tensor(
        np.sin(position_enc[:, 0::2]), dtype=mindspore.float32)
    out[:, 1::2] = mindspore.Tensor(
        np.cos(position_enc[:, 1::2]), dtype=np.float32)
    out.detach_()
    out.requires_grad = False


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], mindspore.Tensor]:
    """
    find_pruneable_heads_and_indices
    """
    mask = ops.ones(n_heads, head_size)
    # Convert to set and remove already pruned heads
    heads = set(heads) - already_pruned_heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).equal(1)
    index: mindspore.Tensor(dtype=mindspore.int64) = mindspore.numpy.arange(
        len(mask))[mask].astype(mindspore.int64)
    return heads, index


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = mindspore.numpy.arange(slen, dtype=mindspore.int64)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max() <= slen
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = lengths.shape[0]
    if causal:
        attn_mask = alen[None, None, :].repeat(
            bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.shape == (bs, slen)
    assert causal is False or attn_mask.shape == (bs, slen, slen)

    return mask, attn_mask


def prune_linear_layer(layer: nn.Dense, index: mindspore.int64, dim: int = 0) -> nn.Dense:
    """
    prune_linear_layer
    """
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Dense(new_size[1],
                         new_size[0],
                         has_bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy()
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy()
        new_layer.bias.requires_grad = True
    return new_layer


def apply_chunking_to_forward(
    forward_fn: Callable[..., mindspore.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> mindspore.Tensor:
    """
    apply_chunking_to_forward
    """

    assert len(
        input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method
    # -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(
        inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.split(
            num_chunks, axis=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk)
                              for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return ops.cat(output_chunks, axis=chunk_dim)

    return forward_fn(*input_tensors)


class MultiHeadAttention(nn.Cell):
    """
    MultiHeadAttention
    """
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Dense(dim, dim)
        self.k_lin = nn.Dense(dim, dim)
        self.v_lin = nn.Dense(dim, dim)
        self.out_lin = nn.Dense(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        prune_heads
        """
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, _ = input.shape  # bs,qlen,dim
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.shape[1]
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.ndim == 3 else (
            bs, 1, 1, klen)

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(0, 2, 1, 3)

        def unshape(x):
            """compute context"""
            return x.transpose(0, 2, 1, 3).view(bs, -1, self.n_heads * dim_per_head)

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
                    # (bs, n_heads, klen, dim_per_head)
                    k = mindspore.ops.cat([k_, k], axis=2)
                    # (bs, n_heads, klen, dim_per_head)
                    v = mindspore.ops.cat([v_, v], axis=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        scores = mindspore.ops.matmul(q, k.transpose(
            0, 1, 3, 2)) / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(
            scores)  # (bs, n_heads, qlen, klen)
        scores.masked_fill(mask, mindspore.Tensor(
                           np.finfo(mindspore.dtype_to_nptype(scores.dtype)).min))  # (bs, n_heads, qlen, klen)

        # (bs, n_heads, qlen, klen)
        weights = ops.softmax(scores.float(), axis=-1).astype(scores.dtype)
        if self.training:
            # (bs, n_heads, qlen, klen)
            weights = ops.dropout(weights, p=self.dropout)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        # (bs, n_heads, qlen, dim_per_head)
        context = mindspore.ops.matmul(weights, v)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TransformerFFN(nn.Cell):
    """
    TransformerFFN
    """

    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Dense(in_dim, dim_hidden)
        self.lin2 = nn.Dense(dim_hidden, out_dim)
        self.act = get_activation(
            "gelu") if config.gelu_activation else ops.relu
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def construct(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        """
        ff_chunk
        """
        x = self.lin1(input)

        x = self.act(x)
        x = self.lin2(x)
        if self.training:
            x = ops.dropout(x, p=self.dropout)
        return x


class XLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # TODO
    def get_position_embeddings(self):
        pass

    # TODO
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    # TODO
    def set_input_embeddings(self, value: "nn.Cell"):
        pass

    config_class = XLMConfig
    load_tf_weights = None
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        """
        dummy_inputs
        """
        inputs_list = mindspore.Tensor(
            [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = mindspore.Tensor(
            [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = mindspore.Tensor(
                [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.init_std),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            embedding_table = initializer(Normal(self.config.embed_init_std),
                                                 cell.embedding_table.shape,
                                                 cell.embedding_table.dtype)
            if cell.padding_idx is not None:
                embedding_table[cell.padding_idx] = 0
            cell.embedding_table.set_data(embedding_table)
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))


class XLMModel(XLMPreTrainedModel):
    """
    XLMMODEL
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError(
                "Currently XLM can only be used as an encoder")
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
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, self.dim)
        if config.sinusoidal_embeddings:
            create_sinusoidal_embeddings(config.max_position_embeddings,
                                         self.dim,
                                         out=self.position_embeddings.embedding_table)
        if config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = nn.Embedding(self.n_langs, self.dim)

        self.embeddings = nn.Embedding(
            self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(
            (self.dim,), epsilon=config.layer_norm_eps)

        # transformer layers
        self.attentions = nn.CellList()
        self.layer_norm1 = nn.CellList()
        self.ffns = nn.CellList()
        self.layer_norm2 = nn.CellList()
        # if self.is_decoder:
        #     self.layer_norm15 = nn.ModuleList()
        #     self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(
                self.n_heads, self.dim, config=config))
            self.layer_norm1.append(nn.LayerNorm(
                (self.dim,), epsilon=config.layer_norm_eps))
            self.ffns.append(TransformerFFN(
                self.dim, self.hidden_dim, self.dim, config=config))
            self.layer_norm2.append(nn.LayerNorm(
                (self.dim,), epsilon=config.layer_norm_eps))

        if hasattr(config, "pruned_heads"):
            pruned_heads = config.pruned_heads.copy().items()
            config.pruned_heads = {}
            for layer, heads in pruned_heads:
                if self.attentions[int(layer)].n_heads == config.n_heads:
                    self.prune_heads({int(layer): list(map(int, heads))})

        # Initialize weights and apply final processing
        self.post_init()
        self.position_ids = Parameter(ops.BroadcastTo(shape=(1, -1))
                                      (ops.arange(config.max_position_embeddings)))

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value: "nn.Cell"):
        self.embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = head_mask.expand_dims(0).expand_dims(
                0).expand_dims(-1).expand_dims(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.ndim == 2:
            head_mask = head_mask.expand_dims(
                1).expand_dims(-1).expand_dims(-1)
            # We can specify head_mask for each layer
        assert head_mask.ndim == 5, f"head_mask.dim != 5, instead {head_mask.ndim}"
        # switch to float if need + fp16 compatibility
        head_mask = head_mask.astype(dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self, head_mask: Optional[mindspore.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> mindspore.Tensor:
        """get_head_mask"""
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(
                head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.expand_dims(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple:
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
                lengths = (input_ids != self.pad_index).sum(
                    axis=1).astype(mindspore.int64)
            else:
                lengths = mindspore.Tensor([slen] * bs)

        # mask = input_ids != self.pad_index
        # check inputs
        assert lengths.shape[0] == bs
        assert mindspore.Tensor.max(lengths) <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(
            slen, lengths, self.causal, padding_mask=attention_mask)

        # position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :slen]
        else:
            assert position_ids.shape == (bs, slen)  # (slen, bs)
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.shape == (bs, slen)  # (slen, bs)
            # langs = langs.transpose(0, 1)

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

        tensor = inputs_embeds + \
            self.position_embeddings(position_ids).expand_as(inputs_embeds)

        if langs is not None and self.use_lang_emb and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)

        if self.training:
            tensor = ops.dropout(tensor, p=self.dropout)

        tensor *= ops.expand_dims(mask, axis=-1).astype(tensor.dtype)

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
            if self.training:
                attn = ops.dropout(attn, p=self.dropout)
            tensor = tensor + attn

            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= ops.expand_dims(mask, axis=-1).astype(tensor.dtype)

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["slen"] += tensor.shape[1]

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)
        return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)


class XLMPredLayer(nn.Cell):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super().__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim

        if config.asm is False:
            self.proj = nn.Dense(dim, config.n_words, has_bias=True)
        # else :TO DO nn.AdaptiveLogSoftmaxWithLoss

        # TODO:def AdaptiveLogSoftmaxWithLoss():

    def construct(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        if self.asm is False:
            scores = self.proj(x)
            outputs = (scores,) + outputs

            if y is not None:
                loss = ops.cross_entropy(scores.view(-1, self.n_words),
                                         y.view(-1),
                                         reduction="mean")
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
    XLMWithLMHeadModel
    """
    _keys_to_ignore_on_load_missing = ["pred_layer.proj.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = XLMModel(config)
        self.pred_layer = XLMPredLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        get_output_embeddings
        """
        return self.pred_layer.proj

    def set_output_embeddings(self, new_embeddings):
        """
        set_output_embeddings
        """
        self.pred_layer.proj = new_embeddings

    # TODO(self, input_ids, **kwargs)
    def prepare_inputs_for_generation(self, input_ids):
        """
        prepare_inputs_for_generation
        """
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        effective_batch_size = input_ids.shape[0]
        mask_token = mindspore.ops.full(
            (effective_batch_size, 1), mask_token_id, dtype=mindspore.int64)
        input_ids = ops.cat([input_ids, mask_token], axis=1)
        if lang_id is not None:
            langs = mindspore.ops.full_like(input_ids, lang_id)
        else:
            langs = None
        return {"input_ids": input_ids, "langs": langs}

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple:

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
        # (loss, logits) or (logits,) depending on if labels are provided.
        outputs = self.pred_layer(output, labels)

        return outputs + transformer_outputs[1:]


class XLMForSequenceClassification(XLMPreTrainedModel):
    """
    XLMForSequenceClassification
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple:

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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(ops.squeeze(logits), ops.squeeze(labels))
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        output = (logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output


class XLMForQuestionAnsweringSimple(XLMPreTrainedModel):
    """
    XLMForQuestionAnsweringSimple
    """

    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple:

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
                start_positions = ops.squeeze(start_positions, axis=-1)
            if len(end_positions.shape) > 1:
                end_positions = ops.squeeze(end_positions, axis=-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + transformer_outputs[1:]
        return ((total_loss,) + output) if total_loss is not None else output


class XLMForQuestionAnswering(XLMPreTrainedModel):
    """
    XLMForQuestionAnswering
    """

    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = SQuADHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        is_impossible=None,
        cls_index=None,
        p_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple:

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
            # return_dict=return_dict,
        )

        return outputs + transformer_outputs[1:]


class XLMForTokenClassification(XLMPreTrainedModel):
    """
    XLMForTokenClassification
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLMModel(config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple:

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
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


class XLMForMultipleChoice(XLMPreTrainedModel):
    """
    XLMForMultipleChoice
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Dense(config.num_labels, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1,
                                   input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.view(
            -1, position_ids.shape[-1]) if position_ids is not None else None
        langs = langs.view(-1, langs.shape[-1]) if langs is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1,
                               inputs_embeds.shape[-2], inputs_embeds.shape[-1])
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
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        output = (reshaped_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output
