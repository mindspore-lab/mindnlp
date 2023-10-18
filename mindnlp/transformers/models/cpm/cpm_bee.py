# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
# pylint: disable=C0103
""" MindNLP CPM Bee"""
import math
from typing import Union, Optional, Tuple
import numpy as np

import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from .cpm_ant import CpmAntPreTrainedModel, CpmAntEncoder
from .cpm_bee_config import CpmBeeConfig

class CpmBeeRotaryEmbedding(nn.Cell):
    """Cpm Bee RotaryEmbedding"""
    def __init__(
        self,
        dim,
        base=10000,
        distance_scale: Union[int, float] = 1,
        dtype: mindspore.tensor_type = mindspore.float16
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
        self.distance_scale = distance_scale
        self.inv_freq = Tensor(inv_freq, dtype)
        self.dtype = dtype

    def construct(self, x: Tensor, x_pos: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(..., dim)``): Inputs.
            x_pos (:obj:`Tensor` of shape ``(...)``): Positions of inputs.
        """
        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None].to(self.dtype) * self.inv_freq[None, :]  # (..., dim/2)

        # the same implementation as sat
        emb = ops.cat((freqs, freqs), axis=-1)  # (..., dim)
        emb_cos = emb.cos()  # (..., dim)
        emb_sin = emb.sin()  # (..., dim)

        rotate_x = ops.cat(
            [-x[..., x.size(-1) // 2 :], x[..., : x.size(-1) // 2]], axis=-1
        )  # (..., dim)

        return x * emb_cos + rotate_x * emb_sin

class CpmBeeEmbeddingExt(nn.Cell):
    """Cpm Bee EmbeddingExt"""
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: mindspore.tensor_dtype = mindspore.float16,
        distance_scale: int = 16
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.rotary_emb = CpmBeeRotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=dtype
        )

        self.weight = Parameter(initializer('normal', (vocab_size, embedding_size), dtype), 'weight')

    def construct(self, ids: Tensor, ids_sub: Tensor):
        """
        Args:
            ids (:obj:`Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        ids_shape = ids.shape
        embeds = ops.gather(self.weight, ids.view(-1), 0).view(ids_shape + (self.dim_model,))
        embeds = embeds / ops.sqrt(self.dim_model)
        return self.rotary_emb(embeds, ids_sub)

    def projection(self, x: Tensor, ext_table: Optional[Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size,
        than projection map embed_size back to vocab_size.

        Args:
            x (:obj:`Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        logits = ops.matmul(x / ops.sqrt(self.dim_model), self.weight)
        if ext_table is not None:
            logits_ext = ops.matmul(x, ext_table)
            logits = ops.cat([logits, logits_ext], axis=-1)
        return logits

class CpmBeeBucketPositionBias(nn.Cell):
    """Cpm Bee BucketPositionBias"""
    def __init__(self, config) -> None:
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_buckets = config.position_bias_num_buckets
        self.max_distance = config.position_bias_max_distance
        self.num_segment_bucket = config.position_bias_num_segment_buckets

        self.relative_attention_bias = Parameter(initializer('zeros', (self.num_buckets + self.num_segment_bucket, config.num_attention_heads)),
                                                 'weight')

    def construct(
        self,
        query_pos: Tensor,  # (batch, len_q)
        key_pos: Tensor,  # (batch, len_k)
        rel_buckets: Tensor,  # (batch, len_q, len_k)
    ):
        batch = key_pos.shape[0]
        keylen = key_pos.shape[1]
        querylen = query_pos.shape[1]

        if key_pos.shape[0] != query_pos.shape[0]:
            raise AssertionError(
                f"key_pos.shape[0] should be equal to query_pos.shape[0], but got {key_pos.shape[0]} and {query_pos.shape[0]}!"
            )
        assert (
            rel_buckets.shape[0] == batch
            and rel_buckets.shape[1] == querylen
            and rel_buckets.shape[2] == keylen
        )

        relative_position_bucket = rel_buckets - 1 + self.num_buckets  # 与相对位置编码区间不重叠

        # b*q*k
        inner_segment_bucket = self._position_bucket(
            key_pos[..., None, :] - query_pos[..., :, None],
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_position_bucket = ops.where(
            rel_buckets == 0,
            inner_segment_bucket,
            relative_position_bucket,
        )
        # (batch, len_q, len_k)
        relative_position_bucket = ops.stop_gradient(relative_position_bucket)

        # (batch, len_q, len_k, num_heads)
        relative_position_bucket_shape = relative_position_bucket.shape
        embeds = ops.gather(
            self.relative_attention_bias, relative_position_bucket.reshape(-1), 0
        )
        embeds = embeds.reshape(relative_position_bucket_shape + (self.num_heads,))
                # (batch, num_heads, len_q, len_k)
        embeds = embeds.transpose(0, 3, 1, 2)
        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        # always bidirectional in CPMAnt
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(mindspore.int32) * num_buckets
        relative_position = ops.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.int32)
        relative_postion_if_large = ops.minimum(
            relative_postion_if_large,
            ops.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += ops.where(
            is_small, relative_position.to(mindspore.int32), relative_postion_if_large
        )
        return relative_buckets

class CpmBeePreTrainedModel(CpmAntPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # config_class = CpmAntConfig
    base_model_prefix = "cpmbee"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (CpmBeeEmbeddingExt, CpmBeeBucketPositionBias)):
            cell.relative_attention_bias.set_data(
                initializer(
                    Normal(self.config.init_std),
                    cell.relative_attention_bias.shape,
                    cell.relative_attention_bias.dtype,
                )
            )
        super()._init_weights(cell)

class CpmBeeModel(CpmAntPreTrainedModel):
    """Cpm Ant Model"""
    def __init__(self, config: CpmBeeConfig):
        super().__init__(config)
        self.encoder = CpmAntEncoder(config)
        self.input_embedding = CpmBeeEmbeddingExt(
            config.vocab_size,
            config.hidden_size,
        )
        self.position_bias = CpmBeeBucketPositionBias(config)
        self.vocab_size = config.vocab_size

        self.post_init()

    def get_input_embeddings(self):
        return self.input_embedding

    def set_input_embeddings(self, new_embeddings):
        self.input_embedding = new_embeddings

    def _prepare_attention_mask(self, input_ids, span, context, length, sample_ids):
        batch = input_ids.shape[0]
        seqlen = input_ids.shape[1]
        # directional mask
        directional_mask_2d = ops.arange(seqlen) <= ops.arange(seqlen).view(-1, 1)
        # sample mask
        sample_mask_2d = (sample_ids[:, :, None] == 0) | (
            sample_ids[:, :, None] == sample_ids[:, None, :]
        )
        # context mask
        attention_mask = context[:, None, :] | (
            context[:, :, None].logical_not() & directional_mask_2d.view((1, seqlen, seqlen))
        )
        # span mask
        attention_mask = (
            attention_mask & sample_mask_2d & (span[:, None, :] == span[:, :, None])
        )        # length mask
        mask_1d = (
            ops.arange(seqlen)[None, :].tile((batch, 1)) < length[:, None]
        )

        attention_mask = (
            mask_1d.view(batch, seqlen, 1)
            & mask_1d.view(batch, 1, seqlen)
            & attention_mask
        )
        return attention_mask

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # add prompts ahead
        if input_ids.dtype != mindspore.int32:
            input_ids = input_ids.to(mindspore.int32)
        dtype = input_ids.dtype
        segment = ops.where(input_ids != 0, Tensor(2), Tensor(0)).to(dtype=dtype)
        length = (segment != 0).sum(-1).to(dtype=dtype)

        batch, seq_length = input_ids.shape
        context = ops.full((batch, seq_length), 1, dtype=dtype)
        position = ops.arange(seq_length, dtype=dtype).tile((batch, 1))
        span = ops.full((batch, seq_length), 0, dtype=dtype)
        sample_ids = ops.full((batch, seq_length), 0, dtype=dtype)
        input_ids_sub = ops.full((batch, seq_length), 0, dtype=dtype)
        num_segments = ops.full((batch, seq_length), 1, dtype=dtype)
        segment_rel = ops.full((batch, seq_length), 0, dtype=dtype)
        segment_rel_offset = ops.full((batch, seq_length), 0, dtype=dtype)

        # calc segment bucket
        segment_rel_2d = ops.masked_fill(
            segment[:, :, None] * num_segments[:, :, None]
            + segment[:, None, :]
            + segment_rel_offset[:, :, None],
            ~(
                (sample_ids[:, :, None] == sample_ids[:, None, :])
                & (span[:, None, :] == span[:, :, None])
            ),  # not in the same span or sample
            0,  # avoid torch.gather overflow
        ).view((batch, seq_length * seq_length))

        segment_bucket = ops.gather_elements(
            input=segment_rel,
            dim=1,
            index=segment_rel_2d.long(),
        ).view((batch, seq_length, seq_length))

        segment_bucket = ops.masked_fill(
            segment_bucket,
            ~(
                (sample_ids[:, :, None] == sample_ids[:, None, :])
                & (span[:, None, :] == span[:, :, None])
            ),  # not in the same span or sample
            1,  # bucket is used for in-context samples
        )


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.encoder.num_layers)
            hidden_states = self.input_embedding(input_ids, input_ids_sub)
        else:
            past_length = past_key_values[0][0].size(-2)

        attention_mask = self._prepare_attention_mask(input_ids, span, context, length, sample_ids)
        position_bias = self.position_bias(position, position, segment_bucket)

        attention_mask = attention_mask[:, past_length:, :]
        position_bias = position_bias[:, :, past_length:, :]
        hidden_states = hidden_states[:, past_length:, :]

        (
            hidden_states,
            present_key_values,
            all_hidden_states,
            all_attentions,
        ) = self.encoder(
            hidden_states,
            attention_mask,
            position_bias,
            output_attentions,
            output_hidden_states,
            past_key_values,
            use_cache,
        )

        if past_length == 0:
            hidden_states = hidden_states[:, self.prompt_length :, :]
            # drop the prompt
            if all_attentions is not None:
                new_attentions = ()
                for attention in all_attentions:
                    new_attentions += (
                        attention[:, :, self.prompt_length :, self.prompt_length :],
                    )
                all_attentions = new_attentions
            if all_hidden_states is not None:
                new_hidden_states = ()
                for hidden_state in all_hidden_states:
                    new_hidden_states += (hidden_state[:, self.prompt_length :, :],)
                all_hidden_states = new_hidden_states

        return tuple(
            v
            for v in [
                hidden_states,
                present_key_values,
                all_hidden_states,
                all_attentions,
            ]
            if v is not None
        )
