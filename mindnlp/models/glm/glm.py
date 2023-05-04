# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""
GLM Model
"""

import math
import random
from typing import Optional

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import numpy as mnp
from mindspore import Tensor
from mindspore.nn import Dense, Dropout, LayerNorm
from mindspore.common import initializer

from .glm_config import GLMConfig


def ensure_divisibility(numerator, denominator):
    """
    ensure_divisibility
    """
    assert numerator % denominator == 0


def divide(numerator, denominator):
    """
    divide
    """
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions):
    """
    split tensor along last dim
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.shape[last_dim], num_partitions)
    # Split.
    tensor_list = ops.split(tensor, split_size_or_sections=last_dim_size, axis=-1)

    return tensor_list


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    init_method = initializer.Normal(sigma=std, mean=0.0)
    return init_method

class PromptSpell(nn.Cell):
    """
    PromptSpell modoel
    """
    def __init__(self, spell_length, hidden_size, spell_func):
        super().__init__()
        self.spell_length = spell_length
        self.hidden_size = hidden_size
        self.spell_embeddings = nn.Embedding(self.spell_length, self.hidden_size)
        self.spell_func = spell_func

        if self.spell_func == "lstm":
            self.lstm_head = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )

            self.mlp_head = nn.SequentialCell(
                Dense(2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                Dense(self.hidden_size, self.hidden_size),
            )
        elif self.spell_func == "mlp":
            self.mlp_head = nn.SequentialCell(
                Dense(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                Dense(self.hidden_size, self.hidden_size),
            )
        elif self.spell_func != "none":
            raise NotImplementedError("Prompt function " + self.spell_func)

    def init_embedding(self, word_embeddings=None, task_tokens=None):
        """
        init_embedding method
        """
        num_words = 5000
        for i in range(self.spell_length):
            rand_token = random.randrange(num_words)
            if task_tokens is None:
                target_embedding = word_embeddings[rand_token]
            else:
                word_embedding = word_embeddings[rand_token]
                task_token = random.choice(task_tokens)
                task_embedding = word_embeddings[task_token]
                ratio = random.random()
                target_embedding = word_embedding * ratio + task_embedding * (1 - ratio)
            self.spell_embeddings.embedding_table[i] = target_embedding

    def construct(self):
        prompt_embeds = self.spell_embeddings.embedding_table.unsqueeze(0)
        if self.spell_func == "lstm":
            prompt_embeds = self.lstm_head(prompt_embeds)[0]
        if self.spell_func in ("lstm", "mlp"):
            prompt_embeds = self.mlp_head(prompt_embeds)
        return prompt_embeds


class PositionalEmbedding(nn.Cell):
    """
    Positional Embedding
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (mnp.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.inv_freq = inv_freq

    def construct(self, pos_seq, bsz=None):
        sinusoid_inp = ops.ger(pos_seq, self.inv_freq)
        pos_emb = ops.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], axis=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        return pos_emb[None, :, :]


class GlmMLP(nn.Cell):
    """
    GlmMLP
    """

    def __init__(
        self,
        config: GLMConfig,
        init_method=initializer.Normal(sigma=0.02),
        output_layer_init_method=None,
    ):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Project to 4h.
        self.dense_h_to_4h = Dense(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size * 4,
            weight_init=init_method,
        )
        # Project back to h.
        self.dense_4h_to_h = Dense(
            in_channels=config.hidden_size * 4,
            out_channels=config.hidden_size,
            weight_init=init_method,
        )

        self.dropout = Dropout(p=config.output_dropout_prob)

    def construct(self, hidden_states):
        # [b, s, 4hp]
        intermediate = self.dense_h_to_4h(hidden_states)

        intermediate = ops.gelu(intermediate, approximate="tanh")

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate)
        output = self.dropout(output)
        return output


class GLMSelfAttention(nn.Cell):
    """
    GLMSelfAttention
    """

    def __init__(
        self,
        config: GLMConfig,
        init_method=initializer.Normal(sigma=0.02),
        output_layer_init_method=None,
        performer=False,
    ):
        super().__init__()
        self.performer = performer
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        world_size = 1
        self.hidden_size_per_partition = divide(config.hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(
            config.hidden_size, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            config.num_attention_heads, world_size
        )
        self.relative_encoding = config.relative_encoding
        self.attention_scale = config.attention_scale
        # Strided linear layer.
        self.query_key_value = Dense(
            in_channels=config.hidden_size,
            out_channels=3 * config.hidden_size,
            weight_init=init_method,
        )

        if config.relative_encoding:
            config.relative_encoding = Dense(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                weight_init=init_method,
            )

        self.attention_dropout = Dropout(p=config.attention_dropout_prob)

        # Output.
        self.dense = Dense(
            config.hidden_size, config.hidden_size, weight_init=init_method
        )

        self.output_dropout = Dropout(p=config.output_dropout_prob)

    def _transpose_for_scores(self, tensor: Tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.shape[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _rel_shift(input_x: Tensor, zero_triu=False):
        zero_pad = ops.zeros((*input_x.shape[:-2], input_x.shape[-2], 1), dtype=input_x.dtype)
        x_padded = ops.cat([zero_pad, input_x], axis=-1)

        x_padded = x_padded.view(*input_x.shape[:-2], input_x.shape[-1] + 1, input_x.shape[-2])

        input_x = x_padded[:, :, 1:].view(input_x.shape)

        if zero_triu:
            ones = ops.ones((input_x.shape[0], input_x.shape[1]))
            input_x = input_x * mnp.tril(ones, input_x.shape[1] - input_x.shape[0])[:, :, None, None]

        return input_x

    def construct(
        self,
        hidden_states,
        ltor_mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tensor] = None,
        r_w_bias: Optional[Tensor] = None,
        r_r_bias: Optional[Tensor] = None,
        mem=None,
    ):
        query_length = hidden_states.shape[1]
        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (
                mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer,
            ) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = ops.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (
                mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer,
            ) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        # Reshape and transpose [b, n, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        if self.relative_encoding:
            relative_layer = self.relative(position_embeddings)
            # 1 (bsz) x n_head x klen x d_head
            relative_layer = self._transpose_for_scores(relative_layer)

            rw_head_q = query_layer + ops.unsqueeze(r_w_bias, 1)
            ac_score = ops.matmul(rw_head_q, key_layer.transpose(0, 1, 3, 2))
            rr_head_q = query_layer + ops.unsqueeze(r_r_bias, 1)
            bd_score = ops.matmul(rr_head_q, relative_layer)
            bd_score = self._rel_shift(bd_score)

            # Raw attention scores. [b, n, s, s]
            attention_scores = ac_score + bd_score
            attention_scores = attention_scores / math.sqrt(
                self.hidden_size_per_attention_head
            )
        else:
            if self.attention_scale > 1.0:
                attention_scores = ops.matmul(
                    query_layer / math.sqrt(self.attention_scale),
                    key_layer.transpose(0, 1, 3, 2)
                    / math.sqrt(
                        self.hidden_size_per_attention_head * self.attention_scale
                    ),
                )
            else:
                attention_scores = ops.matmul(
                    query_layer,
                    key_layer.transpose(0, 1, 3, 2)
                    / math.sqrt(self.hidden_size_per_attention_head),
                )

        # Apply the left to right attention mask.
        if ltor_mask is not None:
            attention_scores = ops.mul(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(axis=-1, keepdims=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale

        if ltor_mask is not None:
            attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
        # Attention probabilities. [b, n, s, s]
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, n, s, hn]
        context_layer = ops.matmul(attention_probs, value_layer)
        # [b, s, n, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (
            self.hidden_size_per_partition,
        )
        # [b, s, h]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class GLMCrossAttention(nn.Cell):
    """
    GLM cross-attention layer
    """

    def __init__(
        self,
        config: GLMConfig,
        init_method=initializer.Normal(sigma=0.02),
        output_layer_init_method=None,
    ):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        world_size = 1
        self.hidden_size_per_partition = divide(config.hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(
            config.hidden_size, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            config.num_attention_heads, world_size
        )

        # Strided linear layer.
        self.query = Dense(
            config.hidden_size, config.hidden_size, weight_init=init_method
        )

        self.key_value = Dense(
            config.hidden_size, 2 * config.hidden_size, weight_init=init_method
        )

        # Dropout.
        self.attention_dropout = Dropout(p=config.attention_dropout_prob)

        # Output.
        self.dense = Dense(
            config.hidden_size, config.hidden_size, weight_init=init_method
        )

        self.output_dropout = Dropout(p=config.output_dropout_prob)

    def _transpose_for_scores(self, tensor: Tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.shape[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def construct(self, hidden_states, encoder_states, cross_mask=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, h]
        mixed_query_layer = self.query(hidden_states)
        mixed_x_layer = self.key_value(encoder_states)
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(
            mixed_x_layer, 2
        )

        # Reshape and transpose [b, n, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Raw attention scores. [b, n, s, s]
        attention_scores = ops.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head
        )
        if cross_mask is not None:
            # Apply the left to right attention mask.
            attention_scores = ops.mul(attention_scores, cross_mask) - 10000.0 * (
                1.0 - cross_mask
            )

        # Attention probabilities. [b, n, s, s]
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, n, s, hn]
        context_layer = ops.matmul(attention_probs, value_layer)
        # [b, s, n, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (
            self.hidden_size_per_partition,
        )
        # [b, s, h]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class GLMTransformerLayer(nn.Cell):
    """
    GLM TransformerLayer
    """

    def __init__(
        self,
        config: GLMConfig,
        init_method=initializer.Normal(sigma=0.02),
        output_layer_init_method=None,
        performer=False,
    ):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            [config.max_sequence_length, config.hidden_size],
            begin_norm_axis=1,
            begin_params_axis=1,
            epsilon=config.layernorm_epsilon,
        )

        # Self attention.
        self.attention = GLMSelfAttention(
            config=config,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            performer=performer,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            [config.max_sequence_length, config.hidden_size],
            begin_norm_axis=1,
            begin_params_axis=1,
            epsilon=config.layernorm_epsilon,
        )

        # MLP
        self.mlp = GlmMLP(
            config=config,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )

    def construct(
        self,
        hidden_states,
        ltor_mask,
        position_embeddings=None,
        r_w_bias=None,
        r_r_bias=None,
        mem=None,
    ):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        if mem is not None:
            mem = self.input_layernorm(mem)
        # Self attention.
        attention_output = self.attention(
            layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem
        )
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


class GLMDecoderLayer(nn.Cell):
    """
    GLM Decoder Layer
    """

    def __init__(
        self,
        config: GLMConfig,
        init_method=initializer.Normal(sigma=0.02),
        output_layer_init_method=None,
    ):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            [config.max_sequence_length, config.hidden_size],
            begin_norm_axis=1,
            begin_params_axis=1,
            epsilon=config.layernorm_epsilon,
        )

        # Self attention.
        self.self_attention = GLMSelfAttention(
            config=config,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )

        # Layernorm after the self attention.
        self.post_self_layernorm = LayerNorm(
            [config.max_sequence_length, config.hidden_size],
            begin_norm_axis=1,
            begin_params_axis=1,
            epsilon=config.layernorm_epsilon,
        )

        # Cross attention
        self.cross_attention = GLMCrossAttention(
            config,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )

        # Layernorm after the cross attention.
        self.post_attention_layernorm = LayerNorm(
            [config.max_sequence_length, config.hidden_size],
            begin_norm_axis=1,
            begin_params_axis=1,
            epsilon=config.layernorm_epsilon,
        )

        # MLP
        self.mlp = GlmMLP(config, init_method, output_layer_init_method)

    def construct(
        self, hidden_states, encoder_states, ltor_mask, cross_mask=None, mem=None
    ):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]
        mem = self.input_layernorm(mem) if mem is not None else None
        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        self_attention_output = self.self_attention(layernorm_output, ltor_mask)
        # Residual connection.
        self_layernorm_input = hidden_states + self_attention_output
        # Layer norm post the self attention.
        self_layernorm_output = self.post_self_layernorm(self_layernorm_input)
        # Cross attention
        attention_output = self.cross_attention(
            self_layernorm_output, encoder_states, cross_mask
        )
        # Residual connection
        layernorm_input = self_layernorm_input + attention_output
        # Layer norm post the cross attention
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


class GLMTransformer(nn.Cell):
    """
    GLM transformer.
    """

    def __init__(
        self,
        config: GLMConfig,
        use_scaled_init_for_output_weights=True,
        performer=False,
        use_decoder_layer=False,
    ):
        super().__init__()
        assert not (
            performer and config.relative_encoding
        )
        self.performer = performer
        self.init_method_std = config.init_method_std
        self.use_scaled_init_for_output_weights = use_scaled_init_for_output_weights
        self.relative_encoding = config.relative_encoding
        self.block_position_encoding = config.block_position_encoding
        self.max_memory_length = config.max_memory_length
        self.use_decoder_layer = use_decoder_layer

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(
                config.init_method_std, config.num_layers
            )

        self.embedding_dropout = nn.Dropout(p=config.embedding_dropout_prob)

        if config.relative_encoding:
            # Relative position embedding
            self.position_embeddings = PositionalEmbedding(config.hidden_size)
            # Per attention head and per partition values.
            world_size = 1
            self.hidden_size_per_attention_head = divide(
                config.hidden_size, config.num_attention_heads
            )
            self.num_attention_heads_per_partition = divide(
                config.num_attention_heads, world_size
            )

            zeroslike = ops.ZerosLike()
            self.r_w_bias = mindspore.Parameter(
                zeroslike(
                    Tensor(
                        self.num_attention_heads_per_partition,
                        self.hidden_size_per_attention_head,
                    )
                )
            )

            self.r_r_bias = mindspore.Parameter(
                zeroslike(
                    Tensor(
                        self.num_attention_heads_per_partition,
                        self.hidden_size_per_attention_head,
                    )
                )
            )
        else:
            # Position embedding (serial).
            if config.block_position_encoding:
                self.position_embeddings = nn.Embedding(
                    config.max_sequence_length + 1,
                    config.hidden_size,
                    embedding_table=initializer.Normal(sigma=config.init_method_std),
                )
                self.block_position_embeddings = nn.Embedding(
                    config.max_sequence_length + 1,
                    config.hidden_size,
                    embedding_table=initializer.Normal(sigma=config.init_method_std),
                )
            else:
                self.position_embeddings = nn.Embedding(
                    config.max_sequence_length,
                    config.hidden_size,
                    embedding_table=initializer.Normal(sigma=config.init_method_std),
                )

        def get_layer():
            if use_decoder_layer:
                return GLMDecoderLayer(
                    config,
                    init_method=initializer.Normal(sigma=config.init_method_std),
                    output_layer_init_method=output_layer_init_method,
                )

            return GLMTransformerLayer(
                config,
                init_method=initializer.Normal(sigma=config.init_method_std),
                output_layer_init_method=output_layer_init_method,
                performer=performer,
            )

        # Transformer layers.
        self.layers = nn.SequentialCell([get_layer() for _ in range(config.num_layers)])
        # Final layer norm before output.

        self.final_layernorm = LayerNorm(
            [config.max_sequence_length, config.hidden_size],
            begin_norm_axis=1,
            begin_params_axis=1,
            epsilon=config.layernorm_epsilon,
        )

    def construct(
        self,
        hidden_states,
        position_ids,
        attention_mask,
        memory_states=None,
        encoder_states=None,
        return_memory=False,
        detach_memory=True,
    ):
        batch_size, query_length = hidden_states.shape[:2]
        if memory_states is not None:
            memory_length = memory_states[0].shape[1]
        else:
            memory_length = 0
        key_length = query_length + memory_length
        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = ops.numel(attention_mask) == 1
        is_sep = is_scalar or ops.numel(attention_mask) == batch_size

        if self.performer:
            assert (
                is_scalar
            ), "attention_mask should be a scalar to indicate the seperation position."
            assert memory_length == 0, "Do not support transformer-xl."
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                oneslike = ops.OnesLike()
                mask_matrix = oneslike(hidden_states).shape(1, seq_length, seq_length)
                mask_matrix = mnp.tril(mask_matrix)
                if is_scalar:
                    mask_matrix[0, :, :sep] = 1
                else:
                    mask_matrix = mask_matrix.expand(batch_size, -1, -1)
                    ids = mnp.arange(seq_length, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    mask_matrix = ops.masked_fill(mask_matrix, ops.unsqueeze(mask, 1).expand_as(mask_matrix), 1)
                if memory_length > 0:
                    mask_matrix = mask_matrix.expand(batch_size, -1, -1)
                    mask_matrix = ops.cat(
                        (
                            oneslike(hidden_states).shape(
                                batch_size, seq_length, memory_length
                            ),
                            mask_matrix,
                        ),
                        axis=2,
                    )
                mask_matrix = ops.unsqueeze(mask_matrix, 1)
                return mask_matrix

            if not self.performer:
                attention_mask = build_mask_matrix(
                    query_length, sep, memory_length=memory_length
                )
        else:
            attention_mask = attention_mask.astype(hidden_states.dtype)
            attention_mask = attention_mask[:, :, :, -query_length - memory_length :]

        if self.relative_encoding:
            position_sequence = mnp.arange(
                key_length - 1,
                -1,
                -1.0,
                dtype=hidden_states.dtype,
            )

            position_embeddings = self.position_embeddings(position_sequence)
            # Apply dropout
            position_embeddings = self.embedding_dropout(position_embeddings)
        else:
            if self.block_position_encoding:
                position_ids, block_position_ids = (
                    position_ids[:, 0],
                    position_ids[:, 1],
                )
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
            if self.block_position_encoding:
                block_position_embeddings = self.block_position_embeddings(
                    block_position_ids
                )
                hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            if detach_memory:
                return ops.stop_gradient(_hidden_states)
            return _hidden_states

        if self.max_memory_length > 0 or return_memory:
            mem_layers = [check_detach(hidden_states)]
        else:
            mem_layers = []

        for i, layer in enumerate(self.layers):
            args = (
                [hidden_states, attention_mask]
                if not self.use_decoder_layer
                else [hidden_states, encoder_states, attention_mask]
            )
            if self.relative_encoding:
                args += [position_embeddings, self.r_w_bias, self.r_r_bias]
            mem_i = memory_states[i] if memory_states else None
            hidden_states = layer(*args, mem=mem_i)
            if self.max_memory_length > 0 or return_memory:
                mem_layers.append(check_detach(hidden_states))

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0 or return_memory:
            mem_layers = self.update_mems(
                mem_layers, memory_states, return_memory=return_memory
            )

        return (output, mem_layers)

    def update_mems(self, hiddens, mems, return_memory=False):
        """
        update mems
        """
        memory_length = mems[0].shape[1] if mems else 0
        query_length = hiddens[0].shape[1]
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []

        for i, layers in enumerate(hiddens):
            if new_memory_length <= query_length:
                new_mems.append(layers[:, -new_memory_length:])
            else:
                new_mems.append(
                    ops.cat(
                        (mems[i][:, -new_memory_length + query_length :], layers),
                        axis=1,
                    )
                )
        return new_mems


class GLMModel(nn.Cell):
    """
    Glm Model
    """
    def __init__(self, config: GLMConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        # Word embeddings
        self.word_embedding = nn.Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            embedding_table=initializer.Normal(sigma=0.02),
        )

        # Transformer
        self.transformer = GLMTransformer(config)

        if config.spell_length is not None:
            self.prompt_spell = PromptSpell(
                config.spell_length, self.hidden_size, config.spell_func
            )

    def construct(
        self,
        input_ids,
        position_ids,
        attention_mask,
        mems,
        return_memory=False,
        detach_memory=True,
        prompt_pos=None,
    ):
        batch_size = input_ids.shape[0]
        words_embeddings = self.word_embedding(input_ids)
        embeddings = words_embeddings
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = mnp.arange(batch_size).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds

        # Transformer.
        transformer_output = self.transformer(
            embeddings,
            position_ids,
            attention_mask,
            mems,
            return_memory=return_memory,
            detach_memory=detach_memory,
        )
        logits, hidden_layers = transformer_output
        outputs = hidden_layers

        return (logits, *outputs)
