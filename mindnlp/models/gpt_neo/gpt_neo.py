# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
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
""" MindNLP GPT Neo model."""

import os
from typing import Union
import mindspore
import numpy as np
from mindspore import ops, nn, Parameter, Tensor, dtype_to_nptype
from mindspore.common.initializer import initializer, Normal
from mindnlp.models.utils import logging
from mindnlp.models.utils.activations import ACT2FN
from mindnlp.models.gpt_neo.gpt_neo_config import GPTNeoConfig
from mindnlp.abc.backbones.pretrained import PretrainedModel

logger = logging.get_logger(__name__)


class GPTNeoSelfAttention(nn.Cell):
    """
    GPTNeo SelfAttention.
    """

    def __init__(self, config, attention_type):
        super().__init__()

        max_positions = config.max_position_embeddings
        bias = ops.tril(ops.ones((max_positions, max_positions), dtype=mindspore.bool_)).view(
            1, 1, max_positions, max_positions
        )

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = ops.bitwise_xor(bias, ops.tril(
                bias, -config.window_size)).astype(mindspore.bool_)

        self.bias = Parameter(bias, requires_grad=False)
        self.masked_bias = Parameter(Tensor(-1e9), requires_grad=False)

        self.attn_dropout = nn.Dropout(p=float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(p=float(config.resid_dropout))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        # (batch, head, seq_length, head_features)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3)
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.astype(mindspore.float32)
        key = key.astype(mindspore.float32)

        attn_weights = ops.matmul(query, key.swapaxes(-1, -2))

        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.bias[:, :, key_length -
                                query_length: key_length, :key_length]
        mask_value = Tensor(np.finfo(dtype_to_nptype(attn_weights.dtype)).min)
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        mask_value = Tensor(mask_value, dtype=attn_weights.dtype)
        attn_weights = ops.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTNeoAttention(nn.Cell):
    """
    GPTNEO Attention.
    """

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = GPTNeoSelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class GPTNeoMLP(nn.Cell):
    """
    GPTNeo MLP.
    """

    # in MLP: intermediate_size= 4 * hidden_size
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Dense(embed_dim, intermediate_size)
        self.c_proj = nn.Dense(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(p=float(config.resid_dropout))

    def construct(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTNeoBlock(nn.Cell):
    """
    GPTNeo Block.
    """

    def __init__(self, config, layer_id):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(
            (hidden_size,), epsilon=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(
            (hidden_size,), epsilon=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(inner_dim, config)

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # hidden_states, present, (attentions, cross_attentions)
        return outputs


class GPTNeoPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoBlock"]

    def init_model_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Dense,)):
            module.weight.set_data(initializer(Normal(
                sigma=self.config.initializer_range, mean=0.0)), module.weight.shape, module.weight.dtype)
            if module.bias is not None:
                module.bias.set_data(initializer('zeros'),
                                     module.bias.shape, module.bias.dtype)
        elif isinstance(module, nn.Embedding):
            module.weight.set_data(initializer(Normal(
                sigma=self.config.initializer_range, mean=0.0)), module.weight.shape, module.weight.dtype)
            if module.padding_idx is not None:
                zeroslike = ops.ZerosLike()
                module.weight.data[module.padding_idx] = zeroslike(
                    module.weight.data[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            module.bias.set_data(initializer('zeros'),
                                 module.bias.shape, module.bias.dtype)
            module.weight.data = ops.fill(
                module.weight.data.dtype, module.weight.data.shape, 1.0)


    def post_init(self):
        pass

    def get_input_embeddings(self) -> "nn.Cell":
        """
        Returns the model's input embeddings.
        """

    def set_input_embeddings(self, value: "nn.Cell"):
        """
        Set model's input embeddings.
        """

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        resize the model position embeddings if necessary
        """

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """

    def save(self, save_dir: Union[str, os.PathLike]):
        "save pretrain model"

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTNeoModel):
            module.gradient_checkpointing = value


class GPTNeoModel(GPTNeoPreTrainedModel):
    """
    GPTNeo Model
    """
