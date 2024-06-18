# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================
"""MindSpore SegGpt model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import ModelOutput
from ....utils import logging
from .configuration_seggpt import SegGptConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SegGptConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "BAAI/seggpt-vit-large"
_EXPECTED_OUTPUT_SHAPE = [3, 896, 448]


@dataclass
class SegGptEncoderOutput(ModelOutput):
    """
    Output type of [`SegGptEncoderOutput`].
    Args:
        last_hidden_state (`ms.Tensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`Tuple[ms.Tensor]`, `optional`, returned when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[ms.Tensor]`, `optional`, returned when `config.output_attentions=True`):
            Tuple of *ms.Tensor* (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
        intermediate_hidden_states (`Tuple[ms.Tensor]`, `optional`, returned when `config.intermediate_hidden_state_indices` is set):
            Tuple of `ms.Tensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`.
            Each element in the Tuple corresponds to the output of the layer specified in `config.intermediate_hidden_state_indices`.
            Additionaly, each feature passes through a LayerNorm.
    """

    last_hidden_state: ms.Tensor
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None
    intermediate_hidden_states: Optional[Tuple[ms.Tensor]] = None


@dataclass
class SegGptImageSegmentationOutput(ModelOutput):
    """
    Output type of [`SegGptImageSegmentationOutput`].

    Args:
        loss (`ms.Tensor`, `optional`, returned when `labels` is provided):
            The loss value.
        pred_masks (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The predicted masks.
        hidden_states (`Tuple[ms.Tensor]`, `optional`, returned when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[ms.Tensor]`, `optional`, returned when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
    """

    loss: Optional[ms.Tensor] = None
    pred_masks: Optional[ms.Tensor] = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


# Copied from transformers.models.sam.modeling_sam.SamPatchEmbeddings with Sam->SegGpt
class SegGptPatchEmbeddings(nn.Cell):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(
            image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(
            patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * \
            (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, has_bias=True, pad_mode='pad', padding=0)

    def construct(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SegGptEmbeddings(nn.Cell):
    """
    Construct the embeddings from patch, position embeddings for input and prompt.
    """

    def __init__(self, config: SegGptConfig) -> None:
        super().__init__()

        self.mask_token = ms.Parameter(ops.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_input = ms.Parameter(
            ops.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_prompt = ms.Parameter(
            ops.zeros(1, 1, 1, config.hidden_size))
        # token for seg types
        self.type_token_semantic = ms.Parameter(
            ops.zeros(1, 1, 1, config.hidden_size))
        self.type_token_instance = ms.Parameter(
            ops.zeros(1, 1, 1, config.hidden_size))

        self.patch_embeddings = SegGptPatchEmbeddings(config)

        num_positions = (config.pretrain_image_size //
                         config.patch_size) ** 2 + 1
        self.position_embeddings = ms.Parameter(
            ops.randn(1, num_positions, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, height: int, width: int) -> ms.Tensor:
        patch_pos_embed = self.position_embeddings[:, 1:]
        num_patches = patch_pos_embed.shape[1]
        pretrain_patch_size = int(math.sqrt(num_patches))

        if pretrain_patch_size != height or pretrain_patch_size != width:
            patch_pos_embed = ops.interpolate(
                patch_pos_embed.reshape(
                    1, pretrain_patch_size, pretrain_patch_size, -1).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )

            return patch_pos_embed.permute(0, 2, 3, 1)
        else:
            return patch_pos_embed.reshape(1, height, width, -1)

    def construct(
        self,
        pixel_values: ms.Tensor,
        prompt_pixel_values: ms.Tensor,
        bool_masked_pos: Optional[ms.Tensor] = None,
        embedding_type: Optional[str] = None,
    ) -> ms.Tensor:
        input_embeddings = self.patch_embeddings(pixel_values)
        prompt_embeddings = self.patch_embeddings(prompt_pixel_values)

        batch_size, patch_height, patch_width, _ = input_embeddings.shape

        mask_token = self.mask_token.expand(
            batch_size, patch_height, patch_width, -1)
        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(
            mask_token).reshape(-1, patch_height, patch_width, 1)
        prompt_embeddings = prompt_embeddings * (1 - w) + mask_token * w

        embedding_type = embedding_type if embedding_type is not None else "instance"

        # add positional encoding to each token
        pos_embed = self.interpolate_pos_encoding(patch_height, patch_width)

        # add segment token
        input_embeddings = input_embeddings + self.segment_token_input
        prompt_embeddings = prompt_embeddings + self.segment_token_prompt

        # add position embedding skipping CLS
        input_embeddings = input_embeddings + pos_embed
        prompt_embeddings = prompt_embeddings + pos_embed

        # add type embedding to each token
        if embedding_type == "semantic":
            type_embedding = self.type_token_semantic
        elif embedding_type == "instance":
            type_embedding = self.type_token_instance
        else:
            raise ValueError(
                f"Embedding type should be either 'semantic' or 'instance', but got {embedding_type}")

        input_embeddings = input_embeddings + type_embedding
        prompt_embeddings = prompt_embeddings + type_embedding

        embeddings = ops.cat((input_embeddings, prompt_embeddings), axis=0)

        return embeddings


class SegGptAttention(nn.Cell):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        image_size = image_size if isinstance(
            image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(
            patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        input_size = (image_size[0] // config.patch_size,
                      image_size[1] // config.patch_size)
        head_dim = config.hidden_size // config.num_attention_heads

        self.num_attention_heads = config.num_attention_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Dense(config.hidden_size,
                            config.hidden_size * 3, has_bias=config.qkv_bias)
        self.proj = nn.Dense(config.hidden_size, config.hidden_size)

        self.use_relative_position_embeddings = config.use_relative_position_embeddings
        if self.use_relative_position_embeddings:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            self.rel_pos_h = ms.Parameter(
                ops.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = ms.Parameter(
                ops.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: ms.Tensor) -> ms.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`ms.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = ops.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(
            -1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = ops.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = ops.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + \
            (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: ms.Tensor,
        query: ms.Tensor,
        rel_pos_h: ms.Tensor,
        rel_pos_w: ms.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> ms.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`ms.Tensor`):
                attention map.
            query (`ms.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`ms.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`ms.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`ms.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(
            query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(
            query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(
            batch_size, query_height, query_width, dim)
        rel_h = ops.einsum("bhwc,hkc->bhwk", reshaped_query,
                           relative_position_height)
        rel_w = ops.einsum("bhwc,wkc->bhwk", reshaped_query,
                           relative_position_width)
        attn = attn.reshape(batch_size, query_height,
                            query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height *
                            query_width, key_height * key_width)
        return attn

    def construct(self, hidden_states: ms.Tensor, output_attentions=False) -> ms.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(
            3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.swapaxes(-2, -1)

        if self.use_relative_position_embeddings:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (
                    height, width), (height, width)
            )

        attn_weights = ops.softmax(
            attn_weights, dtype=ms.float32, axis=-1).astype(query.dtype)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                batch_size, self.num_attention_heads, height * width, -1)
            attn_weights = attn_weights_reshaped.view(
                batch_size * self.num_attention_heads, height * width, -1)
        else:
            attn_weights_reshaped = None

        attn_output = (attn_weights @ value).reshape(batch_size,
                                                     self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(
            0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        return (attn_output, attn_weights_reshaped)


# Copied from transformers.models.sam.modeling_sam.SamMLPBlock with SamMLPBlock->SegGptMlp
class SegGptMlp(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Dense(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Dense(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: ms.Tensor, drop_prob: float = 0.0, training: bool = False) -> ms.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + ops.rand(shape, dtype=input.dtype)
    random_tensor = random_tensor.floor()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->SegGpt
class SegGptDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SegGptLayer(nn.Cell):
    def __init__(self, config: SegGptConfig, drop_path_rate: float) -> None:
        super().__init__()
        self.attention = SegGptAttention(config)
        self.mlp = SegGptMlp(config)
        self.drop_path = SegGptDropPath(
            drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        ensemble_cond: int,
        feature_ensemble: bool = False,
        output_attentions: bool = False,
    ) -> Union[Tuple[ms.Tensor, ms.Tensor], Tuple[ms.Tensor]]:
        self_attention_outputs = self.attention(
            # in SegGpt, layernorm is applied before self-attention
            self.layernorm_before(hidden_states),
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        if feature_ensemble and attention_output.shape[0] // 2 >= ensemble_cond:
            prompt, inputs = attention_output.split(
                attention_output.shape[1] // 2, axis=1)
            if ensemble_cond == 2:
                num_prompts = attention_output.shape[0] // 2
                inputs = inputs.reshape(2, num_prompts, -1)
                inputs = inputs.mean(axis=1, keep_dims=True).expand_as(inputs)
                inputs = inputs.reshape(*prompt.shape)
            else:
                inputs = inputs.mean(axis=0, keep_dims=True).expand_as(inputs)
            attention_output = ops.cat([prompt, inputs], axis=1)

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states
        residual = hidden_states

        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.drop_path(hidden_states)

        outputs = (hidden_states,) + outputs

        return outputs


class SegGptEncoder(nn.Cell):
    def __init__(self, config: SegGptConfig) -> None:
        super().__init__()
        self.config = config
        dpr = [x.item() for x in ops.linspace(
            0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.CellList([SegGptLayer(config, dpr[i])
                                  for i in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        feature_ensemble: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, SegGptEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        intermediate_hidden_states = []

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Condition to check if we have the appropriate number of prompts to ensemble
            ensemble_cond = 2 if self.config.merge_index > i else 1

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    ensemble_cond,
                    feature_ensemble,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, ensemble_cond, feature_ensemble, output_attentions)

            hidden_states = layer_outputs[0]

            if i == self.config.merge_index:
                hidden_states = (
                    hidden_states[: hidden_states.shape[0] // 2] +
                    hidden_states[hidden_states.shape[0] // 2:]
                ) * 0.5

            if i in self.config.intermediate_hidden_state_indices:
                intermediate_hidden_states.append(
                    self.layernorm(hidden_states))

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, intermediate_hidden_states]
                if v is not None
            )
        return SegGptEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            intermediate_hidden_states=intermediate_hidden_states,
        )


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->SegGpt
class SegGptLayerNorm(nn.Cell):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = ms.Parameter(ops.ones(normalized_shape))
        self.bias = ms.Parameter(ops.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(
                f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.data_format == "channels_last":
            x = ops.layer_norm(x, self.normalized_shape,
                               self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keep_dims=True)
            s = (x - u).pow(2).mean(1, keep_dims=True)
            x = (x - u) / ops.sqrt(s + self.eps)
            x = x.astype(input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SegGptDecoderHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
            kernel_size=3,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        self.layernorm = SegGptLayerNorm(
            normalized_shape=config.decoder_hidden_size, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.act_fct = ACT2FN[config.hidden_act]
        self.head = nn.Conv2d(config.decoder_hidden_size, 3,
                              kernel_size=1, has_bias=True, pad_mode='pad', padding=0)  # decoder to patch

    def construct(self, hidden_states: ms.Tensor):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.act_fct(hidden_states)
        hidden_states = self.head(hidden_states)

        return hidden_states


class SegGptDecoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.decoder_embed = nn.Dense(
            config.hidden_size * len(config.intermediate_hidden_state_indices),
            config.patch_size**2 * config.decoder_hidden_size,
            has_bias=True,
        )
        self.decoder_pred = SegGptDecoderHead(config)
        self.patch_size = config.patch_size
        self.decoder_hidden_size = config.decoder_hidden_size
        self.config = config

    def _reshape_hidden_states(self, hidden_states: ms.Tensor) -> ms.Tensor:
        batch_size, patch_height, patch_width, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, patch_height, patch_width, self.patch_size, self.patch_size, self.decoder_hidden_size
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        hidden_states = hidden_states.reshape(
            batch_size, -1, patch_height * self.patch_size, patch_width * self.patch_size
        )

        return hidden_states

    def construct(self, hidden_states: ms.Tensor):
        hidden_states = self.decoder_embed(hidden_states)
        hidden_states = self._reshape_hidden_states(hidden_states)
        hidden_states = self.decoder_pred(hidden_states)

        return hidden_states


class SegGptPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SegGptConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SegGptEmbeddings", "SegGptLayer"]

    def _init_weights(self, cell: Union[nn.Dense, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            cell.weight.data.initialize(Normal(std))
            if cell.bias is not None:
                cell.bias.initialize('zeros')

        elif isinstance(cell, nn.LayerNorm):
            cell.bias.initialize('zeros')
            cell.weight.data.fill(1.0)

        elif isinstance(cell, SegGptAttention):
            cell.rel_pos_h.data.initialize(Normal(std))
            cell.rel_pos_w.data.initialize(Normal(std))

        elif isinstance(cell, SegGptEmbeddings):
            cell.position_embeddings.data.initialize(Normal(std))

            cell.mask_token.data.initialize(Normal(std))
            cell.segment_token_input.data.initialize(Normal(std))
            cell.segment_token_prompt.data.initialize(Normal(std))
            cell.type_token_semantic.data.initialize(Normal(std))
            cell.type_token_instance.data.initialize(Normal(std))


class SegGptModel(SegGptPreTrainedModel):
    def __init__(self, config: SegGptConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = SegGptEmbeddings(config)
        self.encoder = SegGptEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> SegGptPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        pixel_values: ms.Tensor,
        prompt_pixel_values: ms.Tensor,
        prompt_masks: ms.Tensor,
        bool_masked_pos: Optional[ms.Tensor] = None,
        feature_ensemble: Optional[bool] = None,
        embedding_type: Optional[str] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegGptEncoderOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`, `optional`):
            Ground truth mask for input images.

        Returns:

        Examples:

        ```python
        >>> from transformers import SegGptImageProcessor, SegGptModel
        >>> from PIL import Image
        >>> import requests

        >>> image_input_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
        >>> image_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1.jpg"
        >>> mask_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1_target.png"

        >>> image_input = Image.open(requests.get(image_input_url, stream=True).raw)
        >>> image_prompt = Image.open(requests.get(image_prompt_url, stream=True).raw)
        >>> mask_prompt = Image.open(requests.get(mask_prompt_url, stream=True).raw).convert("L")

        >>> checkpoint = "BAAI/seggpt-vit-large"
        >>> model = SegGptModel.from_pretrained(checkpoint)
        >>> image_processor = SegGptImageProcessor.from_pretrained(checkpoint)

        >>> inputs = image_processor(images=image_input, prompt_images=image_prompt, prompt_masks=mask_prompt, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> list(outputs.last_hidden_state.shape)
        [1, 56, 28, 1024]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        feature_ensemble = feature_ensemble if feature_ensemble is not None else False

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        pixel_values = pixel_values.astype(expected_dtype)
        prompt_pixel_values = prompt_pixel_values.astype(expected_dtype)

        # Prepare inputs
        pixel_values = ops.cat((prompt_pixel_values, pixel_values), axis=2)
        prompt_pixel_values = (
            ops.cat((prompt_masks, prompt_masks), axis=2)
            if labels is None
            else ops.cat((prompt_masks, labels), axis=2)
        )
        prompt_pixel_values = prompt_pixel_values.astype(expected_dtype)

        if bool_masked_pos is None and labels is not None:
            logger.warning_once(
                "Labels were provided, but bool_masked_pos were not. It will be set to default value. If you're training the model, make sure to provide a bool_masked_pos."
            )

        # We concat on height axis so SegGPT can handle as a single image, hence we need to mask the portion
        # of the mask prompt pixels that will be destinated to the prediction as they don't add any information.
        # This is only the case for inference. In training, the model concat of prompt mask and label is masked
        # and reconstructed together (In-Context Painting).
        if bool_masked_pos is None:
            num_patches = self.embeddings.patch_embeddings.num_patches
            bool_masked_pos = ops.zeros(num_patches, dtype=ms.bool_)
            bool_masked_pos[num_patches // 2:] = 1
            bool_masked_pos = bool_masked_pos.unsqueeze(0)

        embedding_output = self.embeddings(
            pixel_values, prompt_pixel_values, embedding_type=embedding_type, bool_masked_pos=bool_masked_pos
        )

        encoder_outputs = self.encoder(
            embedding_output,
            feature_ensemble=feature_ensemble,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


def patchify(tensor: ms.Tensor, patch_size: int) -> ms.Tensor:
    batch_size, num_channels, height, width = tensor.shape
    patch_height = height // patch_size
    patch_width = width // patch_size

    tensor = tensor.reshape(batch_size, num_channels,
                            patch_height, patch_size, patch_width, patch_size)
    tensor = tensor.permute(0, 2, 4, 3, 5, 1)
    tensor = tensor.reshape(batch_size, patch_height *
                            patch_width, patch_size**2 * 3)

    return tensor


def unpatchify(tensor: ms.Tensor, patch_height: int, patch_width: int) -> ms.Tensor:
    batch_size = tensor.shape[0]
    patch_size = int((tensor.shape[-1] / 3) ** 0.5)
    if patch_height * patch_width != tensor.shape[1]:
        raise ValueError(
            f"Number of patches {tensor.shape[1]} does not match patch height ({patch_height}) and width ({patch_width})."
        )

    tensor = tensor.reshape(batch_size, patch_height,
                            patch_width, patch_size, patch_size, 3)
    tensor = tensor.permute(0, 5, 1, 3, 2, 4)
    tensor = tensor.reshape(batch_size, 3, patch_height *
                            patch_size, patch_width * patch_size)

    return tensor


class SegGptLoss(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.beta = config.beta
        self.patch_size = config.patch_size

    def construct(
        self,
        prompt_masks: ms.Tensor,
        pred_masks: ms.Tensor,
        labels: ms.Tensor,
        bool_masked_pos: ms.Tensor,
    ):
        """Computes the L1 loss between the predicted masks and the ground truth masks.

        Args:
            prompt_masks (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values from mask prompt.

            pred_masks (`ms.Tensor` of shape `(batch_size, num_channels, 2*height, width)`):
                Predicted masks.

            labels (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Ground truth mask for input images.

            bool_masked_pos (`ms.Tensor` of shape `(batch_size, num_patches)`):
                Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:
            `ms.Tensor`: The mean L1 loss between the predicted masks and the ground truth masks.
        """
        ground_truth = ops.cat((prompt_masks, labels), axis=2)

        mask = bool_masked_pos[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        mask = unpatchify(
            mask, ground_truth.shape[2] // self.patch_size, ground_truth.shape[3] // self.patch_size)

        loss = ops.smooth_l1_loss(
            pred_masks, ground_truth, reduction="none", beta=self.beta)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss


class SegGptForImageSegmentation(SegGptPreTrainedModel):
    def __init__(self, config: SegGptConfig):
        super().__init__(config)
        self.config = config

        self.model = SegGptModel(config)
        self.decoder = SegGptDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: ms.Tensor,
        prompt_pixel_values: ms.Tensor,
        prompt_masks: ms.Tensor,
        bool_masked_pos: Optional[ms.Tensor] = None,
        feature_ensemble: Optional[bool] = None,
        embedding_type: Optional[str] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegGptImageSegmentationOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`, `optional`):
            Ground truth mask for input images.

        Returns:

        Examples:

        ```python
        >>> from transformers import SegGptImageProcessor, SegGptForImageSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> image_input_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
        >>> image_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1.jpg"
        >>> mask_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1_target.png"

        >>> image_input = Image.open(requests.get(image_input_url, stream=True).raw)
        >>> image_prompt = Image.open(requests.get(image_prompt_url, stream=True).raw)
        >>> mask_prompt = Image.open(requests.get(mask_prompt_url, stream=True).raw).convert("L")

        >>> checkpoint = "BAAI/seggpt-vit-large"
        >>> model = SegGptForImageSegmentation.from_pretrained(checkpoint)
        >>> image_processor = SegGptImageProcessor.from_pretrained(checkpoint)

        >>> inputs = image_processor(images=image_input, prompt_images=image_prompt, prompt_masks=mask_prompt, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> result = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image_input.size[::-1]])[0]
        >>> print(list(result.shape))
        [170, 297]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is None:
            num_patches = self.model.embeddings.patch_embeddings.num_patches
            bool_masked_pos = ops.zeros(num_patches, dtype=ms.bool_)
            bool_masked_pos[num_patches // 2:] = 1
            bool_masked_pos = bool_masked_pos.unsqueeze(0)

        outputs = self.model(
            pixel_values=pixel_values,
            prompt_pixel_values=prompt_pixel_values,
            prompt_masks=prompt_masks,
            bool_masked_pos=bool_masked_pos,
            feature_ensemble=feature_ensemble,
            embedding_type=embedding_type,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        intermediate_hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[-1]
        intermediate_hidden_states = ops.cat(
            intermediate_hidden_states, axis=-1)
        pred_masks = self.decoder(intermediate_hidden_states)

        loss = None
        if labels is not None:
            loss_fn = SegGptLoss(self.config)
            loss = loss_fn(prompt_masks, pred_masks, labels, bool_masked_pos)

        if not return_dict:
            output = (pred_masks,)
            if output_hidden_states:
                output = output + (outputs[1],)

            if output_attentions:
                idx = 2 if output_hidden_states else 1
                output = output + (outputs[idx],)

            if loss is not None:
                output = (loss,) + output
            return output

        return SegGptImageSegmentationOutput(
            loss=loss,
            pred_masks=pred_masks,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "SegGptModel",
    "SegGptForImageSegmentation",
]
