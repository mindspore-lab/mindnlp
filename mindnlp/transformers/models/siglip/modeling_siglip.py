# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
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
"""PyTorch Siglip model."""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor

from mindspore.common.initializer import initializer, Normal, XavierUniform, Uniform, TruncatedNormal

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from mindnlp.utils import (
    ModelOutput,
    logging,
)
from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SiglipConfig"
_CHECKPOINT_FOR_DOC = "google/siglip-base-patch16-224"


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->Siglip
class SiglipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[Tensor] = None
    last_hidden_state: Tensor = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->Siglip
class SiglipTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: Optional[Tensor] = None
    last_hidden_state: Tensor = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->Siglip
class SiglipOutput(ModelOutput):
    """
    Args:
        loss (`Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`SiglipTextModel`].
        image_embeds(`Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`SiglipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipVisionModel`].
    """

    loss: Optional[Tensor] = None
    logits_per_image: Tensor = None
    logits_per_text: Tensor = None
    text_embeds: Tensor = None
    image_embeds: Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class SiglipVisionEmbeddings(nn.Cell):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = ops.arange(self.num_positions).broadcast_to((1, -1))

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        """
        This method is an adapted method for SigLIP (due to SigLIP not having class embedding unlike other ViTs)
        that allows the model to interpolate the pre-trained position encodings such that it can be usable on
        higher resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        position_embeddings = self.position_embedding.weight.unsqueeze(0)
        num_patches = embeddings.shape[1]
        num_positions = position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return position_embeddings

        dim = embeddings.shape[-1]
        height = height // self.patch_size
        width = width // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1

        patch_pos_embed = position_embeddings.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = ops.interpolate(
            patch_pos_embed,
            scale_factor=(height / math.sqrt(num_positions), width / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def construct(self, pixel_values: Tensor, interpolate_pos_encoding=False) -> Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).swapaxes(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->Siglip
class SiglipTextEmbeddings(nn.Cell):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized

        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class ScaleDotProductAttention(nn.Cell):
    def __init__(self, dropout_p=0):
        super().__init__()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=dropout_p)
        self.sqrt = ops.Sqrt()

    def construct(self, query, key, value, attn_mask=None):
        embedding_size = query.shape[-1]
        scaling_factor = self.sqrt(Tensor(embedding_size, dtype=mindspore.float32))

        attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = ops.matmul(attn, value)
        return output, attn


class SiglipAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        scaled_dot_product_attention = ScaleDotProductAttention(dropout_p=self.dropout if self.training else 0.0)
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) * self.scale

        if attn_weights.shape != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipSdpaAttention(SiglipAttention):

    """
    Siglip attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `SiglipAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from SiglipAttention.forward and transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning(
                "SiglipModel is using SiglipSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        attn_output = self.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask
        )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


SIGLIP_ATTENTION_CLASSES = {
    "eager": SiglipAttention,
    "sdpa": SiglipSdpaAttention,
}


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Cell):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SIGLIP_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.layer_norm1 = ops.LayerNorm([self.embed_dim], epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = ops.LayerNorm([self.embed_dim], epsilon=config.layer_norm_eps)

    # Ignore copy
    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor]:
        """
        Args:
            hidden_states (`Tensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`Tensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SiglipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SiglipConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "SiglipTextEmbeddings",
        "SiglipEncoderLayer",
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
        "SiglipMultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, SiglipVisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, SiglipConfig)
                else self.config.hidden_size
            )
            cell.position_embedding.weight.set_data(initializer(Normal(sigma=1/np.sqrt(width)), cell.position_embedding.weight.shape,
                                                                cell.position_embedding.weight.dtype))
        elif isinstance(cell, nn.Embedding):
            # default_flax_embed_init(cell.weight)  # 标准正态分布
            # cell.weight.set_data(initializer(Normal(), cell.weight.shape, cell.weight.dtype))
            embedding_table = np.random.normal(0.0, 0.02, cell.embedding_table)
            if cell.padding_idx:
                embedding_table[cell.padding_idx] = 0
            cell.embedding_table.set_data(Tensor(embedding_table, cell.embedding_table.dtype))
        elif isinstance(cell, SiglipAttention):
            cell.q_proj.weight.set_data(initializer(XavierUniform(), cell.q_proj.weight.shape, cell.q_proj.weight.dtype))
            # nn.init.xavier_uniform_(cell.k_proj.weight)
            cell.k_proj.weight.set_data(initializer(XavierUniform(), cell.k_proj.weight.shape, cell.k_proj.weight.dtype))
            cell.v_proj.weight.set_data(initializer((XavierUniform(), cell.v_proj.weight.shape, cell.v_proj.weight.dtype)))
            cell.out_proj.weight.set_data(initializer(XavierUniform(), cell.out_proj.weight.shape, cell.out_proj.weight.dtype))
            cell.q_proj.bias.set_data(initializer("zeros", cell.q_proj.bias.shape, cell.q_proj.bias.dtype))
            cell.k_proj.bias.set_data(initializer("zeros", cell.k_proj.bias.shape, cell.k_proj.bias.dtype))
            cell.v_proj.bias.set_data(initializer("zeros", cell.v_proj.bias.shape, cell.v_proj.bias.dtype))
            cell.out_proj.bias.set_data(initializer("zeros", cell.out_proj.bias.shape, cell.out_proj.bias.dtype))
        elif isinstance(cell, SiglipMLP):
            cell.fc1.weight.set_data(initializer(XavierUniform(), cell.fc1.weight.shape, cell.fc1.weight.dtype))
            cell.fc2.weight.set_data(initializer(XavierUniform(), cell.fc2.weight.shape, cell.fc2.weight.dtype))
            cell.fc1.bias.set_data(initializer(Normal(sigma=1e-6), cell.fc1.bias.shape, cell.fc1.bias.dtype))
            cell.fc2.bias.set_data(initializer(Normal(sigma=1e-6), cell.fc2.bias.shape, cell.fc2.bias.dtype))
        elif isinstance(cell, SiglipMultiheadAttentionPoolingHead):
            cell.probe.data.set_data(initializer(XavierUniform(), cell.probe.data.shape, cell.probe.data.dtype))
            cell.attention.in_proj_weight.data.set_data(initializer(XavierUniform(),cell.attention.in_proj_weight.data.shape,
                                                                    cell.attention.in_proj_weight.data.dtype))
            cell.attention.in_proj_bias.data.set_data(initializer("zeros", cell.attention.in_proj_bias.data.shape,
                                                                  cell.attention.in_proj_bias.data.dtype))
        elif isinstance(cell, SiglipModel):
            logit_scale_init = ops.log(mindspore.tensor(1.0))
            cell.logit_scale.data.set_data(initializer(logit_scale_init, cell.logit_scale.data.shape,
                                                       cell.logit_scale.data.dtype))
            cell.logit_bias.data.set_data(initializer("zeros", cell.logit_bias.data.shape, cell.logit_bias.data.dtype))
        elif isinstance(cell, SiglipForImageClassification):
            cell.classifier.weight.set_data(initializer(Normal(sigma=self.config.vision_config.hidden_size**-0.5*self.config.initializer_factor),
                                                        cell.classifier.weight.shape, cell.classifier.weight.dtype))
        elif isinstance(cell, (nn.Dense, nn.Conv2d)):
            cell.weight.set_data(initializer(TruncatedNormal(), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer(Normal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            # cell.bias.data.zero_()
            # cell.weight.data.fill_(1.0)
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->Siglip
class SiglipEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class SiglipTextTransformer(nn.Cell):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.final_layer_norm = ops.LayerNorm([embed_dim], epsilon=config.layer_norm_eps)

        self.head = nn.Dense(embed_dim, embed_dim)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiglipTextModel(SiglipPreTrainedModel):
    config_class = SiglipTextConfig

    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        self.text_model = SiglipTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SiglipVisionTransformer(nn.Cell):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = ops.LayerNorm([embed_dim], epsilon=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)


    def construct(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiglipMultiheadAttentionPoolingHead(nn.Cell):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = mindspore.Parameter(ops.randn(1, 1, config.hidden_size), 'probe')
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = ops.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def construct(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )


class SiglipModel(SiglipPreTrainedModel):
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)

        if not isinstance(config.text_config, SiglipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        # First, initialize the text and vision models with proper attention implementation
        text_model = SiglipTextModel._from_config(text_config, attn_implementation=config._attn_implementation)
        vision_model = SiglipVisionModel._from_config(vision_config, attn_implementation=config._attn_implementation)

        # Second, get the text and vision submodules (for backward compatibility)
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = mindspore.Parameter(ops.randn(1), 'logit_scale')
        self.logit_bias = mindspore.Parameter(ops.randn(1), 'logit_bias')

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        r"""
        Returns:
            text_features (`Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipTextModel`].

        Examples:

        ```python
        ```"""
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        return pooled_output

    def get_image_features(
        self,
        pixel_values: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor:
        r"""
        Returns:
            image_features (`Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipVisionModel`].

        Examples:

        ```python
        ```"""
        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        pooled_output = vision_outputs[1]

        return pooled_output

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, SiglipOutput]:
        r"""
        Returns:

        Examples:

        ```python
        31.9% that image 0 is 'a photo of 2 cats'
        ```"""
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = (
            ops.matmul(text_embeds, image_embeds) * self.logit_scale.exp()
            + self.logit_bias
        )
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # Adapted from https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip.py#L287
            eye = ops.eye(logits_per_text.shape[0])
            m1_diag1 = -ops.ones_like(logits_per_text) + 2 * eye
            loglik = ops.logsigmoid(m1_diag1 * logits_per_text)
            nll = -ops.sum(loglik, dim=-1)
            loss = nll.mean()

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class SiglipForImageClassification(SiglipPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels

        # Create the vision model with proper attention
        # and take only vision_model submodule (for backward compatibility)
        vision_model = SiglipVisionModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.vision_model = vision_model.vision_model

        # Classifier head
        self.classifier = (
            nn.Dense(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        Predicted class: LABEL_1
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]

        # average pool the patch tokens
        sequence_output = ops.mean(sequence_output, axis=1)
        # apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == mindspore.int64 or labels.dtype == mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["SiglipVisionModel",
           "SiglipModel",
           "SiglipPreTrainedModel",
           "SiglipMLP",
           "SiglipTextModel",
           "SiglipForImageClassification",
           "SiglipTextModelOutput",
           "SiglipOutput",
           "SiglipVisionEmbeddings",
           "SiglipTextEmbeddings",
           "SiglipAttention",
           "SiglipSdpaAttention",
           "SiglipEncoderLayer",
           "SiglipEncoder",
           "SiglipTextTransformer",
           "SiglipMultiheadAttentionPoolingHead",]
