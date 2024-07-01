# coding=utf-8
# Copyright 2022 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
"""Mindspore CLIPSeg model."""

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.nn import BCEWithLogitsLoss
from mindspore.common.initializer import initializer, Normal
from mindnlp.utils import (
    ModelOutput,
    logging,
)
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel

from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "CIDAS/clipseg-rd64-refined"


# contrastive loss function, adapted from
def contrastive_loss(logits: mindspore.Tensor) -> mindspore.Tensor:
    return ops.cross_entropy(logits, ops.arange(len(logits)))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->clipseg
def clipseg_loss(similarity: mindspore.Tensor) -> mindspore.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->CLIPSeg
class CLIPSegOutput(ModelOutput):
    """
    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`mindspore.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`mindspore.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegTextModel`].
        image_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """

    loss: Optional[mindspore.Tensor] = None
    logits_per_image: mindspore.Tensor = None
    logits_per_text: mindspore.Tensor = None
    text_embeds: mindspore.Tensor = None
    image_embeds: mindspore.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class CLIPSegDecoderOutput(ModelOutput):
    """
    Args:
        logits (`mindspore.Tensor` of shape `(batch_size, height, width)`):
            Classification scores for each pixel.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class CLIPSegImageSegmentationOutput(ModelOutput):
    """
    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        ...
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    conditional_embeddings: mindspore.Tensor = None
    pooled_output: mindspore.Tensor = None
    vision_model_output: BaseModelOutputWithPooling = None
    decoder_output: CLIPSegDecoderOutput = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["vision_model_output", "decoder_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLIPSegVisionEmbeddings(nn.Cell):
    # Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.__init__ with CLIP->CLIPSeg
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = Parameter(ops.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            has_bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = ops.arange(self.num_positions).expand((1, -1))

    def interpolate_position_embeddings(self, new_size):
        if len(new_size) != 2:
            raise ValueError("new_size should consist of 2 values")

        num_patches_one_direction = int(self.num_patches**0.5)
        # we interpolate the position embeddings in 2D
        a = self.position_embedding.weight[1:].T.view(
            1, self.config.hidden_size, num_patches_one_direction, num_patches_one_direction
        )
        b = (
            ops.interpolate(a, new_size, mode="bicubic", align_corners=False)
            .squeeze(0)
            .view(self.config.hidden_size, new_size[0] * new_size[1])
            .T
        )
        result = ops.cat([self.position_embedding.weight[:1], b])

        return result

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)

        if embeddings.shape[1] != self.num_positions:
            new_shape = int(math.sqrt(embeddings.shape[1] - 1))
            embeddings = embeddings + self.interpolate_position_embeddings((new_shape, new_shape))
            embeddings = embeddings.to(embeddings.dtype)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->CLIPSeg
class CLIPSegTextEmbeddings(nn.Cell):
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).expand((1, -1))


    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->CLIPSeg
class CLIPSegAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
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

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        causal_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->CLIPSeg
class CLIPSegMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->CLIPSeg
class CLIPSegEncoderLayer(nn.Cell):
    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPSegAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = CLIPSegMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        causal_attention_mask: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
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


class CLIPSegPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPSegConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(cell, CLIPSegTextEmbeddings):
            cell.token_embedding.weight.set_data(initializer(Normal(factor * 0.02),
                                                 cell.token_embedding.weight.shape, cell.token_embedding.weight.dtype))
            cell.position_embedding.weight.set_data(initializer(Normal(factor * 0.02),
                                        cell.position_embedding.weight.shape, cell.position_embedding.weight.dtype))
        elif isinstance(cell, CLIPSegVisionEmbeddings):
            factor = self.config.initializer_factor
            cell.class_embedding.set_data(initializer(Normal(cell.embed_dim**-0.5 * factor),
                                        cell.class_embedding.shape, cell.class_embedding.dtype))
            cell.patch_embedding.weight.set_data(initializer(Normal(cell.config.initializer_range * factor),
                                                 cell.patch_embedding.weight.shape, cell.patch_embedding.weight.dtype))
            cell.position_embedding.weight.set_data(initializer(Normal(cell.config.initializer_range * factor),
                                                 cell.position_embedding.weight.shape, cell.position_embedding.weight.dtype))

        elif isinstance(cell, CLIPSegAttention):
            factor = self.config.initializer_factor
            in_proj_std = (cell.embed_dim**-0.5) * ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (cell.embed_dim**-0.5) * factor

            cell.q_proj.weight.set_data(initializer(Normal(in_proj_std),
                                        cell.q_proj.weight.shape, cell.q_proj.weight.dtype))
            cell.k_proj.weight.set_data(initializer(Normal(in_proj_std),
                                        cell.k_proj.weight.shape, cell.k_proj.weight.dtype))
            cell.v_proj.weight.set_data(initializer(Normal(in_proj_std),
                                        cell.v_proj.weight.shape, cell.v_proj.weight.dtype))
            cell.out_proj.weight.set_data(initializer(Normal(out_proj_std),
                                        cell.out_proj.weight.shape, cell.out_proj.weight.dtype))

        elif isinstance(cell, CLIPSegMLP):
            factor = self.config.initializer_factor
            in_proj_std = (cell.config.hidden_size**-0.5) * ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * cell.config.hidden_size) ** -0.5 * factor

            cell.fc1.weight.set_data(initializer(Normal(fc_std),
                                    cell.fc1.weight.shape, cell.fc1.weight.dtype))
            cell.fc2.weight.set_data(initializer(Normal(in_proj_std),
                                    cell.fc2.weight.shape, cell.fc2.weight.dtype))

        elif isinstance(cell, CLIPSegModel):
            cell.text_projection.weight.set_data(initializer(Normal(cell.text_embed_dim**-0.5 * self.config.initializer_factor),
                                    cell.text_projection.weight.shape, cell.text_projection.weight.dtype))

            cell.visual_projection.weight.set_data(initializer(Normal(cell.vision_embed_dim**-0.5 * self.config.initializer_factor),
                                    cell.visual_projection.weight.shape, cell.visual_projection.weight.dtype))

        if isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        if isinstance(cell, nn.Dense) and cell.bias is not None:
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->CLIPSeg
class CLIPSegEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPSegEncoderLayer`].

    Args:
        config: CLIPSegConfig
    """

    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([CLIPSegEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[mindspore.Tensor] = None,
        causal_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

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
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
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


class CLIPSegTextTransformer(nn.Cell):
    # Copied from transformers.models.clip.modeling_clip.CLIPTextTransformer.__init__ with CLIP->CLIPSeg
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPSegTextEmbeddings(config)
        self.encoder = CLIPSegEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id
    # Copied from transformers.models.clip.modeling_clip.CLIPTextTransformer.forward with clip->clipseg, CLIP->CLIPSeg
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
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

        # CLIPSeg's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIPSeg/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clipseg/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIPSeg model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to mindspore.int32 for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                ops.arange(last_hidden_state.shape[0]),
                input_ids.to(dtype=mindspore.int32).argmax(axis=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                ops.arange(last_hidden_state.shape[0]),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=mindspore.int32) == self.eos_token_id)
                .int()
                .argmax(axis=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPSegTextModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegTextConfig

    _no_split_modules = ["CLIPSegTextEmbeddings", "CLIPSegEncoderLayer"]

    def __init__(self, config: CLIPSegTextConfig):
        super().__init__(config)
        self.text_model = CLIPSegTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPSegVisionTransformer(nn.Cell):
    # Copied from transformers.models.clip.modeling_clip.CLIPVisionTransformer.__init__ with CLIP->CLIPSeg
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPSegVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)
        self.encoder = CLIPSegEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)
    # Copied from transformers.models.clip.modeling_clip.CLIPVisionTransformer.forward
    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
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

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPSegVisionModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPSegVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegVisionModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPSegModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPSegTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPSegTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPSegVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPSegVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPSegTextTransformer(text_config)
        self.vision_model = CLIPSegVisionTransformer(vision_config)

        self.visual_projection = nn.Dense(self.vision_embed_dim, self.projection_dim, has_bias=False)
        self.text_projection = nn.Dense(self.text_embed_dim, self.projection_dim, has_bias=False)
        self.logit_scale = mindspore.Parameter(mindspore.Tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""
        Returns:
            text_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPSegTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
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
        text_features = self.text_projection(pooled_output)

        return text_features
    def get_image_features(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""
        Returns:
            image_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPSegVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
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
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPSegOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
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
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = ops.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clipseg_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPSegOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CLIPSegDecoderLayer(nn.Cell):
    """
    CLIPSeg decoder layer, which is identical to `CLIPSegEncoderLayer`, except that normalization is applied after
    self-attention/MLP, rather than before.
    """

    # Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer.__init__ with CLIP->CLIPSeg
    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPSegAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = CLIPSegMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        causal_attention_mask: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPSegDecoder(CLIPSegPreTrainedModel):
    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)

        self.conditional_layer = config.conditional_layer

        self.film_mul = nn.Dense(config.projection_dim, config.reduce_dim)
        self.film_add = nn.Dense(config.projection_dim, config.reduce_dim)

        if config.use_complex_transposed_convolution:
            transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)

            self.transposed_convolution = nn.SequentialCell(
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3),
                nn.ReLU(),
                nn.Conv2dTranspose(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=transposed_kernels[0],
                    stride=transposed_kernels[0],
                ),
                nn.ReLU(),
                nn.Conv2dTranspose(
                    config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                ),
            )
        else:
            self.transposed_convolution = nn.Conv2dTranspose(
                config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size
            )

        depth = len(config.extract_layers)
        self.reduces = nn.CellList(
            [nn.Dense(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
        )

        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        self.layers = nn.CellList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

    def construct(
        self,
        hidden_states: Tuple[mindspore.Tensor],
        conditional_embeddings: mindspore.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        activations = hidden_states[::-1]

        output = None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )
                output = output.permute(1, 0, 2)

            layer_outputs = layer(
                output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
            )

            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        output = output[:, 1:, :].permute(0, 2, 1)  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

        size = int(math.sqrt(output.shape[2]))

        batch_size = conditional_embeddings.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output).squeeze(1)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)

        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)

        self.config = config

        self.clip = CLIPSegModel(config)
        self.extract_layers = config.extract_layers

        self.decoder = CLIPSegDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_conditional_embeddings(
        self,
        batch_size: int = None,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        conditional_pixel_values: Optional[mindspore.Tensor] = None,
    ):
        if input_ids is not None:
            # compute conditional embeddings from texts
            if len(input_ids) != batch_size:
                raise ValueError("Make sure to pass as many prompt texts as there are query images")
            #with torch.no_grad():
            conditional_embeddings = self.clip.get_text_features(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
        elif conditional_pixel_values is not None:
            # compute conditional embeddings from images
            if len(conditional_pixel_values) != batch_size:
                raise ValueError("Make sure to pass as many prompt images as there are query images")
            #with torch.no_grad():
            conditional_embeddings = self.clip.get_image_features(conditional_pixel_values)
        else:
            raise ValueError(
                "Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`"
            )

        return conditional_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        conditional_pixel_values: Optional[mindspore.Tensor] = None,
        conditional_embeddings: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPSegOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["a cat", "a remote", "a blanket"]
        >>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> print(logits.shape)
        mindspore.shape([3, 352, 352])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the query images through the frozen CLIP vision encoder
        #with torch.no_grad():
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        pooled_output = self.clip.visual_projection(vision_outputs[1])

        hidden_states = vision_outputs.hidden_states if return_dict else vision_outputs[2]
        # we add +1 here as the hidden states also include the initial embeddings
        activations = [hidden_states[i + 1] for i in self.extract_layers]

        # update vision_outputs
        if return_dict:
            vision_outputs = BaseModelOutputWithPooling(
                last_hidden_state=vision_outputs.last_hidden_state,
                pooler_output=vision_outputs.pooler_output,
                hidden_states=vision_outputs.hidden_states if output_hidden_states else None,
                attentions=vision_outputs.attentions,
            )
        else:
            vision_outputs = (
                vision_outputs[:2] + vision_outputs[3:] if not output_hidden_states else vision_outputs
            )

        # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        if conditional_embeddings is None:
            conditional_embeddings = self.get_conditional_embeddings(
                batch_size=pixel_values.shape[0],
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                conditional_pixel_values=conditional_pixel_values,
            )
        else:
            if conditional_embeddings.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    "Make sure to pass as many conditional embeddings as there are query images in the batch"
                )
            if conditional_embeddings.shape[1] != self.config.projection_dim:
                raise ValueError(
                    "Make sure that the feature dimension of the conditional embeddings matches"
                    " `config.projection_dim`."
                )

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.decoder(
            activations,
            conditional_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]

        loss = None
        if labels is not None:
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        if not return_dict:
            output = (logits, conditional_embeddings, pooled_output, vision_outputs, decoder_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPSegImageSegmentationOutput(
            loss=loss,
            logits=logits,
            conditional_embeddings=conditional_embeddings,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,
            decoder_output=decoder_outputs,
        )
__all__ = [
    "CLIPSegModel",
    "CLIPSegPreTrainedModel",
    "CLIPSegTextModel",
    "CLIPSegVisionModel",
    "CLIPSegForImageSegmentation",
    "CLIPSegDecoder",
]
