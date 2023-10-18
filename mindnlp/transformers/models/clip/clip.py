# Copyright 2022 Huawei Technologies Co., Ltd
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
"""MindNLP clip model"""
# pylint: disable=C0103
# pylint: disable=C0415
# pylint: disable=W0621

from typing import Optional
from collections import OrderedDict
import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, Parameter, dtype_to_nptype
from ...modeling_utils import PreTrainedModel
from .clip_config import CLIPVisionConfig, CLIPTextConfig, CLIPConfig


def _expand_mask(mask: Tensor, dtype: mindspore.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """

    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = ops.broadcast_to((mask[:, None, None, :]),
                               (bsz, 1, tgt_len, src_len)).astype(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.astype(mindspore.bool_),
        Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min)
    )


def contrastive_loss(logits):
    """
    Contrastive loss
    """
    return ops.cross_entropy(logits, ops.arange(len(logits)))


def clip_loss(similarity: Tensor) -> Tensor:
    """
    Clip loss
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.transpose())
    return (caption_loss + image_loss) / 2.0


class CLIPVisionEmbeddings(nn.Cell):
    """
    Clip Vision Embeddings
    """

    def __init__(self, config: CLIPVisionConfig):
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
            pad_mode='valid',
            has_bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = ops.broadcast_to(ops.arange(self.num_positions), (1, -1))


    def construct(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]

        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)


        class_embeds = ops.broadcast_to(self.class_embedding, (batch_size, 1, -1))
        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPTextEmbeddings(nn.Cell):
    """
    Clip Text Embeddings
    """
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.broadcast_to(ops.arange(config.max_position_embeddings), (1, -1))

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


class CLIPAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    """

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

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return ops.swapaxes(tensor.view(bsz, seq_len, self.num_heads, self.head_dim), 1, 2)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):

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
        attn_weights = ops.bmm(query_states, ops.swapaxes(key_states, 1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of shape {(bsz, 1, tgt_len, src_len)}, but is"
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
        attn_output = ops.swapaxes(attn_output, 1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class ClassInstantier(OrderedDict):
    """
    Class Instantier
    """

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


class QuickGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. 
    """

    def construct(self, x):
        return x * ops.sigmoid(1.702 * x)

ACT2CLS = {
    "quick_gelu": QuickGELUActivation
}

ACT2FN = ClassInstantier(ACT2CLS)


class CLIPMLP(nn.Cell):
    """
    Clip MLP
    """

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

class CLIPEncoderLayer(nn.Cell):
    """
    Clip Encoder Layer
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm((self.embed_dim,), epsilon=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm((self.embed_dim,), epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        causal_attention_mask: Tensor,
        output_attentions: Optional[bool] = False,
    ):

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


class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        pass

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLIPEncoder):
            module.gradient_checkpointing = value

    def get_position_embeddings(self):
        pass

    def init_model_weights(self):
        """
        Init model weights
        """

    def post_init(self):
        pass

    def resize_position_embeddings(self):
        pass

    def save(self):
        """
        Save
        """


class CLIPTextTransformer(nn.Cell):
    """
    Clip Text Transformer
    """
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm((embed_dim,), epsilon=config.layer_norm_eps)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        bsz, seq_len = input_shape

        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[
            ops.arange(last_hidden_state.shape[0]),
            ops.argmax(input_ids.astype(mindspore.int32), dim=-1),
        ]

        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = ops.fill(dtype, (bsz, seq_len, seq_len), Tensor(np.finfo(dtype_to_nptype(dtype)).min))
        mask = mindspore.numpy.triu(mask, k=1)
        mask = mask.expand_dims(1)
        return mask


class CLIPEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].
    Args:
        config: CLIPConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                custom_forward_func = create_custom_forward(encoder_layer)
                layer_outputs = custom_forward_func(hidden_states, attention_mask, causal_attention_mask)
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

        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


class CLIPTextModel(CLIPPreTrainedModel):
    """
    Clip Text Model
    """
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        """
        Clip Text Model init
        """
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        Get input embeddings
        """
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, new_embeddings):
        """
        Set input embeddings
        """
        self.text_model.embeddings.token_embedding = new_embeddings

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """
        Clip Text Model contruct
        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class CLIPVisionTransformer(nn.Cell):
    """
    CLip Vision Transformer
    """
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm((embed_dim,), epsilon=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm((embed_dim,), epsilon=config.layer_norm_eps)

    def construct(
        self,
        pixel_values: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return (last_hidden_state, pooled_output) + encoder_outputs[1:]


class CLIPVisionModel(CLIPPreTrainedModel):
    """
    Clip Vision Model
    """
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        """
        Clip Vision Model init
        """
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        Get input embeddings
        """
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """
        Clip Vision Model construct
        """
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class CLIPModel(CLIPPreTrainedModel):
    """
    Clip Model
    """
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        """
        Clip Model init
        """
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Dense(self.vision_embed_dim, self.projection_dim, has_bias=False)
        self.text_projection = nn.Dense(self.text_embed_dim, self.projection_dim, has_bias=False)
        self.logit_scale = Parameter(ops.ones([]) * self.config.logit_scale_init_value)

    def get_text_features(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        """
        Get text features
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        """
        Get image features
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """
        Clip Model construct
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)


        image_embeds = image_embeds / ops.norm(image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / ops.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        logit_scale = ops.exp(self.logit_scale)
        logits_per_text = ops.matmul(text_embeds, ops.transpose(image_embeds, (1, 0))) * logit_scale
        logits_per_image = ops.transpose(logits_per_text, (1, 0))

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
        return ((loss,) + output) if loss is not None else output


class CLIPTextModelWithProjection(CLIPPreTrainedModel):
    """
    Clip Text Model with projection
    """
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        """
        Clip Text Model with projection init
        """
        super().__init__(config)

        self.text_model = CLIPTextTransformer(config)
        self.text_projection = nn.Dense(config.hidden_size, config.projection_dim, has_bias=False)

    def get_input_embeddings(self) -> nn.Cell:
        """
        Get input embeddings
        """
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, new_embeddings):
        """
        Set input embeddings
        """
        self.text_model.embeddings.token_embedding = new_embeddings

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """
        Clip Text Model with projection construct
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs[1]

        text_embeds = self.text_projection(pooled_output)

        outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
        return tuple(output for output in outputs if output is not None)


class CLIPVisionModelWithProjection(CLIPPreTrainedModel):
    """
    Clip Vision Model with projection
    """
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        """
        Clip Vision Model with projection init
        """
        super().__init__(config)

        self.vision_model = CLIPVisionTransformer(config)

        self.visual_projection = nn.Dense(config.hidden_size, config.projection_dim, has_bias=False)

        # Initialize weights and apply final processing
        #self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        Get input embeddigns
        """
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """
        Clip Vision Model with projection construct
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = vision_outputs[1]

        image_embeds = self.visual_projection(pooled_output)

        outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
        return tuple(output for output in outputs if output is not None)

__all__ = [
        "CLIPModel",
        "CLIPPreTrainedModel",
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
        "CLIPVisionModel",
        "CLIPVisionModelWithProjection",
    ]
