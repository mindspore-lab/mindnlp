# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
"""MindSpore Llava model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import reduce

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Normal

from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ....utils import logging
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_llava import LlavaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaConfig"


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(mindspore.Tensor)`, *optional*):
            Tuple of `mindspore.Tensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    past_key_values: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    image_hidden_states: Optional[Tuple[mindspore.Tensor]] = None


class LlavaMultiModalProjector(nn.Cell):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Dense(config.vision_config.hidden_size, config.text_config.hidden_size, has_bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Dense(config.text_config.hidden_size, config.text_config.hidden_size, has_bias=True)

    def construct(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(cell, "class_embedding"):
            cell.class_embedding.initialize(Normal(std))

        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            cell.weight.data.initialize(Normal(std))
            if cell.bias is not None:
                cell.bias.initialize('zeros')
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class LlavaForConditionalGeneration(LlavaPreTrainedModel):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.vocab_size
        self.vocab_size = model_embeds.vocab_size
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not ops.sum(input_ids[:, -1] == mindspore.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = ops.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)).item() + sequence_length
        nonzero = ops.nonzero(input_ids != self.config.image_token_index)
        batch_indices, non_image_indices = ops.tensor_split(nonzero, 2, -1)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = ops.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = ops.zeros(
            (batch_size, max_embed_dim, embed_dim), dtype=inputs_embeds.dtype
        )
        final_attention_mask = ops.zeros(
            (batch_size, max_embed_dim), dtype=attention_mask.dtype
        )
        if labels is not None:
            final_labels = ops.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype
            )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = ops.all(final_embedding == 0, axis=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]

        if image_to_overwrite.sum() != reduce(lambda x, y: x * y, image_features.shape[:-1]):
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {ops.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.reshape(-1, embed_dim)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        nonzero = ops.nonzero(input_ids == self.pad_token_id)
        batch_indices, pad_indices = ops.tensor_split(nonzero, 2, -1)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        pixel_values: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = ops.full_like(attention_mask, self.config.ignore_index).to(mindspore.int64)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                nonzero = ops.nonzero(first_layer_past_key_value.float().sum(-2) == 0)
                batch_index, non_attended_tokens = ops.tensor_split(nonzero, 2, -1)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = ops.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.shape[-1]
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = ops.cat((extended_attention_mask, attention_mask[:, -target_length:]), axis=1)
                position_ids = ops.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][ops.ne(shift_attention_mask, 0)]
                shift_labels = labels[..., 1:][ops.ne(shift_attention_mask, 0)]
            else:
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

__all__ = [
    "LlavaCausalLMOutputWithPast",
    "LlavaMultiModalProjector",
    "LlavaPreTrainedModel",
    "LlavaForConditionalGeneration",
]
