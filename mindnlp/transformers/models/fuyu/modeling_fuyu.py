# coding=utf-8
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
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
"""Mindspore Fuyu model."""

from typing import List, Optional, Tuple, Union

import mindspore

from mindnlp.core import nn, ops
from mindnlp.utils import logging

from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto.modeling_auto import AutoModelForCausalLM

from .configuration_fuyu import FuyuConfig

logger = logging.get_logger(__name__)

class FuyuPreTrainedModel(PreTrainedModel):
    config_class = FuyuConfig
    base_model_prefix = "fuyu"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0

class FuyuForCausalLM(FuyuPreTrainedModel):
    def __init__(self, config: FuyuConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size
        # self.language_model = AutoModelForCausalLM.from_config(
        #     config.text_config, attn_implementation=config._attn_implementation
        # )
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config
        )

        self.vision_embed_tokens = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels, config.hidden_size
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
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
        # TODO: config.vocab_size is deprecated.43.
        # `resize_token_embeddings` should work from `modeling_utils.py``
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        # self.config.text_config.vocab_size = model_embeds.vocab_size
        # self.config.vocab_size = model_embeds.vocab_size
        # self.vocab_size = model_embeds.vocab_size
        return model_embeds

    def gather_continuous_embeddings(
        self,
        word_embeddings: mindspore.Tensor,
        continuous_embeddings: List[mindspore.Tensor],
        image_patch_input_indices: mindspore.Tensor,
    ) -> mindspore.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Tensor of word embeddings.
            continuous_embeddings (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape
                [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
                indices in image_patch_input_indices for that batch element.
            image_patch_input_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Tensor of indices of the image patches in the input_ids tensor.
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )
        output_embeddings = word_embeddings.copy()
        for batch_idx in range(word_embeddings.shape[0]):
            # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
            # positions in word_embeddings that we want to replace with content from continuous_embeddings.

            dst_indices = ops.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            # dst_indices = mindspore.ops.nonzero(image_patch_input_indices[batch_idx] >= 0).reshape(-1)

            # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
            # want to use to replace the values in word_embeddings.
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        image_patches: mindspore.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import FuyuProcessor, FuyuForCausalLM
        >>> from PIL import Image
        >>> import requests

        >>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        >>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

        >>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Generate a coco-style caption.\n"

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=7)
        >>> generation_text = processor.batch_decode(generated_ids[:, -7:], skip_special_tokens=True)
        >>> print(generation_text[0])
        A blue bus parked on the side of a road.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_is or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            # device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                patch_embeddings = [
                    self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype))
                    .squeeze(0)
                    for patch in image_patches
                ]
                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_patches=None,
        image_patches_indices=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_patches_indices": image_patches_indices if past_key_values is None else None,
                "image_patches": image_patches if past_key_values is None else None,
            }
        )
        return model_inputs

__all__ = [
        "FuyuForCausalLM",
        "FuyuPreTrainedModel",
    ]
