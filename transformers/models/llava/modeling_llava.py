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

import mindspore

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
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
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
            or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
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


class LlavaMultiModalProjector(nn.Module):

    """
    LlavaMultiModalProjector is a class representing a multi-modal projector for processing image and text data
    simultaneously. It facilitates the transformation of image features through linear layers with activation functions
    to map them to text features.
    
    This class inherits from nn.Module and contains methods for initialization and forwarding the projection of image
    features to text features.
    The initialization method initializes the linear layers and activation function based on the provided configuration. 
    The forward method applies the linear transformations and activation functions to the input image features to
    generate the final hidden states for text representation.
    
    Example:
        ```python
        >>> config = LlavaConfig(vision_config=..., text_config=..., projector_hidden_act=...)
        >>> projector = LlavaMultiModalProjector(config)
        >>> hidden_states = projector.forward(image_features)
        ```
    """
    def __init__(self, config: LlavaConfig):
        """
        Initializes an instance of the LlavaMultiModalProjector class.
        
        Args:
            self: The object instance.
            config (LlavaConfig):
                The configuration object containing the settings for the projector.

                - config.vision_config.hidden_size (int): The size of the hidden layer for the visual input.
                - config.text_config.hidden_size (int): The size of the hidden layer for the text input.
                - config.projector_hidden_act (str): The activation function for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        '''
        This method forwards a multi-modal projector within the LlavaMultiModalProjector class.

        Args:
            self (LlavaMultiModalProjector): The instance of the LlavaMultiModalProjector class.
            image_features (tensor): The input tensor containing image features.

        Returns:
            tensor:
                The hidden states tensor obtained after processing the image features through linear and activation layers.

        Raises:
            None.
        '''
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaPreTrainedModel(PreTrainedModel):

    """
    The `LlavaPreTrainedModel` class is a subclass of the `PreTrainedModel` class in the Hugging Face library.
    It represents a pre-trained model for natural language processing tasks.

    This class provides functionality for initializing the weights of the model's cells.
    The `_init_weights` method is used to set the initial weights of the model's cells based on the specified configuration.
    The method supports different types of cells, including `Dense`, `Conv2d`, and `Embedding`.

    If the cell has a `class_embedding` attribute, the method initializes it using a normal distribution with
    a standard deviation specified by the `initializer_range` attribute of the configuration.

    For `Dense` and `Conv2d` cells, the method initializes the `weight` attribute using a normal distribution
    with the same standard deviation as above. If the cell has a `bias` attribute, it is initialized with zeros.

    For `Embedding` cells, the method initializes the `weight` attribute using a normal distribution with the same
    standard deviation as above. If the cell has a `padding_idx` attribute, the corresponding element in the
    weight matrix is set to zero.

    Note:
        The `LlavaPreTrainedModel` class assumes that the `PreTrainedModel` class is available in the code environment.

    """
    config_class = LlavaConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            nn.init.normal_(module.class_embedding, mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0

class LlavaForConditionalGeneration(LlavaPreTrainedModel):

    """
    LlavaForConditionalGeneration

    This class is a language model for conditional generation based on the Llava architecture.
    It extends the LlavaPreTrainedModel class.

    Attributes:
        vision_tower (AutoModel): The vision tower model for extracting image features.
        multi_modal_projector (LlavaMultiModalProjector): The multi-modal projector for combining image and text features.
        vocab_size (int): The size of the vocabulary used by the language model.
        language_model (AutoModelForCausalLM): The language model for generating text.
        pad_token_id (int): The ID of the padding token in the vocabulary. Defaults to -1 if not provided.

    Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import LlavaForConditionalGeneration
        ...
        >>> model = LlavaForConditionalGeneration(config)
        ...
        >>> input_ids = [1, 2, 3]
        >>> pixel_values = [0.1, 0.2, 0.3]
        >>> attention_mask = [1, 1, 1]
        ... 
        >>> output = model.forward(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        ```
    """
    def __init__(self, config: LlavaConfig):
        """
        Initializes an instance of the LlavaForConditionalGeneration class.

        Args:
            self: The instance of the class.
            config (LlavaConfig): An object of type LlavaConfig containing the configuration settings for the model.
                It specifies the configuration parameters for the vision tower, multi-modal projector, vocab size, 
                language model, pad token id, and other model settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings from the language model.

        Args:
            self (LlavaForConditionalGeneration): An instance of the LlavaForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the LlavaForConditionalGeneration language model.

        Args:
            self (LlavaForConditionalGeneration): The instance of the LlavaForConditionalGeneration class.
            value (Any): The input embeddings to be set for the language model.

        Returns:
            None.

        Raises:
            None.
        """
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """
        Retrieve the output embeddings from the language model used for conditional generation.

        Args:
            self: An instance of the LlavaForConditionalGeneration class.

        Returns:
            None: This method returns None, it simply delegates the call to the language model's
                get_output_embeddings method.

        Raises:
            None: However, if the language_model.get_output_embeddings() method raises any exceptions,
                they will propagate up to the caller.
        """
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the LlavaForConditionalGeneration model.

        Args:
            self (LlavaForConditionalGeneration): An instance of the LlavaForConditionalGeneration class.
            new_embeddings (Tensor): The new output embeddings to be set for the model.
                It should have the same shape as the original output embeddings.

        Returns:
            None.

        Raises:
            TypeError: If the provided new_embeddings parameter is not of type Tensor.
            ValueError: If the shape of the new_embeddings parameter does not match the
                shape of the original output embeddings.
        """
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        """
        Sets the decoder for the language model used in LlavaForConditionalGeneration.

        Args:
            self (LlavaForConditionalGeneration): The instance of the LlavaForConditionalGeneration class.
            decoder: The decoder object to be set for the language model.
                It should be compatible with the language model's requirements.

        Returns:
            None.

        Raises:
            TypeError: If the provided decoder is not of the correct type.
            ValueError: If the decoder object is invalid or incompatible with the language model.
        """
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        """
        Returns the decoder of the LlavaForConditionalGeneration model.

        Args:
            self: An instance of the LlavaForConditionalGeneration class.

        Returns:
            The decoder from the language model used by the LlavaForConditionalGeneration model.

        Raises:
            None.
        """
        return self.language_model.get_decoder()

    def tie_weights(self):
        """
        Ties the weights of the language model used for conditional generation in the LlavaForConditionalGeneration class.

        Args:
            self (LlavaForConditionalGeneration): An instance of the LlavaForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        """
        Resize the token embeddings for conditional generation in the LlavaForConditionalGeneration class.

        Args:
            self: The instance of the LlavaForConditionalGeneration class.

            new_num_tokens (int, optional): The new number of tokens to resize the embeddings to. Defaults to None.
                If provided, the token embeddings will be resized to accommodate this number of tokens.

            pad_to_multiple_of (int): The value to pad the token embeddings to a multiple of. Defaults to None.
                If provided, the token embeddings will be padded to a multiple of this value.

        Returns:
            nn.Embedding: The resized token embeddings after the operation.
                This updated nn.Embedding object reflects the changes made to the token embeddings.

        Raises:
            None specified.
        """
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        """
        Merges image features with input embeddings and applies necessary modifications.

        Args:
            self (LlavaForConditionalGeneration): The instance of the LlavaForConditionalGeneration class.
            image_features (Tensor): A tensor of shape (num_images, num_image_patches, embed_dim)
                representing the image features.
            inputs_embeds (Tensor): A tensor of shape (batch_size, sequence_length, embed_dim)
                representing the input embeddings.
            input_ids (Tensor): A tensor of shape (batch_size, sequence_length) representing the input token IDs.
            attention_mask (Tensor): A tensor of shape (batch_size, sequence_length) representing the attention mask.
            labels (Tensor): A tensor of shape (batch_size, sequence_length) representing the labels.

        Returns:
            None

        Raises:
            ValueError: If the number of image tokens provided in the input does not match the number of
                images given to the model.
        """
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not ops.sum(input_ids[:, -1] == mindspore.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = ops.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)).item() + sequence_length
        nonzero = ops.nonzero(input_ids != self.config.image_token_index)
        batch_indices, non_image_indices = ops.chunk(nonzero, 2, -1)

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
        image_to_overwrite = ops.all(final_embedding == 0, dim=-1)
        image_to_overwrite = (image_to_overwrite.int() & (image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]).int()).bool()

        if image_to_overwrite.sum() != reduce(lambda x, y: x * y, image_features.shape[:-1]):
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {ops.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.reshape(-1, embed_dim)
        final_attention_mask = (final_attention_mask.int() | image_to_overwrite.int()).bool()
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = ops.nonzero(input_ids == self.pad_token_id, as_tuple=True)
        if 0 not in batch_indices.shape:
            indices_to_mask = new_token_positions[batch_indices, pad_indices]
            final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
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
            Union[Tuple, LlavaCausalLMOutputWithPast]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, LlavaForConditionalGeneration
            ...
            >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
            >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            ...
            >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
            >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
            >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
            ```
        """
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
                batch_index, non_attended_tokens = ops.chunk(nonzero, 2, -1)

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

                attention_mask = ops.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
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
            loss = F.cross_entropy(
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
        '''
        Prepares inputs for text generation in the LlavaForConditionalGeneration class.
        
        Args:
            self (LlavaForConditionalGeneration): The instance of the LlavaForConditionalGeneration class.
            input_ids (Tensor): The input tensor containing the tokenized input sequence.
            past_key_values (Cache or tuple of Tensor, optional):
                The cache of past key values or tuple of tensors containing past key values.
            inputs_embeds (Tensor, optional): The input embeddings tensor.
            pixel_values (Tensor, optional): The tensor containing the pixel values.
            attention_mask (Tensor, optional): The attention mask tensor.
            **kwargs: Additional keyword arguments.
        
        Returns:
            dict: A dictionary containing the prepared model inputs for text generation.
        
        Raises:
            None.
        '''
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
        """
        Method _reorder_cache in class LlavaForConditionalGeneration.
        
        Args:
            self: LlavaForConditionalGeneration instance. Represents the current instance of the class.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.language_model._reorder_cache(*args, **kwargs)

__all__ = [
    "LlavaCausalLMOutputWithPast",
    "LlavaMultiModalProjector",
    "LlavaPreTrainedModel",
    "LlavaForConditionalGeneration",
]
