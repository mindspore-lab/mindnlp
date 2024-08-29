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
""" MindSpore VipLlava model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import reduce

import numpy as np

import mindspore as ms
from mindspore import Tensor

from mindnlp.core import nn, ops
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ....utils import logging
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_vipllava import VipLlavaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VipLlavaConfig"


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->VipLlava
class VipLlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for VipLlava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed
            or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """
    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[List[Tensor]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None
    image_hidden_states: Optional[Tuple[Tensor]] = None


class VipLlavaMultiModalProjector(nn.Module):

    """
    Represents a multi-modal projector for the VipLlava model, used to project hidden states from both vision and
    text modalities.
    
    This class inherits from nn.Module and contains methods to initialize the projector and forward the projection
    process.
    
    Attributes:
        projector_layernorm (nn.LayerNorm): Layer normalization for the projector.
        linear_1 (nn.Linear): First linear transformation for the projector.
        act (function): Activation function applied after the first linear transformation.
        linear_2 (nn.Linear): Second linear transformation for the projector.
    
    Methods:
        __init__: Initializes the multi-modal projector with the provided configuration.
        forward: Constructs the projection process by applying layer normalization, linear transformations,
            and activation function.
    
    """
    def __init__(self, config: VipLlavaConfig):
        """
        Initializes an instance of the VipLlavaMultiModalProjector class.
        
        Args:
            self: The instance of the class.
            config (VipLlavaConfig): The configuration object that contains the parameters for the projector.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.projector_layernorm = nn.LayerNorm(
            len(config.vision_feature_layers) * config.vision_config.hidden_size, eps=config.projector_layernorm_eps
        )

        self.linear_1 = nn.Linear(
            len(config.vision_feature_layers) *
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, hidden_states):
        """
        Constructs the multi-modal projector for the VipLlava model.

        Args:
            self (VipLlavaMultiModalProjector): An instance of the VipLlavaMultiModalProjector class.
            hidden_states (Tensor): The input hidden states to be projected. Should be of shape (batch_size, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        hidden_states = self.projector_layernorm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


VIPLLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VipLlavaConfig`] or [`VipLlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Copied from transformers.models.llava.modeling_llava.LlavaPreTrainedModel with Llava->VipLlava,llava->vipllava
class VipLlavaPreTrainedModel(PreTrainedModel):

    """
    This class represents a pre-trained model for VipLlava. It is a subclass of the PreTrainedModel class.

    The VipLlavaPreTrainedModel class provides methods for initializing the weights of the model and checking whether
    the model supports SDPA (Semi-Definite Programming Algorithm).

    Methods:
        _init_weights:
            Initializes the weights of the given module using random normal distribution with a standard deviation
            determined by the configuration.

            - If the module has a class_embedding attribute, it sets the data of the class_embedding tensor with random
            values.
            - If the module is an instance of nn.Linear or nn.Conv2d, it sets the data of the weight tensor with random
            values and initializes the bias tensor with zeros.
            - If the module is an instance of nn.Embedding, it sets the data of the weight tensor with random values and
            initializes the padding_idx tensor with zeros.
         _supports_sdpa:
            Retrieves the language_model attribute of the class to check whether the model supports SDPA or not.

    Note:
        Please refer to the PreTrainedModel class for additional inherited methods and attributes.
    """
    config_class = VipLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VipLlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """
        This method '_init_weights' initializes the weights of the provided 'module' based on the specified
        standard deviation.

        Args:
            self:
                An instance of the VipLlavaPreTrainedModel class.

                - Purpose: Represents the current instance of the class.
                - Restrictions: None.

            module:
                The module for which weights are to be initialized.

                - Type: Any valid module object.
                - Purpose: Represents the module whose weights are to be initialized.
                - Restrictions: The module should be a valid PyTorch module.

        Returns:
            None.

        Raises:
            None.
        """
        # important: this ported version of VipLlava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/vipllava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.set_data(Tensor(np.random.normal(
                0.0, std, module.class_embedding.shape), dtype=module.class_embedding.dtype))

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.set_data(Tensor(np.random.normal(
                0.0, std, module.weight.shape), dtype=module.weight.dtype))
            if module.bias is not None:
                module.bias.set_data(ms.common.initializer.initializer(
                    "zeros", module.bias.shape, module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            module.weight.set_data(Tensor(np.random.normal(
                0.0, std, module.weight.shape), dtype=module.weight.dtype))
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = ms.common.initializer.initializer(
                    "zeros", module.weight.data[module.padding_idx].shape, module.weight.dtype)

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


VIPLLAVA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`LlavaProcessor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration with LLAVA->VIPLLAVA,Llava->VipLlava
class VipLlavaForConditionalGeneration(VipLlavaPreTrainedModel):

    """
    This class represents a model for conditional generation using the VipLlava architecture.
    It inherits from VipLlavaPreTrainedModel and provides methods for preparing inputs for generation, forwarding the
    model, and reordering cache.

    Methods:
        forward: Generates output based on input tokens, image features, attention mask, and other optional parameters.
            It returns a tuple or VipLlavaCausalLMOutputWithPast object.
        prepare_inputs_for_generation: Prepares model inputs for generation, considering past key values, inputs embeds,
            pixel values, attention mask, and position ids.
        _reorder_cache: Reorders the cache for the model.

    The class also includes methods for handling input and output embeddings, decoder settings, tying weights,
    and resizing token embeddings.
    """
    def __init__(self, config: VipLlavaConfig):
        """
        Initializes an instance of the VipLlavaForConditionalGeneration class.

        Args:
            self: The instance of the VipLlavaForConditionalGeneration class.
            config (VipLlavaConfig): The configuration object containing settings for the model.
                It is used to initialize the vision tower, multi-modal projector, language model, and other attributes.
                This parameter is mandatory and must be an instance of VipLlavaConfig.

        Returns:
            None.

        Raises:
            ValueError: If the provided config parameter is not an instance of VipLlavaConfig.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = VipLlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the language model used for conditional generation.

        Args:
            self: An instance of the VipLlavaForConditionalGeneration class.

        Returns:
            embeddings: This method returns the input embeddings of the language model used for conditional generation.
                The embeddings are obtained by calling the 'get_input_embeddings()' method of the language model.

        Raises:
            None.
        """
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the VipLlavaForConditionalGeneration language model.

        Args:
            self (VipLlavaForConditionalGeneration): The instance of the VipLlavaForConditionalGeneration class.
            value: The input embeddings to be set for the language model. It should be a tensor of shape
                (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """
        This method returns the output embeddings from the language model for the VipLlavaForConditionalGeneration class.

        Args:
            self: The instance of the VipLlavaForConditionalGeneration class.

        Returns:
            embeddings: This method returns None as it simply retrieves the output embeddings from the language model.

        Raises:
            None.
        """
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the VipLlavaForConditionalGeneration model.

        Args:
            self (VipLlavaForConditionalGeneration): The instance of the VipLlavaForConditionalGeneration class.
            new_embeddings (object): The new embeddings to be set for the model's output.
                It should be compatible with the model's requirements.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of the correct type.
            ValueError: If the new_embeddings parameter does not meet the model's requirements.
        """
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        """
        Sets the decoder for the VipLlavaForConditionalGeneration instance.

        Args:
            self (VipLlavaForConditionalGeneration): The VipLlavaForConditionalGeneration instance.
            decoder: The decoder to be set for the language model. It should be an instance of the decoder class.

        Returns:
            None.

        Raises:
            This method does not raise any exceptions.
        """
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        """
        This method returns the decoder from the language model associated with the VipLlavaForConditionalGeneration class.

        Args:
            self (VipLlavaForConditionalGeneration): The instance of the VipLlavaForConditionalGeneration class.
                It is used to access the language model and retrieve the decoder.

        Returns:
            decoder: This method does not return any value directly. It returns the decoder from the language model.

        Raises:
            This method does not raise any exceptions explicitly. However, exceptions related to accessing the
            language model or retrieving the decoder may be raised indirectly.
        """
        return self.language_model.get_decoder()

    def tie_weights(self):
        """
        Method to tie weights for the VipLlavaForConditionalGeneration class.

        Args:
            self: VipLlavaForConditionalGeneration object.
                Represents an instance of the VipLlavaForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        """
        Resizes the token embeddings of the VipLlavaForConditionalGeneration model.

        Args:
            self (VipLlavaForConditionalGeneration): The instance of the VipLlavaForConditionalGeneration class.
            new_num_tokens (Optional[int]): The new number of tokens for resizing the embeddings. Defaults to None.
            pad_to_multiple_of: The value to pad the embeddings to a multiple of. Defaults to None.

        Returns:
            nn.Embedding: The resized token embeddings as an instance of nn.Embedding.

        Raises:
            None.

        This method resizes the token embeddings of the VipLlavaForConditionalGeneration model based on the provided
        parameters. It first resizes the token embeddings of the underlying language model using the
        'resize_token_embeddings' method. It then updates the 'vocab_size' configuration parameter and the
        'vocab_size' attribute of the model to reflect the new size of the embeddings. Finally, it returns the resized
        token embeddings as an instance of nn.Embedding.
        """
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.vocab_size
        self.vocab_size = model_embeds.vocab_size
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        """
        This method '_merge_input_ids_with_image_features' in the class 'VipLlavaForConditionalGeneration' merges input
        ids with image features to create final embeddings for conditional generation.

        Args:
            self: The instance of the class.
            image_features: Tensor containing image features with shape (num_images, num_image_patches, embed_dim).
            inputs_embeds: Tensor containing embeddings for input tokens.
            input_ids: Tensor containing input token IDs with shape (batch_size, sequence_length).
            attention_mask: Tensor containing attention mask for input tokens.
            labels: Optional tensor containing labels for tokens.

        Returns:
            final_embedding: Tensor containing final embeddings with shape (batch_size, max_embed_dim, embed_dim).
            final_attention_mask: Tensor containing final attention mask with shape (batch_size, max_embed_dim).
            final_labels: Tensor containing final labels with shape (batch_size, max_embed_dim). Returns None if labels
                is None.
            position_ids: Tensor containing position IDs.

        Raises:
            ValueError: If the input provided to the model is incorrect, raising an exception with details on the
                mismatch of image tokens and images given.
        """
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not ops.sum(
            input_ids[:, -1] == Tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = ops.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() *
                         (num_image_patches - 1)).item() + sequence_length
        nonzero = ops.nonzero(input_ids != self.config.image_token_index)
        batch_indices, non_image_indices = ops.tensor_split(nonzero, 2, -1)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = ops.cumsum(
            (special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            # offset for left padding
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices,
                                                non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = ops.zeros(
            (batch_size, int(max_embed_dim), embed_dim), dtype=inputs_embeds.dtype
        )
        final_attention_mask = ops.zeros(
            (batch_size, int(max_embed_dim)), dtype=attention_mask.dtype
        )
        if labels is not None:
            final_labels = ops.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices,
                        text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices,
                             text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices,
                         text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = ops.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(
            -1) - 1 >= nb_image_pad[:, None]

        if image_to_overwrite.sum() != reduce(lambda x, y: x * y, image_features.shape[:-1]):
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {ops.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.reshape(
            -1, embed_dim)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) -
                        1).masked_fill((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        nonzero = ops.nonzero(input_ids == self.pad_token_id)
        batch_indices, pad_indices = ops.tensor_split(nonzero, 2, -1)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        if batch_indices.asnumpy() != []:
            final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    # Ignore copy
    def forward(
        self,
        input_ids: Tensor = None,
        pixel_values: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        vision_feature_layers: Optional[List[int]] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VipLlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple, VipLlavaCausalLMOutputWithPast]

        Example:
            ```python
            >>> import torch
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, VipLlavaForConditionalGeneration
            ...
            >>> model = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", device_map="auto", ms_dtype=torch.float16)
            >>> processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")
            ...
            >>> prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human:<image>\n{}###Assistant:"
            >>> question = "Can you please describe this image?"
            >>> prompt = prompt.format(question)
            >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(text=text, images=image, return_tensors="pt").to(0, torch.float16)
            ...
            >>> # Generate
            >>> generate_ids = model.generate(**inputs, max_new_tokens=20)
            >>> processor.decode(generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            The image features a brown and white cat sitting on a green surface, with a red ball in its
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layers = (
            vision_feature_layers if vision_feature_layers is not None else self.config.vision_feature_layers
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(
                    pixel_values, output_hidden_states=True)
                # For VIP-llava, the image features are computed this way
                # We select the features from index 1: for the layers -2, -5, -8, -11 and 6
                image_features = [image_outputs.hidden_states[index][:, 1:]
                                  for index in vision_feature_layers]
                image_features = ops.cat(image_features, dim=-1)

                image_features = self.multi_modal_projector(image_features)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = ops.full_like(
                        attention_mask, self.config.ignore_index).to(ms.int64)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-1) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    nonzero = ops.nonzero(first_layer_past_key_value.float().sum(-2) == 0)
                    batch_index, non_attended_tokens = ops.tensor_split(nonzero, 2, -1)

                    target_length = input_ids.shape[1]
                    past_length = first_layer_past_key_value.shape[-1]

                    extended_attention_mask = ops.ones(
                        (attention_mask.shape[0], past_length),
                        dtype=attention_mask.dtype,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # in the case one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.shape[-1]
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index,
                                            new_non_attended_tokens] = 0

                    attention_mask = ops.cat(
                        (extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                    position_ids = ops.sum(
                        attention_mask, dim=1).unsqueeze(-1) - 1

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
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0]
                shift_labels = labels[..., 1:][shift_attention_mask != 0]
            else:
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]
                                  ), shift_labels.astype(ms.int32).view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VipLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        """
        Method to prepare inputs for generation in the VipLlavaForConditionalGeneration class.

        Args:
            self: The instance of the class.
            input_ids (Tensor): The input tensor containing token ids of the input sequence.
            past_key_values (Cache or tuple): The cache of key values from previous computations or tuple
                representing past and cache length.
            inputs_embeds (Tensor): An optional tensor containing embeddings for input tokens.
            pixel_values (Tensor): An optional tensor containing pixel values for image inputs.
            attention_mask (Tensor): An optional tensor indicating the attention mask for the input sequence.

        Returns:
            dict: A dictionary containing model inputs for generation, including inputs_embeds, input_ids, position_ids,
                past_key_values, use_cache, attention_mask, and pixel_values.

        Raises:
            TypeError: If past_key_values is not of type Cache or tuple.
            IndexError: If the attention_mask shape is incompatible with input_ids.
            ValueError: If the pixel_values tensor is missing.
            RuntimeError: If there is an issue with calculating position_ids based on attention_mask.
        """
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
                input_ids = input_ids[:, -
                                      (attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1:]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -
                                                (cache_length + input_ids.shape[1]):]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

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
        Method _reorder_cache in class VipLlavaForConditionalGeneration.
        
        Args:
            self: VipLlavaForConditionalGeneration object. The instance of the class.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.language_model._reorder_cache(*args, **kwargs)


__all__ = [
    "VipLlavaCausalLMOutputWithPast",
    "VipLlavaMultiModalProjector",
    "VipLlavaPreTrainedModel",
    "VipLlavaForConditionalGeneration",
]
