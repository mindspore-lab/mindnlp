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
""" MindSpore Llava-NeXT model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import reduce

import mindspore
from mindspore import Tensor, Parameter

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...image_processing_utils import select_best_resolution
from ...modeling_outputs import ModelOutput
from ....utils import logging
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_llava_next import LlavaNextConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaNextConfig"

LLAVA_NEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    # See all LLaVA-NeXT models at https://huggingface.co/models?filter=llava_next
]


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`mindspore.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `mindspore.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding: current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding: current_width - padding]

    return unpadded_tensor


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->LlavaNext
class LlavaNextCausalLMOutputWithPast(ModelOutput):
    """
    Base class for LlavaNext causal language model (or autoregressive) outputs.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True`
            is passed or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True`
            is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(mindspore.Tensor)`, *optional*):
            Tuple of `mindspore.Tensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """
    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[List[Tensor]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None
    image_hidden_states: Optional[Tuple[Tensor]] = None


# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->LlavaNext
class LlavaNextMultiModalProjector(nn.Module):

    """
    This class represents a multi-modal projector for the LlavaNext model.
    It is used to project image features and text embeddings into a shared hidden space.
    
    Inherits from:
        nn.Module
    
    Attributes:
        linear_1 (nn.Linear): A fully connected layer that maps image features to the hidden size specified
            in the configuration.
        act (function): An activation function chosen based on the configuration's specified projector hidden activation.
        linear_2 (nn.Linear): A fully connected layer that maps the hidden states from linear_1 to the hidden size
            specified in the configuration.
    
    Methods:
        forward(image_features):
            Projects the given image features into the shared hidden space by applying the linear transformations
            and activation function.
    
    """
    def __init__(self, config: LlavaNextConfig):
        """
        Initializes an instance of the LlavaNextMultiModalProjector class.
        
        Args:
            self: The instance of the class.
            config (LlavaNextConfig): An object of type LlavaNextConfig containing configuration settings for the projector.
                It is used to set up the linear layers and activation function for the projector.
                
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        """
        Constructs the hidden states for the LlavaNextMultiModalProjector.
        
        Args:
            self (LlavaNextMultiModalProjector): The instance of the LlavaNextMultiModalProjector class.
            image_features (Tensor): The input image features to be processed.
            
        Returns:
            None: This method modifies the hidden_states attribute of the LlavaNextMultiModalProjector instance.
        
        Raises:
            TypeError: If the input image_features is not a Tensor.
            RuntimeError: If an error occurs during the linear transformation or activation function application.
        """
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# Copied from transformers.models.llava.modeling_llava.LlavaPreTrainedModel with Llava->LlavaNext,llava->llava_next
class LlavaNextPreTrainedModel(PreTrainedModel):

    """
    Represents a pre-trained model for the LlavaNext model architecture, inheriting from PreTrainedModel.
    
    This class includes methods for initializing weights based on the configuration settings.
    It initializes weights for different types of cells such as Dense, Conv2d, and Embedding based on the provided
    standard deviation value. The initialization process handles class embeddings, biases, and padding indices as needed.
    """
    config_class = LlavaNextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaNextVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of LlavaNext isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava_next should serve for that purpose
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

class LlavaNextForConditionalGeneration(LlavaNextPreTrainedModel):

    """
    This class represents a model for conditional text generation with multimodal capabilities.
    It is designed to generate text based on input text prompts along with associated images. The model utilizes a
    pre-trained language model for text generation and incorporates image features for enhanced context understanding.

    The class provides methods for setting and getting input embeddings, output embeddings, decoder, and for tying weights.
    It also includes functionality for resizing token embeddings and merging input IDs with image features.
    Additionally, the class offers a 'forward' method for generating text based on input IDs, pixel values,
    attention masks, and other optional parameters. The 'prepare_inputs_for_generation' method prepares input data
    for text generation by handling past key values, inputs embeddings, pixel values, and attention masks.

    This class inherits from LlavaNextPreTrainedModel and is designed to be used for conditional text generation tasks
    in a multimodal setting.
    """
    def __init__(self, config: LlavaNextConfig):
        """Initializes an instance of the LlavaNextForConditionalGeneration class.

        Args:
            self: The instance of the class.
            config (LlavaNextConfig): The configuration object that contains the necessary parameters for
                setting up the instance.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaNextMultiModalProjector(config)

        self.image_newline = Parameter(
            ops.zeros(int(config.text_config.hidden_size)))

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        """
        Returns the input embeddings of the language model used for conditional generation.

        Args:
            self (LlavaNextForConditionalGeneration): The instance of the LlavaNextForConditionalGeneration class.

        Returns:
            The input embeddings of the language model.

        Raises:
            None.
        """
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Method to set input embeddings for the LlavaNextForConditionalGeneration class.

        Args:
            self (LlavaNextForConditionalGeneration): The instance of the LlavaNextForConditionalGeneration class.
            value (object): The input embeddings to be set for the language model.

        Returns:
            None.

        Raises:
            None.
        """
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        """
        Retrieve the output embeddings from the language model for the LlavaNextForConditionalGeneration class.

        Args:
            self: The instance of the LlavaNextForConditionalGeneration class.

        Returns:
            The output embeddings from the language model associated with the LlavaNextForConditionalGeneration instance.

        Raises:
            None.
        """
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the LlavaNextForConditionalGeneration class.

        Args:
            self: An instance of the LlavaNextForConditionalGeneration class.
            new_embeddings: The new embeddings to be set for the language model. 
                It should be of type 'torch.nn.Embedding' or a subclass of it.

        Returns:
            None.

        Raises:
            None.
        """
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        """
        Sets the decoder for the LlavaNextForConditionalGeneration language model.

        Args:
            self (LlavaNextForConditionalGeneration): The instance of the LlavaNextForConditionalGeneration class.
            decoder: The decoder to be set for the language model. 
                It should be compatible with the language model for proper functioning.

        Returns:
            None.

        Raises:
            None.
        """
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder
    def get_decoder(self):
        """
        Retrieve the decoder from the language model for conditional generation.

        Args:
            self (LlavaNextForConditionalGeneration): The instance of the LlavaNextForConditionalGeneration class.
                This parameter is automatically passed when calling the method.

        Returns:
            The decoder obtained from the language model.

        Raises:
            None.
        """
        return self.language_model.get_decoder()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights
    def tie_weights(self):
        """
        Ties the weights of the language model for conditional generation in the LlavaNextForConditionalGeneration class.

        Args:
            self: An instance of the LlavaNextForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.

        This method is responsible for tying the weights of the language model used for conditional generation in the
        LlavaNextForConditionalGeneration class. Tying the weights refers to sharing the parameters of the language
        model with other parts of the model, such as the encoder or the decoder.
        By tying the weights, the model can learn more efficiently and effectively by reducing the number of parameters
        that need to be learned.

        Note:
            This method internally calls the 'tie_weights' method of the language model to perform the weight
            tying operation.
        """
        return self.language_model.tie_weights()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.resize_token_embeddings
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        """
        Resizes the token embeddings for conditional generation in the LlavaNext model.

        Args:
            self (LlavaNextForConditionalGeneration): The instance of the LlavaNextForConditionalGeneration class.
            new_num_tokens (Optional[int]): The desired number of tokens for the resized embeddings. Defaults to None.
            pad_to_multiple_of: (Optional[int]): The value to which the embedding size should be padded. Defaults to None.

        Returns:
            nn.Embedding: The resized token embeddings of type nn.Embedding.

        Raises:
            None.
        """
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._merge_input_ids_with_image_features
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        """
        Merges image features with input embeddings, input IDs, attention masks, and labels.

        Args:
            self (LlavaNextForConditionalGeneration): The object instance.
            image_features (Tensor): A tensor containing image features.
            inputs_embeds (Tensor): A tensor containing input embeddings.
            input_ids (Tensor): A tensor containing input IDs.
            attention_mask (Tensor): A tensor containing attention masks.
            labels (Tensor): A tensor containing labels.

        Returns:
            None.

        Raises:
            ValueError: If the number of image tokens provided to the model does not match the number of images given.
        """
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not ops.sum(input_ids[:, -1] == mindspore.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = ops.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)).item() + sequence_length
        batch_indices, non_image_indices = ops.nonzero(
            input_ids != self.config.image_token_index, as_tuple=True)

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
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        if batch_indices.asnumpy() != []:
            final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        pixel_values: mindspore.Tensor = None,
        image_sizes: Optional[mindspore.Tensor] = None,
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
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple, LlavaNextCausalLMOutputWithPast]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration
            ...
            >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
            >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
            ...
            >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
            >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(**inputs, max_length=30)
            >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
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
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                batch_size, num_patches, num_channels, height, width = pixel_values.shape
                reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)
                image_features = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)

                selected_image_feature = image_features.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]

                image_features = self.multi_modal_projector(selected_image_feature)

                # split up image_features for each of the individual images
                # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
                # if we assume each image has 5 image features (base image + 4 patches)
                split_sizes = [image.shape[0] for image in pixel_values]
                image_features = ops.split(image_features, split_sizes, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                        if height * width != base_image_feature.shape[0]:
                            raise ValueError("The number of patches is not consistent with the image size.")
                        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                            image_sizes[image_idx],
                            self.config.image_grid_pinpoints,
                            self.config.vision_config.image_size,
                        )
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        image_feature = image_feature.permute(4, 0, 2, 1, 3)
                        image_feature = image_feature.flatten(start_dim=1, end_dim=2).flatten(start_dim=2, end_dim=3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = ops.cat(
                            (
                                image_feature,
                                self.image_newline[:, None, None].broadcast_to((*image_feature.shape[:-1], 1)),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(start_dim=1, end_dim=2).swapaxes(0, 1)
                        image_feature = ops.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        image_feature = ops.cat((image_feature, self.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                image_features = ops.stack(new_image_features, dim=0)

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
                batch_index, non_attended_tokens = ops.nonzero(first_layer_past_key_value.float().sum(-2) == 0, as_tuple=True)

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
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
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
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0]
                shift_labels = labels[..., 1:][shift_attention_mask != 0]
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

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Prepare the inputs for text generation.

        Args:
            self (LlavaNextForConditionalGeneration): The instance of the LlavaNextForConditionalGeneration class.
            input_ids (Tensor): The input token IDs tensor for text generation.
            past_key_values (Cache or tuple of Tensors): The cached key values from previous generation steps.
                If Cache object is passed, cache_length is obtained from it, else from the tuple of Tensors.
                Defaults to None.
            inputs_embeds (Tensor): The input embeddings tensor. Defaults to None.
            pixel_values (Tensor): The pixel values tensor for image inputs. Defaults to None.
            image_sizes (Tensor): The sizes of the input images. Defaults to None.
            attention_mask (Tensor): The attention mask tensor to mask certain tokens during generation. Defaults to None.

        Returns:
            model_inputs (dict): A dictionary containing the model inputs for text generation, including 'inputs_embeds',
                'input_ids', 'position_ids', 'past_key_values', 'use_cache', 'attention_mask', 'pixel_values',
                and 'image_sizes'.

        Raises:
            TypeError: If past_key_values is not of type Cache or tuple of Tensors.
            IndexError: If the attention_mask shape is not compatible with input_ids shape.
            ValueError: If there are inconsistencies in handling input token IDs based on cache and attention mask lengths.
            AttributeError: If the image token index is missing in the input_ids.
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
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
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
                "image_sizes": image_sizes,
            }
        )
        return model_inputs

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._reorder_cache
    def _reorder_cache(self, *args, **kwargs):
        """
        Reorders the cache for the language model.
        
        Args:
            self:
                The instance of the LlavaNextForConditionalGeneration class.

                - Type: LlavaNextForConditionalGeneration
                - Purpose: Represents the current instance of the class.
                - Restrictions: This parameter is required and should be the first positional argument.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.language_model._reorder_cache(*args, **kwargs)


__all__ = [
    "LlavaNextPreTrainedModel",
    "LlavaNextForConditionalGeneration",
    "LlavaNextCausalLMOutputWithPast",
    "LlavaNextMultiModalProjector",
]
