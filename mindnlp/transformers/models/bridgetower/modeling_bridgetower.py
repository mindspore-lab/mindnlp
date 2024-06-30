# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""MindSpore BridgeTower Model"""
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Normal

from mindnlp.modules.functional import normalize
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer, apply_chunking_to_forward
from ....utils import logging
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BridgeTowerConfig"
_CHECKPOINT_FOR_DOC = "BridgeTower/bridgetower-base"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"


@dataclass
class BridgeTowerModelOutput(ModelOutput):
    """
    Output type of [`BridgeTowerModel`].

    Args:
        text_features (`mindspore.Tensor` of shape `(batch_size, text_sequence_length, hidden_size)`):
            Sequence of hidden-states at the text output of the last layer of the model.
        image_features (`mindspore.Tensor` of shape `(batch_size, image_sequence_length, hidden_size)`):
            Sequence of hidden-states at the image output of the last layer of the model.
        pooler_output (`mindspore.Tensor` of shape `(batch_size, hidden_size x 2)`):
            Concatenation of last layer hidden-state of the first token of the text and image sequence (classification
            token), respectively, after further processing through layers used for auxiliary pretraining tasks.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    text_features: mindspore.Tensor = None
    image_features: mindspore.Tensor = None
    pooler_output: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class BridgeTowerContrastiveOutput(ModelOutput):
    """
    Output type of ['BridgeTowerForContrastiveLearning']

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`:
            Image-text contrastive loss.
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        text_embeds (`mindspore.Tensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        image_embeds (`mindspore.Tensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        cross_embeds  (`mindspore.Tensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text-image cross-modal embeddings obtained by applying the projection layer to the pooler_output.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    text_embeds: Optional[Tuple[mindspore.Tensor]] = None
    image_embeds: Optional[Tuple[mindspore.Tensor]] = None
    cross_embeds: Optional[Tuple[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class BridgeTowerResidualAttention(nn.Cell):

    """
    This class represents a bridge tower residual attention module in a neural network model. It is a subclass of nn.Cell and is used to perform attention calculations on input hidden states.
    
    Attributes:
        attn (nn.MultiheadAttention): A multi-head attention module that performs attention calculations on hidden states.
        ln_1 (nn.LayerNorm): A layer normalization module that normalizes the hidden states after the attention calculation.
        mlp (nn.CellDict): A dictionary containing the modules for the multi-layer perceptron (MLP) used in the attention module.
        ln_2 (nn.LayerNorm): A layer normalization module that normalizes the hidden states after the MLP operations.
        attn_mask (None or mindspore.Tensor): A tensor that represents the attention mask used during the attention calculation.
    
    Methods:
        __init__(self, config):
            Initializes a new instance of the BridgeTowerResidualAttention class.

            Args:

            - config: A configuration object that contains the necessary parameters for the attention module.

        attention(self, hidden_state: mindspore.Tensor, attention_mask: mindspore.Tensor):
            Performs the attention calculation on the given hidden state.

            Args:

            - hidden_state (mindspore.Tensor): The input hidden state on which the attention calculation is performed.
            - attention_mask (mindspore.Tensor): An optional tensor representing the attention mask.

            Returns:

            - mindspore.Tensor: The output hidden state after performing the attention calculation.

        construct(self, hidden_state: mindspore.Tensor, attention_mask: mindspore.Tensor = None):
            Constructs the bridge tower residual attention module by applying attention and MLP operations on the
            input hidden state.

            Args:

            - hidden_state (mindspore.Tensor): The input hidden state on which the attention and MLP operations are applied.
            - attention_mask (mindspore.Tensor): An optional tensor representing the attention mask.

            Returns:

            - mindspore.Tensor: The output hidden state after applying the attention and MLP operations.
    """
    def __init__(self, config):
        """Initialize the BridgeTowerResidualAttention class.

        Args:
            self: The instance of the class.
            config: An object containing configuration parameters for the model.
                It should have attributes like hidden_size, layer_norm_eps, etc.

                - Type: object
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(config.hidden_size, config.hidden_size // 64)
        self.ln_1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = nn.CellDict(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(config.hidden_size, config.hidden_size * 4)),
                    ("gelu", QuickGELUActivation()),
                    ("c_proj", nn.Dense(config.hidden_size * 4, config.hidden_size)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attn_mask = None

    def attention(self, hidden_state: mindspore.Tensor, attention_mask: mindspore.Tensor):
        """
        This method calculates the attention mechanism for the BridgeTowerResidualAttention module.

        Args:
            self: The instance of the BridgeTowerResidualAttention class.
            hidden_state (mindspore.Tensor): The input tensor representing the hidden state for attention calculation.
            attention_mask (mindspore.Tensor): An optional tensor used for masking the attention weights.
                If provided, it should have the same shape as hidden_state.
                It should be of type mindspore.Tensor and can be None.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=mindspore.bool_)
        self.attn_mask = (
            self.attn_mask.to(dtype=hidden_state.dtype)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            hidden_state,
            hidden_state,
            hidden_state,
            need_weights=False,
            attn_mask=self.attn_mask,
            key_padding_mask=attention_mask,
        )[0]

    def construct(self, hidden_state: mindspore.Tensor, attention_mask: mindspore.Tensor = None):
        """
        Method to construct the output of the BridgeTowerResidualAttention model.

        Args:
            self: An instance of the BridgeTowerResidualAttention class.
            hidden_state (mindspore.Tensor): The hidden state tensor to be processed.
            attention_mask (mindspore.Tensor, optional): A tensor representing the attention mask. Default is None.

        Returns:
            mindspore.Tensor: The updated hidden state tensor after processing.

        Raises:
            None.
        """
        residual_state = hidden_state + self.attention(self.ln_1(hidden_state), attention_mask)
        hidden_state = self.ln_2(residual_state)
        for _, layer in self.mlp.items():
            hidden_state = layer(hidden_state)
        hidden_state = residual_state + hidden_state
        return hidden_state


class BridgeTowerTransformer(nn.Cell):

    """
    A class representing a BridgeTowerTransformer, a type of transformer model with customizable hidden layers and attention mechanisms.

    This class inherits from nn.Cell and can be used to construct a transformer model with BridgeTowerResidualAttention blocks.

    Attributes:
        hidden_size (int): The size of the hidden layers in the transformer.
        num_hidden_layers (int): The number of hidden layers in the transformer.
        resblocks (nn.CellList): A list of BridgeTowerResidualAttention blocks used in the transformer.
        stop_gradient (bool): A flag indicating whether to use stop gradient during training.

    Methods:
        __init__(config): Initializes the BridgeTowerTransformer with the given configuration.
        construct(hidden_state, attention_mask): Constructs the transformer by applying the
            BridgeTowerResidualAttention blocks to the hidden state.

    Example:
        ```python
        >>> config = TransformerConfig(hidden_size=512, num_hidden_layers=6, remove_last_layer=False, stop_gradient=True)
        >>> transformer = BridgeTowerTransformer(config)
        >>> hidden_states = transformer.construct(hidden_state, attention_mask)
        ```
    """
    def __init__(self, config):
        """
        This method initializes the BridgeTowerTransformer class with the provided configuration.

        Args:
            self (object): The instance of the BridgeTowerTransformer class.
            config (object):
                An object containing configuration parameters for the BridgeTowerTransformer.

                - hidden_size (int): The size of the hidden layers.
                - num_hidden_layers (int): The number of hidden layers.
                - remove_last_layer (bool): A flag indicating whether to remove the last layer.
                - stop_gradient (bool): A flag indicating whether to stop gradient computation.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter contains invalid values or is missing required attributes.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        if config.remove_last_layer:
            self.resblocks = nn.CellList(
                [BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers - 1)]
            )
        else:
            self.resblocks = nn.CellList(
                [BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers)]
            )
        self.stop_gradient = config.stop_gradient

    def construct(self, hidden_state: mindspore.Tensor, attention_mask: Optional[mindspore.Tensor] = None):
        """
        Constructs the BridgeTowerTransformer model.

        Args:
            self (BridgeTowerTransformer): The instance of the BridgeTowerTransformer class.
            hidden_state (mindspore.Tensor): The input hidden state tensor for the transformer.
            attention_mask (Optional[mindspore.Tensor], optional): An optional tensor for attention mask. Defaults to None.

        Returns:
            List[mindspore.Tensor]: A list of hidden states after passing through the transformer blocks.

        Raises:
            None
        """
        hidden_states = []
        for block in self.resblocks:
            hidden_state = block(hidden_state, attention_mask)
            if self.stop_gradient:
                hidden_states.append(ops.stop_gradient(hidden_state))
            else:
                hidden_states.append(hidden_state)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings with CLIP->BridgeTower
class BridgeTowerVisionEmbeddings(nn.Cell):

    """
    BridgeTowerVisionEmbeddings class represents a module for generating embeddings for vision tasks using the BridgeTower architecture.

    This class inherits from nn.Cell and is responsible for constructing embeddings for input pixel values based on the provided configuration.

    Attributes:
        config (BridgeTowerVisionConfig): The configuration object containing parameters for the vision model.
        embed_dim (int): The dimension of the embeddings.
        image_size (int): The size of the input image.
        patch_size (int): The size of the patches used for processing the image.
        class_embedding (Parameter): Learnable class embedding vector.
        patch_embedding (Conv2d): Convolutional layer for generating patch embeddings.
        num_patches (int): Number of patches in the image.
        num_positions (int): Total number of positions for embeddings.
        position_embedding (Embedding): Embedding layer for positional encodings.
        position_ids (Tensor): Tensor containing position indices for embeddings.

    Methods:
        construct: Constructs the embeddings for the input pixel values.

            Args:

            - pixel_values (Tensor): Input tensor containing pixel values.

            Returns:

            - Tensor: Output embeddings for the input pixel values.
    """
    def __init__(self, config: BridgeTowerVisionConfig):
        """
        Initializes an instance of the BridgeTowerVisionEmbeddings class.

        Args:
            self (object): The instance of the class.
            config (BridgeTowerVisionConfig):
                An object of BridgeTowerVisionConfig class containing configuration parameters.

                - config.hidden_size (int): The size of the hidden dimension.
                - config.image_size (int): The size of the input image.
                - config.patch_size (int): The size of the image patch.
                - config.num_channels (int): The number of input channels.

        Returns:
            None.

        Raises:
            None.
        """
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
            pad_mode='valid'
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = ops.arange(self.num_positions).expand((1, -1))

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        construct method in the BridgeTowerVisionEmbeddings class.

        This method takes two parameters: self and pixel_values.

        Args:
            self: BridgeTowerVisionEmbeddings object. The instance of the class.
            pixel_values: mindspore.Tensor. A tensor containing the pixel values.

        Returns:
            mindspore.Tensor. A tensor representing the constructed embeddings.

        Raises:
            None.
        """
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class BridgeTowerVisionTransformer(nn.Cell):

    """
    This class represents a Vision Transformer for processing pixel values in the context of BridgeTower vision tasks.
    It inherits from the nn.Cell class.

    Attributes:
        embeddings (BridgeTowerVisionEmbeddings): An instance of the BridgeTowerVisionEmbeddings class,
            responsible for converting pixel values into embedded representations.
        ln_pre (nn.LayerNorm): A LayerNorm module that normalizes the hidden states before the transformer layers.
        transformer (BridgeTowerTransformer): An instance of the BridgeTowerTransformer class,
            responsible for performing the transformer operations on the hidden states.
        ln_post (nn.LayerNorm): A LayerNorm module that normalizes the hidden states after the transformer layers.
        share_layernorm (bool): A flag indicating whether to share the LayerNorm module across transformer layers.
        ln_separate (nn.CellList): A list of LayerNorm modules for separate normalization of
            hidden states in each transformer layer.

    Methods:
        construct(pixel_values: mindspore.Tensor, attention_mask):
            Constructs the forward pass of the BridgeTowerVisionTransformer.

            Args:

            - pixel_values (mindspore.Tensor): A tensor containing the pixel values.
            - attention_mask: A tensor representing the attention mask.

            Returns:

            - hidden_states (mindspore.Tensor): A tensor containing the processed hidden states.

        construct_pre(pixel_values: mindspore.Tensor):
            Constructs the forward pass of the BridgeTowerVisionTransformer up to the layer normalization
            before the transformer layers.

            Args:

            - pixel_values (mindspore.Tensor): A tensor containing the pixel values.

            Returns:

            - hidden_states (mindspore.Tensor): A tensor containing the processed hidden states.

        construct_post(hidden_state: mindspore.Tensor):
            Constructs the forward pass of the BridgeTowerVisionTransformer after the transformer layers up to the final layer normalization.

            Args:

            - hidden_state (mindspore.Tensor): A tensor containing the hidden states.

            Returns:

            - visual_output_post (mindspore.Tensor): A tensor containing the processed visual output.

    Note:
        The BridgeTowerVisionTransformer class is designed for BridgeTower vision tasks and provides flexibility in
        layer normalization. It can be used to process pixel values and generate visual representations.
    """
    def __init__(self, config):
        """
        Initializes the BridgeTowerVisionTransformer class.

        Args:
            self: The instance of the class.
            config:
                A dictionary containing the configuration parameters for the transformer.

                - Type: dict
                - Purpose: It holds the configuration parameters required for initializing the transformer.
                - Restrictions: Must be a valid dictionary.

        Returns:
            None.

        Raises:
            ValueError: If the provided 'config' parameter is invalid or missing required fields.
            TypeError: If the data type of the 'config' parameter is not a dictionary.
            RuntimeError: If any runtime error occurs during the initialization process.
        """
        super().__init__()

        self.embeddings = BridgeTowerVisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.transformer = BridgeTowerTransformer(config)
        self.ln_post = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.share_layernorm = config.share_layernorm
        if not config.share_layernorm:
            self.ln_separate = nn.CellList(
                [nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps) for _ in range(config.num_hidden_layers)]
            )

    def construct(self, pixel_values: mindspore.Tensor, attention_mask):
        """
        Constructs the BridgeTowerVisionTransformer.

        Args:
            self: The instance of the BridgeTowerVisionTransformer class.
            pixel_values (mindspore.Tensor): The input pixel values for the transformer. It should be a tensor of shape [B, H, W, C],
                where B is the batch size, H and W are the height and width of the input image, and C is the number of channels.
            attention_mask: The attention mask for the transformer. It can be a tensor of shape [B, S, S] or None, where B is the
                batch size and S is the sequence length. If provided, the attention mask will be applied to the transformer
                self-attention module.

        Returns:
            mindspore.Tensor: The constructed hidden states of the BridgeTowerVisionTransformer. It will be a tensor of shape
                [B, C, H, W], where B is the batch size, C is the number of channels, and H and W are the height and width of the
                transformed image.

        Raises:
            None.
        """
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.ln_pre(hidden_states)
        # NLD -> LND
        hidden_states = hidden_states.permute(1, 0, 2)

        hidden_states = self.transformer(hidden_states, attention_mask)
        # shape = [num_hidden_layers, hidden_size, *, grid ** 2]
        hidden_states = ops.stack(hidden_states, axis=0)
        # shape = [num_hidden_layers, *, hidden_size, grid ** 2]
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        if self.share_layernorm:
            hidden_states = self.ln_post(hidden_states)
        else:
            hidden_states_stack = []
            for hidden_states, ln in zip(hidden_states, self.ln_separate):
                hidden_states = ln(hidden_states)
                hidden_states_stack.append(hidden_states)
            # shape = [num_hidden_layers, *, hidden_size, grid ** 2]
            hidden_states = ops.stack(hidden_states_stack, axis=0)
        return hidden_states

    def construct_pre(self, pixel_values: mindspore.Tensor):
        """
        Constructs the pre-processed hidden states for the BridgeTowerVisionTransformer model.

        Args:
            self (BridgeTowerVisionTransformer): An instance of the BridgeTowerVisionTransformer class.
            pixel_values (mindspore.Tensor): A tensor containing pixel values of the input images. The shape of the tensor is
                expected to be (batch_size, sequence_length, num_channels, image_height, image_width).

        Returns:
            mindspore.Tensor: A tensor representing the pre-processed hidden states. The shape of the tensor is
                (sequence_length, batch_size, hidden_size).

        Raises:
            None.

        This method takes in the pixel values of input images and performs the following steps to construct the
        pre-processed hidden states:

        1. Passes the pixel values through the 'embeddings' layer to obtain the initial hidden states.
        2. Applies layer normalization ('ln_pre') to the hidden states.
        3. Permutes the dimensions of the hidden states to match the expected shape.
        4. Returns the pre-processed hidden states.
        Note that the input images are expected to be in the format (batch_size, sequence_length, num_channels, image_height, image_width).
        The returned tensor represents the pre-processed hidden states and has the shape (sequence_length, batch_size, hidden_size).
        """
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.ln_pre(hidden_states)
        # NLD -> LND
        hidden_states = hidden_states.permute(1, 0, 2)
        return hidden_states

    def construct_post(self, hidden_state: mindspore.Tensor):
        """
        Constructs the post-processed visual output based on the given hidden state.

        Args:
            self (BridgeTowerVisionTransformer): The instance of the BridgeTowerVisionTransformer class.
            hidden_state (mindspore.Tensor): The hidden state tensor representing the visual input.
                It is expected to have the shape (sequence_length, batch_size, hidden_size).

        Returns:
            None: The method modifies the visual output in place.

        Raises:
            None.
        """
        visual_output_post = hidden_state.permute(1, 0, 2)
        visual_output_post = self.ln_post(visual_output_post)
        return visual_output_post


class BridgeTowerLinkTower(nn.Cell):

    """
    This class represents a BridgeTowerLinkTower, which is a component used in a neural network model for
    linking towers in a bridge tower architecture. It inherits from the nn.Cell class.

    Attributes:
        link_tower_type (str): The type of link tower to be used. It can be one of ['add', 'scaled_add', 'interpolate'].
        hidden_size (int): The size of the hidden states.
        scaled_factor (mindspore.Parameter): The scaling factor used in the 'scaled_add' link tower type.
        beta (mindspore.Parameter): The interpolation factor used in the 'interpolate' link tower type.
        LayerNorm (mindspore.nn.LayerNorm): The layer normalization module.

    Raises:
        NotImplementedError: If the specified link tower type is not implemented.

    Methods:
        __init__(self, config):
            Initializes a new instance of the BridgeTowerLinkTower class.

        construct(self, hidden_states, cross_modal_hidden_states, attention_mask):
            Constructs the link tower based on the specified link tower type and input hidden states.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BridgeTowerLinkTower class.

        Args:
            self: The instance of the class.
            config (Config):
                An object containing the configuration parameters for the link tower.

                - link_tower_type (str): The type of link tower. Possible values are 'add', 'scaled_add', or 'interpolate'.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for LayerNorm.

        Returns:
            None

        Raises:
            NotImplementedError: If the specified link_tower_type is not implemented.
        """
        super().__init__()
        self.link_tower_type = config.link_tower_type
        self.hidden_size = config.hidden_size
        if config.link_tower_type in ["add", "scaled_add", "interpolate"]:
            if config.link_tower_type == "scaled_add":
                self.scaled_factor = Parameter(mindspore.tensor(1.0))
            elif config.link_tower_type == "interpolate":
                self.beta = Parameter(mindspore.tensor(0.5))
            self.LayerNorm = nn.LayerNorm(self.hidden_size, epsilon=config.layer_norm_eps)
        else:
            raise NotImplementedError(f"link_tower_type {config.link_tower_type} is not implemented")

    def construct(self, hidden_states, cross_modal_hidden_states, attention_mask):
        """
        Constructs a link tower for the BridgeTowerLinkTower class.

        This method takes in four parameters: self, hidden_states, cross_modal_hidden_states, and attention_mask. It returns None.

        Args:
            self: The instance of the BridgeTowerLinkTower class.
            hidden_states (Tensor): The hidden states.
            cross_modal_hidden_states (Tensor): The hidden states from the cross-modal network.
            attention_mask (Tensor): The attention mask.

        Returns:
            None.

        Raises:
            NotImplementedError: If the link_tower_type specified is not implemented.

        """
        if self.link_tower_type == "add":
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        elif self.link_tower_type == "scaled_add":
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        elif self.link_tower_type == "interpolate":
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        else:
            raise NotImplementedError(f"link_tower_type {self.link_tower_type} is not implemented")


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->BridgeTower
class BridgeTowerSelfOutput(nn.Cell):

    """
    The 'BridgeTowerSelfOutput' class represents a neural network cell for self-output in a bridge tower architecture.
    This class inherits from nn.Cell and contains methods for initializing the cell and constructing the self-output operation.

    Attributes:
        dense (nn.Dense): A fully connected layer for transforming hidden states.
        LayerNorm (nn.LayerNorm): A layer for normalizing hidden states.
        dropout (nn.Dropout): A layer for applying dropout to hidden states.

    Methods:
        __init__: Initializes the BridgeTowerSelfOutput cell with the given configuration.
        construct: Constructs the self-output operation using the given hidden states and input tensor.
    """
    def __init__(self, config):
        """
        Initializes the BridgeTowerSelfOutput class.

        Args:
            self: An instance of the BridgeTowerSelfOutput class.
            config:
                A configuration object containing parameters for initializing the BridgeTowerSelfOutput.

                - Type: Any
                - Purpose: Specifies the configuration settings for the BridgeTowerSelfOutput initialization.
                - Restrictions: Must contain the necessary parameters for configuring the BridgeTowerSelfOutput.

        Returns:
            None.

        Raises:
            ValueError: If the configuration object 'config' is not provided or is invalid.
            TypeError: If the provided configuration object 'config' is not of the expected type.
            AttributeError: If there are issues with accessing or setting attributes during initialization.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output of the BridgeTowerSelfOutput layer.

        Args:
            self (BridgeTowerSelfOutput): An instance of the BridgeTowerSelfOutput class.
            hidden_states (mindspore.Tensor):
                The hidden states tensor.

                - Shape: (batch_size, seq_length, hidden_size)
                - Purpose: Represents the input hidden states to the layer.
                - Restrictions: None
            input_tensor (mindspore.Tensor):
                The input tensor.

                - Shape: (batch_size, seq_length, hidden_size)
                - Purpose: Represents the input tensor to be added to the hidden states.
                - Restrictions: None

        Returns:
            mindspore.Tensor:
                The constructed output tensor.

                - Shape: (batch_size, seq_length, hidden_size)
                - Purpose: Represents the output of the BridgeTowerSelfOutput layer.

        Raises:
            None: This method does not raise any exceptions.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->BridgeTower
class BridgeTowerIntermediate(nn.Cell):

    """
    This class represents a bridge tower intermediate module, which is a part of a neural network model. It is a subclass of the nn.Cell class.

    Attributes:
        dense (nn.Dense): A fully connected layer used for linear transformation of the input tensor.
        intermediate_act_fn (function): Activation function applied to the output of the dense layer.

    Methods:
        __init__: Initializes the BridgeTowerIntermediate instance.
        construct: Constructs the bridge tower intermediate module.

    """
    def __init__(self, config):
        """
        Initializes an instance of the 'BridgeTowerIntermediate' class.

        Args:
            self: The instance of the 'BridgeTowerIntermediate' class.
            config:
                An object representing the configuration settings for the 'BridgeTowerIntermediate' class.

                - Type: object
                - Purpose: Stores the configuration settings required for initialization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the intermediate hidden states in the BridgeTowerIntermediate class.

        Args:
            self: The instance of the BridgeTowerIntermediate class.
            hidden_states (mindspore.Tensor): The input tensor containing the hidden states to be processed.
                It should be of type mindspore.Tensor.

        Returns:
            mindspore.Tensor:
                The processed hidden states after passing through the dense layer and intermediate activation function.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BridgeTower
class BridgeTowerOutput(nn.Cell):

    """
    Represents the output layer of a bridge tower neural network model.

    This class inherits from nn.Cell and implements the output layer operations including dense transformation,
    dropout, layer normalization, and residual connection.

    The BridgeTowerOutput class provides the construct method for applying the output layer operations to the
    input hidden states and input tensor, and returns the transformed hidden states.

    Attributes:
        dense (nn.Dense): The dense transformation module with configurable intermediate and hidden sizes.
        LayerNorm (nn.LayerNorm): The layer normalization module with configurable hidden size and epsilon.
        dropout (nn.Dropout): The dropout module with configurable dropout probability.

    Methods:
        construct:
            Applies dense transformation, dropout, layer normalization, and residual connection to the
            input hidden states and input tensor, and returns the transformed hidden states.
    """
    def __init__(self, config):
        """
        __init__ method in the BridgeTowerOutput class.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters.

        Returns:
            None.

        Raises:
            ValueError: If the config parameters are invalid or missing.
            TypeError: If the config parameters are of incorrect type.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the BridgeTowerOutput.

        Args:
            self (BridgeTowerOutput): An instance of the BridgeTowerOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor.
                It should have the shape (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor.
                It should have the same shape as `hidden_states`.

        Returns:
            mindspore.Tensor: The output tensor after applying the BridgeTowerOutput.
                It has the same shape as `hidden_states`.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->BridgeTower
class BridgeTowerPooler(nn.Cell):

    """
    The 'BridgeTowerPooler' class represents a pooler module for the Bridge Tower model in mindspore.
    It is responsible for computing the pooled output of the first token tensor of the input hidden states.

    This class inherits from the 'nn.Cell' base class.

    Attributes:
        dense (nn.Dense): A fully connected layer used to transform the first token tensor.
        activation (nn.Tanh): An activation function applied to the pooled output.

    Methods:
        __init__(self, config):
            Initializes a new instance of the 'BridgeTowerPooler' class.

            Args:

            - config (object): An object that contains the configuration parameters for the pooler.

        construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
            Constructs the pooler module by computing the pooled output of the first token tensor.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states tensor.

            Returns:

            - mindspore.Tensor: The computed pooled output tensor.
    """
    def __init__(self, config):
        """
        Initializes the BridgeTowerPooler class.

        Args:
            self: The object instance.
            config:
                An object containing configuration parameters for the BridgeTowerPooler.

                - Type: object
                - Purpose: To configure the BridgeTowerPooler with specified parameters.
                - Restrictions: Must be a valid config object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the pooled output tensor for the BridgeTowerPooler model.

        Args:
            self (BridgeTowerPooler): An instance of the BridgeTowerPooler class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states of the input sequence.

        Returns:
            mindspore.Tensor: The pooled output tensor.

        Raises:
            None.

        Description:
            This method takes the hidden states tensor and constructs the pooled output tensor for the BridgeTowerPooler model.
            The hidden_states tensor should have a shape of (batch_size, sequence_length, hidden_size) and represents
            the hidden states of the input sequence.
            The method extracts the first token tensor from the hidden_states tensor by selecting the first element
            of each sequence in the batch, resulting in a tensor of shape (batch_size, hidden_size).
            Then, the first token tensor is passed through a dense layer, followed by an activation function.
            The resulting tensor is the pooled output tensor, which is returned as the output of this method.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->BridgeTower
class BridgeTowerSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism used in the BridgeTower model. It is a subclass of nn.Cell.

    Attributes:
        num_attention_heads (int): The number of attention heads in the self-attention mechanism.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): The total size of all attention heads.
        query (nn.Dense): The dense layer for computing the query representation.
        key (nn.Dense): The dense layer for computing the key representation.
        value (nn.Dense): The dense layer for computing the value representation.
        dropout (nn.Dropout): The dropout layer for attention probabilities.
        position_embedding_type (str): The type of position embedding used in the attention mechanism.
        distance_embedding (nn.Embedding): The embedding layer for computing relative positional embeddings.
        is_decoder (bool): Whether the self-attention mechanism is used in a decoder layer.

    Methods:
        swapaxes_for_scores:
            Reshapes the input tensor to prepare it for computing attention scores.

        construct:
            Computes the self-attention mechanism given the input and optional arguments.

    Note:
        - The hidden size must be a multiple of the number of attention heads.
        - The attention mechanism supports different types of positional embeddings.
        - The attention mechanism can be used in both encoder and decoder layers of the model.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initialize the BridgeTowerSelfAttention class with the provided configuration.

        Args:
            self: The instance of the class.
            config (object): An object containing configuration parameters.
                Required parameters:

                - hidden_size (int): The size of the hidden layers.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (int): The size of the embedding.
                - attention_probs_dropout_prob (float): The dropout probability for attention weights.
                - max_position_embeddings (int): The maximum number of position embeddings.
                - is_decoder (bool): Indicates if the model is a decoder.
            position_embedding_type (str, optional): The type of position embedding to use. Defaults to 'absolute'.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def swapaxes_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method swaps and reshapes the input tensor for self-attention scores calculation.

        Args:
            self (BridgeTowerSelfAttention): The instance of BridgeTowerSelfAttention class.
            x (mindspore.Tensor): The input tensor containing the scores to be reshaped and permuted.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The reshaped and permuted tensor containing the self-attention scores.
                It has a shape of (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            ValueError: If the input tensor 'x' does not have the expected shape (batch_size, sequence_length, hidden_size).
            RuntimeError: If there is an issue with reshaping or permuting the input tensor.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        '''
        Constructs the self-attention mechanism in the BridgeTowerSelfAttention class.

        Args:
            self (BridgeTowerSelfAttention): An instance of the BridgeTowerSelfAttention class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size)
                representing the hidden states.
            attention_mask (Optional[mindspore.Tensor]): An optional input tensor of shape (batch_size, sequence_length)
                representing the attention mask. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): An optional input tensor of shape (num_heads,)
                representing the head mask. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional input tensor of shape
                (batch_size, encoder_sequence_length, hidden_size) representing the hidden states of the encoder.
                Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional input tensor of shape
                (batch_size, encoder_sequence_length) representing the attention mask of the encoder. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                An optional tuple containing the past key and value tensors. Defaults to None.
            output_attentions (Optional[bool]): An optional flag indicating whether to output attention probabilities.
                Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the context layer tensor of shape (batch_size, sequence_length, hidden_size)
                and optionally attention probabilities tensor of shape (batch_size, num_heads, sequence_length,
                encoder_sequence_length) if output_attentions is True.

        Raises:
            None.
        '''
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.swapaxes_for_scores(self.key(encoder_hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))
            key_layer = ops.cat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.cat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = mindspore.tensor(key_length - 1, dtype=mindspore.int64).view(
                    -1, 1
                )
            else:
                position_ids_l = ops.arange(query_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(key_length, dtype=mindspore.int64).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = ops.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BridgeTowerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->BridgeTower
class BridgeTowerAttention(nn.Cell):

    """
    This class represents the BridgeTowerAttention module, which is used for attention mechanism in the BridgeTower model.
    It is a subclass of the nn.Cell class.

    Attributes:
        self (BridgeTowerSelfAttention): The self-attention layer of the BridgeTowerAttention module.
        output (BridgeTowerSelfOutput): The output layer of the BridgeTowerAttention module.
        pruned_heads (set): A set containing the indices of the pruned attention heads.

    Methods:
        __init__(self, config, position_embedding_type=None):
            Initializes a new instance of the BridgeTowerAttention module.

            Args:

            - config: The configuration for the module.
            - position_embedding_type: The type of position embedding to be used. (optional)

        prune_heads(self, heads):
            Prunes specified attention heads from the module.

            Args:

            - heads: A list of attention head indices to be pruned.

        construct(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            Performs the forward pass of the BridgeTowerAttention module.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states.
            - attention_mask (Optional[mindspore.Tensor]): The attention mask. (optional)
            - head_mask (Optional[mindspore.Tensor]): The head mask. (optional)
            - encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder. (optional)
            - encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder. (optional)
            - past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key-value tensors. (optional)
            - output_attentions (Optional[bool]): Whether to output attentions. (optional)

            Returns:
                Tuple[mindspore.Tensor]: The output tensor of the BridgeTowerAttention module.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the BridgeTowerAttention class.

        Args:
            self: The instance of the class (automatically passed).
            config: A configuration object containing various settings and parameters (type: object).
            position_embedding_type: An optional parameter specifying the type of position embedding (type: object, default: None).

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = BridgeTowerSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BridgeTowerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'BridgeTowerAttention' and is used for pruning
        certain attention heads in the attention mechanism.

        Args:
            self (object): The instance of the BridgeTowerAttention class.
            heads (list): A list of integers representing the indices of attention heads to be pruned.
            If the list is empty, no pruning will be performed.

        Returns:
            None: The method does not return any value explicitly,
                as it operates by modifying attributes within the BridgeTowerAttention instance.

        Raises:
            This method may raise exceptions if:

            - The 'heads' parameter is not a list.
            - The 'heads' list contains non-integer elements.
            - The 'heads' list contains indices that are out of bounds for the attention heads.
            - Any of the pruning operations encounter errors or exceptions during execution.
            - Internal functions like 'find_pruneable_heads_and_indices' or 'prune_linear_layer' raise exceptions.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Constructs the attention mechanism for the BridgeTower model.

        Args:
            self (BridgeTowerAttention): The instance of the BridgeTowerAttention class.
            hidden_states (mindspore.Tensor): The input hidden states for the attention mechanism.
            attention_mask (Optional[mindspore.Tensor], optional):
                The attention mask to be applied to the input hidden states. Defaults to None.
            head_mask (Optional[mindspore.Tensor], optional): The head mask for multi-head attention. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor], optional):
                The hidden states from the encoder if the attention mechanism is used in a decoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor], optional):
                The attention mask for the encoder hidden states. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]], optional):
                Tuple of past key and value projection tensors. Defaults to None.
            output_attentions (Optional[bool], optional):
                Flag to indicate if the attention weights should be returned. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the attention output tensor and any additional outputs from the attention mechanism.

        Raises:
            ValueError: If the input tensors are not of the expected shape or type.
            RuntimeError: If an error occurs during the attention mechanism computation.
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BridgeTowerBertCrossLayer(nn.Cell):

    """
    This Python class, 'BridgeTowerBertCrossLayer', represents a single layer in the BridgeTowerBert model.
    It is a subclass of nn.Cell and is responsible for performing cross-attention operations between
    hidden states from the encoder and decoder.

    Attributes:
        chunk_size_feed_forward (int): The chunk size used in the forward pass of the feed-forward network.
        seq_len_dim (int): The dimension along which the sequence length is defined.
        attention (BridgeTowerAttention): An instance of the BridgeTowerAttention class,
            responsible for performing self-attention operations.
        is_decoder (bool): A flag indicating whether the layer is part of the decoder or not.
        add_cross_attention (bool): A flag indicating whether to include cross-attention operations or not.
        crossattention (BridgeTowerAttention): An instance of the BridgeTowerAttention class,
            responsible for performing cross-attention operations.
        intermediate (BridgeTowerIntermediate): An instance of the BridgeTowerIntermediate class,
            responsible for applying an intermediate transformation to the attention output.
        output (BridgeTowerOutput): An instance of the BridgeTowerOutput class,
            responsible for producing the final output of the layer.

    Methods:
        construct: Performs the forward pass of the layer. It applies self-attention to the hidden states,
            followed by cross-attention if specified. The outputs are then passed through the feed-forward chunk function.
        feed_forward_chunk: Applies the intermediate and output transformations to the attention output.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BridgeTowerBertCrossLayer class.

        Args:
            self: The instance of the class.
            config: An object representing the configuration settings for the BridgeTowerBertCrossLayer.
                It must have the following attributes:

                - chunk_size_feed_forward: An integer representing the chunk size for feed forward layers.
                - is_decoder: A boolean indicating if the layer is used as a decoder.
                - add_cross_attention: A boolean indicating whether to add cross attention.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BridgeTowerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BridgeTowerAttention(config)
        self.intermediate = BridgeTowerIntermediate(config)
        self.output = BridgeTowerOutput(config)

    def construct(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        This method constructs a layer in the BridgeTowerBertCrossLayer class.

        Args:
            self: The instance of the class.
            hidden_states (torch.Tensor): The hidden states input to the layer.
            encoder_hidden_states (torch.Tensor): The hidden states of the encoder.
            attention_mask (torch.Tensor, optional): Mask for attention computation. Default is None.
            head_mask (torch.Tensor, optional): Mask for attention heads. Default is None.
            encoder_attention_mask (torch.Tensor, optional): Mask for encoder attention computation. Default is None.
            past_key_value (tuple, optional): Tuple containing past key and value tensors. Default is None.
            output_attentions (bool): Flag indicating whether to output attentions. Default is False.

        Returns:
            tuple: The outputs of the layer, including the final layer output and any additional outputs.

        Raises:
            ValueError: If the dimensions of the input tensors are incompatible.
            RuntimeError: If the attention computation encounters an issue.
            TypeError: If the input types are incorrect.
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        # add cross attentions if we output attention weights
        outputs = outputs + cross_attention_outputs[1:-1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Performs a feed-forward operation on the given attention_output.

        Args:
            self (BridgeTowerBertCrossLayer): An instance of the BridgeTowerBertCrossLayer class.
            attention_output: The attention output tensor to be processed. (Type: Tensor)

        Returns:
            None.

        Raises:
            None.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BridgeTowerTextLayer(nn.Cell):

    """
    This class represents a BridgeTowerTextLayer, which is a component of a neural network model. It inherits from the nn.Cell class.

    Attributes:
        chunk_size_feed_forward (int): The chunk size for the feed forward operation.
        seq_len_dim (int): The dimension along which the sequence length is considered.
        attention (BridgeTowerAttention): An instance of the BridgeTowerAttention class.
        is_decoder (bool): A flag indicating if the layer is used as a decoder model.
        add_cross_attention (bool): A flag indicating if cross attention is added.
        crossattention (BridgeTowerAttention): An instance of the BridgeTowerAttention class for cross attention.
        intermediate (BridgeTowerIntermediate): An instance of the BridgeTowerIntermediate class.
        output (BridgeTowerOutput): An instance of the BridgeTowerOutput class.

    Methods:
        __init__: Initializes the BridgeTowerTextLayer with the given configuration.
        construct: Constructs the layer by applying attention and feed forward operations to the input hidden states.
        feed_forward_chunk: Applies the feed forward operation to the attention output.

    """
    def __init__(self, config):
        """
        Initialize the BridgeTowerTextLayer class.

        Args:
            self: The instance of the class.
            config (object):
                An object containing configuration parameters for the BridgeTowerTextLayer.

                - chunk_size_feed_forward (int): The size of the chunk for feed-forward operations.
                - seq_len_dim (int): The dimension of the sequence length.
                - attention (object): An instance of BridgeTowerAttention class.
                - is_decoder (bool): Indicates if the model is a decoder.
                - add_cross_attention (bool): Indicates if cross-attention is added.
                - crossattention (object): An instance of BridgeTowerAttention class for cross-attention.
                - intermediate (object): An instance of BridgeTowerIntermediate class.
                - output (object): An instance of BridgeTowerOutput class.

        Returns:
            None.

        Raises:
            ValueError: Raised if add_cross_attention is True but the model is not a decoder.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BridgeTowerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BridgeTowerAttention(config, position_embedding_type="absolute")
        self.intermediate = BridgeTowerIntermediate(config)
        self.output = BridgeTowerOutput(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Constructs the BridgeTowerTextLayer.

        This method takes in the following parameters:
        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The hidden states of the input sequence.
            attention_mask (Optional[mindspore.Tensor]): Mask to avoid performing attention on padding tokens. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): Mask to nullify selected heads of the self-attention modules. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder output sequence. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask to avoid performing attention on encoder padding tokens. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Tuple containing the past key-value states. Defaults to None.
            output_attentions (Optional[bool]): Whether to output the attention weights. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: Outputs of the BridgeTowerTextLayer.

        Raises:
            ValueError:
                If `encoder_hidden_states` are passed, the `BridgeTowerTextLayer` must be instantiated with
                cross-attention layers by setting `config.add_cross_attention=True`.
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Method 'feed_forward_chunk' in the class 'BridgeTowerTextLayer'.

        This method performs a feed-forward operation on the given attention output.

        Args:
            self (object): The instance of the BridgeTowerTextLayer class.
            attention_output (object): The attention output tensor to be processed. It should be a valid tensor object.

        Returns:
            object: The layer output after the feed-forward operation. It may be None if the operation fails.

        Raises:
            ValueError: If the attention_output is not a valid tensor object.
            TypeError: If the intermediate or output methods encounter type-related issues during processing.
            RuntimeError: If any unexpected runtime error occurs during the feed-forward operation.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaEncoder with Roberta->BridgeTowerText
class BridgeTowerTextEncoder(nn.Cell):

    """
    BridgeTowerTextEncoder represents a text encoder for a specific model, containing multiple layers for processing input text data.
    This class inherits from nn.Cell and is designed to construct the text encoder layers and handle various input parameters during the encoding process.

    Attributes:
        config: A configuration object containing settings for the text encoder.
        layer: A CellList containing the individual text encoder layers.
        gradient_checkpointing: A boolean indicating whether gradient checkpointing is enabled.

    Methods:
        __init__(config):
            Initializes the BridgeTowerTextEncoder with the provided configuration.

        construct:
            Constructs the text encoder using the specified input tensors and parameters, and returns the output based on the given settings.

            Parameters:

            - hidden_states: Input tensor representing the hidden states of the text data.
            - attention_mask: Optional tensor for attention masking.
            - head_mask: Optional tensor for masking specific heads in the attention mechanism.
            - encoder_hidden_states: Optional tensor representing hidden states of the encoder.
            - encoder_attention_mask: Optional tensor for encoder attention masking.
            - past_key_values: Optional tuple of past key values for caching.
            - use_cache: Optional boolean indicating whether to use caching during encoding.
            - output_attentions: Optional boolean indicating whether to output attention values.
            - output_hidden_states: Optional boolean indicating whether to output hidden states.
            - return_dict: Optional boolean indicating whether to return the output as a dictionary.
            - Returns:

                - Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                - The output tensor or a custom data structure containing various encoding outputs.
            - Raises:

                - Warning: If 'use_cache' is set to True while using gradient checkpointing.

        _gradient_checkpointing_func():
            Internal method for handling gradient checkpointing during encoding.

            Parameters:

            - layer_module: The specific layer module to apply gradient checkpointing.
            -  hidden_states: Input tensor representing hidden states.
            - attention_mask: Tensor for attention masking.
            - layer_head_mask: Tensor for masking specific heads in the attention mechanism.
            - encoder_hidden_states: Tensor representing hidden states of the encoder.
            - encoder_attention_mask: Tensor for encoder attention masking.
            - past_key_value: Tuple of past key values for caching.
            - output_attentions: Boolean indicating whether to output attention values.

            Returns:
                Tuple: Output tuple from applying gradient checkpointing on the layer.

        BaseModelOutputWithPastAndCrossAttentions():
            Method for organizing and returning the final output structure of the text encoder.

            Parameters:

            - last_hidden_state: The final hidden state output.
            - past_key_values: Tuple of past key values for caching.
            - hidden_states: Tuple containing all hidden states.
            - attentions: Tuple containing self-attention values.
            - cross_attentions: Tuple containing cross-attention values.

            Returns:

            - BaseModelOutputWithPastAndCrossAttentions: Custom data structure with organized encoding outputs.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the BridgeTowerTextEncoder class.

        Args:
            self: The instance of the BridgeTowerTextEncoder class.
            config: A dictionary containing configuration parameters for the text encoder.
                The config parameter should include the following keys:

                - num_hidden_layers: An integer specifying the number of hidden layers in the text encoder.
                Must be a positive integer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not a dictionary.
            ValueError: If the num_hidden_layers key is missing in the config dictionary.
            ValueError: If num_hidden_layers is not a positive integer.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([BridgeTowerTextLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """
        This method 'construct' is defined within the 'BridgeTowerTextEncoder' class and
        is responsible for processing the input hidden states and generating output based on the specified parameters.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states to be processed.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): An optional tensor representing the head mask. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor representing the encoder hidden states. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the encoder attention mask. Defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of past key values. Defaults to None.
            use_cache (Optional[bool]): An optional boolean indicating whether to use cache. Defaults to None.
            output_attentions (Optional[bool]): An optional boolean indicating whether to output attentions. Defaults to False.
            output_hidden_states (Optional[bool]): An optional boolean indicating whether to output hidden states. Defaults to False.
            return_dict (Optional[bool]): An optional boolean indicating whether to return a dictionary. Defaults to True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                Returns the processed hidden states, past key values, hidden states, attentions,
                and cross attentions based on the specified parameters.

        Raises:
            None
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->BridgeTowerText
class BridgeTowerTextEmbeddings(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        """
        Args:
            self (object): The instance of the BridgeTowerTextEmbeddings class.
            config (object):
                An object containing configuration parameters for the text embeddings.

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layer.
                - pad_token_id (int): The index of the padding token in the vocabulary.
                - max_position_embeddings (int): The maximum number of positions for positional embeddings.
                - type_vocab_size (int): The size of the type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.
                - position_embedding_type (str, optional): The type of position embedding, defaults to 'absolute'.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters are not valid or if the padding index is out of range.
            TypeError: If the types of configuration parameters are not as expected.
            IndexError: If there is an index error when accessing configuration parameters.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).expand((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def construct(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        """
        This method constructs the text embeddings for the BridgeTowerTextEmbeddings class.

        Args:
            self: The instance of the class.
            input_ids (Tensor, optional): The input tensor containing the token indices. Default is None.
            token_type_ids (Tensor, optional): The input tensor containing the token type indices. Default is None.
            position_ids (Tensor, optional): The input tensor containing the position indices. Default is None.
            inputs_embeds (Tensor, optional): The input embeddings tensor. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            None.

        Raises:
            ValueError: If input_ids and inputs_embeds are both None, or if the input tensor shapes are invalid.
            RuntimeError: If an error occurs during the computation process.
            TypeError: If the input types are not as expected.
            IndexError: If there is an index out of range error.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: mindspore.Tensor

        Returns: mindspore.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ops.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: mindspore.Tensor x:

    Returns:
        mindspore.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (ops.cumsum(mask, axis=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class BridgeTowerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BridgeTowerConfig
    base_model_prefix = "bridgetower"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BridgeTowerSelfAttention", "BridgeTowerResidualAttention"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """
        Initializes the weights of the model's cells.

        Args:
            self: An instance of the BridgeTowerPreTrainedModel class.
            cell: The cell whose weights need to be initialized.

        Returns:
            None

        Raises:
            None
        """
        if isinstance(cell, BridgeTowerVisionModel):
            proj_std = (cell.visual.transformer.hidden_size**-0.5) * (
                (2 * cell.visual.transformer.num_hidden_layers) ** -0.5
            )
            attn_std = cell.visual.transformer.hidden_size**-0.5
            fc_std = (2 * cell.visual.transformer.hidden_size) ** -0.5
            for block in cell.visual.transformer.resblocks:
                block.attn.in_proj_weight.initialize(Normal(attn_std * self.config.initializer_factor))
                block.attn.out_proj.weight.initialize(Normal(proj_std * self.config.initializer_factor))
                block.mlp.c_fc.weight.initialize(Normal(fc_std * self.config.initializer_factor))
                block.mlp.c_proj.weight.initialize(Normal(proj_std * self.config.initializer_factor))

            cell.visual.embeddings.class_embedding.initialize(Normal(attn_std * self.config.initializer_factor))
            cell.visual.embeddings.position_embedding.weight.initialize(Normal(attn_std * self.config.initializer_factor))
        elif isinstance(cell, (nn.Dense, nn.Conv2d, nn.Embedding)):
            cell.weight.initialize(Normal(0.05 * self.config.initializer_factor))
        elif isinstance(cell, nn.LayerNorm):
            cell.bias.initialize('zeros')
            cell.weight.initialize('ones')

        if 'Dense' in str(type(cell)) and cell.bias is not None:
            cell.bias.initialize('zeros')


class BridgeTowerVisionModel(BridgeTowerPreTrainedModel):

    """
    BridgeTowerVisionModel represents a vision model that incorporates a BridgeTowerVisionTransformer for processing images.
    This class inherits from BridgeTowerPreTrainedModel and provides methods for initializing the model,
    accessing the data type, and constructing the model output based on input images and optional masks.

    Attributes:
        visual: An instance of BridgeTowerVisionTransformer used for processing visual inputs.

    Methods:
        __init__(config): Initializes the BridgeTowerVisionModel with the provided configuration.
        dtype: Returns the data type of the patch embeddings weight.
        construct(image, image_mask=None): Constructs the model output based on the input image and optional mask.
    """
    config_class = BridgeTowerVisionConfig

    def __init__(self, config):
        """
        Initializes an instance of the BridgeTowerVisionModel class.

        Args:
            self: The instance of the BridgeTowerVisionModel class.
            config: A configuration object containing settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain valid settings.
            RuntimeError: If there is an issue with initializing the model.
        """
        super().__init__(config)
        self.visual = BridgeTowerVisionTransformer(config)

    @property
    def dtype(self):
        """
        Method to get the data type of the patch embedding weights in the BridgeTowerVisionModel.

        Args:
            self (BridgeTowerVisionModel): The instance of the BridgeTowerVisionModel class.
                This parameter refers to the current instance of the BridgeTowerVisionModel.
                It is used to access the patch embedding weights for which the data type is retrieved.

        Returns:
            None:
                This method does not return a value.
                It simply retrieves and provides information about the data type of the patch embedding weights.

        Raises:
            None.
        """
        return self.visual.embeddings.patch_embedding.weight.dtype

    def construct(self, image, image_mask=None):
        """
        Constructs the BridgeTowerVisionModel by processing the input image and its corresponding mask.

        Args:
            self (BridgeTowerVisionModel): An instance of the BridgeTowerVisionModel class.
            image (Tensor): The input image to be processed. It should be of the same dtype as self.dtype.
            image_mask (Tensor, optional):
                The optional mask corresponding to the image. The mask should have the same dimensions as the image.
                If not provided, the method will process the image without considering any specific mask.

        Returns:
            None.

        Raises:
            None.
        """
        return self.visual(image.type(self.dtype), image_mask)


class BridgeTowerTextModel(BridgeTowerPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """
    config_class = BridgeTowerTextConfig

    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new instance of the BridgeTowerTextModel class.

        Args:
            self: The object instance.
            config (object): The configuration object containing the model settings.
            add_pooling_layer (bool): Specifies whether to add a pooling layer. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BridgeTowerTextEmbeddings(config)
        self.encoder = BridgeTowerTextEncoder(config)

        self.pooler = BridgeTowerPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings for the BridgeTowerTextModel.

        Args:
            self (BridgeTowerTextModel): The instance of the BridgeTowerTextModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the BridgeTowerTextModel.

        Args:
            self (BridgeTowerTextModel): The instance of the BridgeTowerTextModel class.
            value (object): The input embeddings to be set for the model. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Args:
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having
                4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ops.ones(((batch_size, seq_length + past_key_values_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BridgeTowerModel(BridgeTowerPreTrainedModel):

    """
    BridgeTowerModel
    Represents a BridgeTower model, which is a model for processing multimodal inputs, combining text and
    image information using cross-modal transformers.

    This class inherits from BridgeTowerPreTrainedModel and implements methods for initializing the model,
    constructing the model, and getting classification features.

    The BridgeTowerModel class includes methods for getting and setting input embeddings, as well as constructing
    the model for processing multimodal inputs. It also provides a method for obtaining
    classification features from the processed multimodal inputs.

    Attributes:
        config: The configuration for the BridgeTowerModel.

    Methods:
        __init__: Initializes the BridgeTowerModel with the provided configuration.
        get_input_embeddings: Retrieves the input embeddings from the text model.
        set_input_embeddings: Sets the input embeddings for the text model.
        construct: Constructs the model for processing multimodal inputs and returns the model output.
        get_cls_features: Retrieves the classification features from the processed multimodal inputs.

    Example:
        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerModel
        >>> from PIL import Image
        >>> import requests
        ...
        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"
        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
        >>> model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
        ...
        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.keys()
        odict_keys(['text_features', 'image_features', 'pooler_output'])
        ```
    """
    def __init__(self, config):
        """
        Initializes a BridgeTowerModel instance.

        Args:
            self (object): The instance of the BridgeTowerModel class.
            config (object):
                An object containing configuration settings for the model.

                - Purpose: Specifies the configuration parameters for the BridgeTowerModel.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.config = config
        vision_config = config.vision_config
        text_config = config.text_config

        if config.share_cross_modal_transformer_layers:
            self.cross_modal_text_transform = nn.Dense(text_config.hidden_size, config.hidden_size)
            self.cross_modal_image_transform = nn.Dense(vision_config.hidden_size, config.hidden_size)
        else:
            self.cross_modal_text_transform = nn.CellList(
                [nn.Dense(text_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )
            self.cross_modal_image_transform = nn.CellList(
                [nn.Dense(vision_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.vision_model = BridgeTowerVisionModel(vision_config)

        self.text_model = BridgeTowerTextModel(text_config)

        if not vision_config.share_layernorm and config.init_layernorm_from_vision_encoder:
            for ln in self.vision_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vision_model.visual.ln_post.weight.data
                ln.bias.data = self.vision_model.visual.ln_post.bias.data

        self.cross_modal_image_layers = nn.CellList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )
        self.cross_modal_text_layers = nn.CellList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )

        # Class token => Linear => Tanh
        self.cross_modal_image_pooler = BridgeTowerPooler(config)
        self.cross_modal_text_pooler = BridgeTowerPooler(config)

        # Initialize BridgeTower Components
        self.cross_modal_text_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.cross_modal_image_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        if config.share_link_tower_layers:
            self.cross_modal_text_link_tower = BridgeTowerLinkTower(config)
            self.cross_modal_image_link_tower = BridgeTowerLinkTower(config)
        else:
            self.cross_modal_text_link_tower = nn.CellList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )
            self.cross_modal_image_link_tower = nn.CellList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )

        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the BridgeTowerModel's text model.

        Args:
            self: An instance of the BridgeTowerModel class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the underlying text model of the BridgeTowerModel.
        The input embeddings are representations of the input text that are used for further processing or analysis.
        By calling this method, you can access the input embeddings that have been generated by the text model.

        Note that the text model must be initialized and trained before calling this method.
        If the text model has not been initialized or trained, this method may not return the expected embeddings or may
        raise an exception.
        """
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the BridgeTowerModel.

        Args:
            self (BridgeTowerModel): The instance of the BridgeTowerModel class.
            value: The input embeddings to be set for the BridgeTowerModel. It should be of type Tensor or None.

        Returns:
            None.

        Raises:
            None.
        """
        self.text_model.set_input_embeddings(value)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        pixel_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        image_embeds: Optional[mindspore.Tensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple[mindspore.Tensor], BridgeTowerModelOutput]:
        r"""
        Args:
            output_hidden_states (`bool`, *optional*):
                If set to `True`, hidden states are returned as a list containing the hidden states of text, image, and
                cross-modal components respectively. i.e. `(hidden_states_text, hidden_states_image,
                hidden_states_cross_modal)` where each element is a list of the hidden states of the corresponding
                modality. `hidden_states_txt/img` are a list of tensors corresponding to unimodal hidden states and
                `hidden_states_cross_modal` is a list of tuples containing `cross_modal_text_hidden_states` and
                `cross_modal_image_hidden_states` of each brdige layer.
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels are currently not supported.

        Returns:
            Union[Tuple[mindspore.Tensor], BridgeTowerModelOutput]:

        Example:
            ```python
            >>> from transformers import BridgeTowerProcessor, BridgeTowerModel
            >>> from PIL import Image
            >>> import requests
            ...
            >>> # prepare image and text
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> text = "hello world"
            >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
            >>> model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
            ...
            >>> inputs = processor(image, text, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> outputs.keys()
            odict_keys(['text_features', 'image_features', 'pooler_output'])
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states_text = () if output_hidden_states else None
        all_hidden_states_image = () if output_hidden_states else None
        all_hidden_states_cross = () if output_hidden_states else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_token_type_idx = image_token_type_idx if image_token_type_idx else 1
        input_shape = input_ids.shape
        text_embeds = self.text_model.embeddings(input_ids=input_ids)

        if output_hidden_states:
            all_hidden_states_text += (text_embeds,)

        if attention_mask is None:
            attention_mask = ops.ones(input_shape, dtype=mindspore.int64)
        extend_text_masks = self.text_model.get_extended_attention_mask(attention_mask, input_shape)

        # The split_index determines how many layers of the uni-modal encoder are applied before the cross-modal encoder
        split_index = len(self.text_model.encoder.layer) - self.config.num_hidden_layers + 1

        # Run the first 'split_index' layers of the textual encoder
        for layer in self.text_model.encoder.layer[:split_index]:
            text_embeds = layer(text_embeds, extend_text_masks)[0]

            if output_hidden_states:
                all_hidden_states_text += (text_embeds,)

        if image_embeds is None:
            image_embeds = self.vision_model.visual.construct_pre(pixel_values.type(self.vision_model.dtype))
        else:
            # Permute as BridgeTowerResidualAttention has batch_first=True
            image_embeds = image_embeds.permute(1, 0, 2)

        if output_hidden_states:
            all_hidden_states_image += (image_embeds,)

        # Run the first 'split_index' layers of the visual encoder
        for block in self.vision_model.visual.transformer.resblocks[:split_index]:
            image_embeds = block(image_embeds)
            if output_hidden_states:
                all_hidden_states_image += (image_embeds,)

        image_embeds_with_ln = self.vision_model.visual.construct_post(image_embeds.type(self.vision_model.dtype))

        # first layer is a special case because we don't have the output from the cross-encoder yet
        cross_modal_text = self.cross_modal_text_transform(text_embeds)

        text_token_type_embeddings = self.token_type_embeddings(
            ops.zeros(1, dtype=mindspore.int64)
        ).expand_as(cross_modal_text)

        cross_modal_text = self.cross_modal_text_layernorm(cross_modal_text + text_token_type_embeddings)

        image_embeds_with_ln = self.cross_modal_image_transform(image_embeds_with_ln)
        image_token_type_embeddings = self.token_type_embeddings(
            ops.full((1,), image_token_type_idx, dtype=mindspore.int64)
        ).expand_as(image_embeds_with_ln)

        image_embeds_with_ln = image_embeds_with_ln + image_token_type_embeddings
        cross_modal_image = self.cross_modal_image_layernorm(image_embeds_with_ln)

        pixel_mask = ops.ones(
            (cross_modal_image.shape[0], cross_modal_image.shape[1]),
            dtype=mindspore.int64,
        )
        extend_image_masks = self.text_model.get_extended_attention_mask(pixel_mask, pixel_mask.shape)

        layer_outputs_text = self.cross_modal_text_layers[0](
            cross_modal_text,
            cross_modal_image,
            attention_mask=extend_text_masks,
            encoder_attention_mask=extend_image_masks,
            output_attentions=output_attentions,
        )
        cross_text_features = layer_outputs_text[0]

        layer_outputs_image = self.cross_modal_image_layers[0](
            cross_modal_image,
            cross_modal_text,
            attention_mask=extend_image_masks,
            encoder_attention_mask=extend_text_masks,
            output_attentions=output_attentions,
        )
        cross_image_features = layer_outputs_image[0]

        if output_hidden_states:
            all_hidden_states_cross += ((cross_text_features, cross_image_features),)

        if output_attentions:
            all_self_attentions += ((layer_outputs_text[1], layer_outputs_image[1]),)

        link_layer_index = 0

        #  Each of the top 6 layers of the visual and textual encoders ([split_index:]) is connected to each layer of
        #  the cross-modal encoder via bridge layers, which brings bottom-up alignment and fusion to the cross-modal encoder.
        for i in range(split_index, len(self.text_model.encoder.layer)):
            text_embeds = self.text_model.encoder.layer[i](text_embeds, extend_text_masks)[0]
            image_embeds = self.vision_model.visual.transformer.resblocks[i](image_embeds).type(
                self.vision_model.dtype
            )
            image_embeds_with_ln = (
                self.cross_modal_image_transform(self.vision_model.visual.construct_post(image_embeds))
                + image_token_type_embeddings
            )

            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            image_link_tower = self.cross_modal_image_link_tower[link_layer_index]

            # Bridge layers for textual and visual encoders
            cross_text_features_ = text_link_tower(
                self.cross_modal_text_transform(text_embeds) + text_token_type_embeddings,
                cross_text_features,
                extend_text_masks,
            )
            cross_image_features_ = image_link_tower(image_embeds_with_ln, cross_image_features, extend_image_masks)

            # Cross-modal encoder via bridge layers of textual and visual encoders
            layer_outputs_text = self.cross_modal_text_layers[link_layer_index + 1](
                cross_text_features_,
                cross_image_features_,
                attention_mask=extend_text_masks,
                encoder_attention_mask=extend_image_masks,
                output_attentions=output_attentions,
            )
            cross_text_features = layer_outputs_text[0]

            layer_outputs_image = self.cross_modal_image_layers[link_layer_index + 1](
                cross_image_features_,
                cross_text_features_,
                attention_mask=extend_image_masks,
                encoder_attention_mask=extend_text_masks,
                output_attentions=output_attentions,
            )
            cross_image_features = layer_outputs_image[0]

            link_layer_index += 1

            if output_hidden_states:
                all_hidden_states_text += (text_embeds,)
                all_hidden_states_image += (image_embeds,)
                all_hidden_states_cross += ((cross_text_features, cross_image_features),)

            if output_attentions:
                all_self_attentions += ((layer_outputs_text[1], layer_outputs_image[1]),)

        #  Concatenate the cls token of the text and image features to get the final represtation
        text_features, image_features = cross_text_features, cross_image_features
        cls_features = self.get_cls_features(text_features, image_features)

        if output_hidden_states:
            all_hidden_states = (all_hidden_states_text, all_hidden_states_image, all_hidden_states_cross)

        if not return_dict:
            return tuple(
                v
                for v in [text_features, image_features, cls_features, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BridgeTowerModelOutput(
            text_features=text_features,
            image_features=image_features,
            pooler_output=cls_features,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def get_cls_features(self, text_features, image_features):
        """
        This method 'get_cls_features' is defined in the class 'BridgeTowerModel'
        and is used to obtain the class features by pooling text and image features.

        Args:
            self (object): The instance of the BridgeTowerModel class.
            text_features (array): The input text features to be pooled for obtaining class features.
            image_features (array): The input image features to be pooled for obtaining class features.

        Returns:
            None:
                This method returns None, as the class features are directly computed and concatenated without any additional processing.

        Raises:
            None.
        """
        cls_features_text = self.cross_modal_text_pooler(text_features)
        cls_features_image = self.cross_modal_image_pooler(image_features)
        return ops.cat([cls_features_text, cls_features_image], axis=-1)


# Copied from transformers.models.vilt.modeling_vilt.ViltPredictionHeadTransform with Vilt->BridgeTower
class BridgeTowerPredictionHeadTransform(nn.Cell):

    """
    Represents a transformation head for a bridge tower prediction task.
    This class inherits from nn.Cell.

    Attributes:
        dense (nn.Dense): A dense layer for transforming input hidden states.
        transform_act_fn (function): Activation function to apply to hidden states.
        LayerNorm (nn.LayerNorm): Layer normalization to normalize hidden states.

    Methods:
        __init__(config): Initializes the BridgeTowerPredictionHeadTransform instance.
        construct(hidden_states): Applies transformation operations to input hidden states.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BridgeTowerPredictionHeadTransform class.

        Args:
            self: The instance of the class.
            config:
                An object of type 'config' containing the configuration settings.

                - Type: object
                - Purpose: Contains the configuration settings for the transformation.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        """
        Class: BridgeTowerPredictionHeadTransform

        Method: construct

        Description:
            This method constructs the prediction head transformation for the BridgeTower model.

        Args:
            self: (object) The instance of the BridgeTowerPredictionHeadTransform class.
            hidden_states: (tensor) The input hidden states to be transformed by the prediction head.

        Returns:
            None: This method does not return any value explicitly, as it modifies the hidden_states in place.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BridgeTowerMLMHead(nn.Cell):

    """
    BridgeTowerMLMHead represents the Masked Language Model (MLM) head for the BridgeTower model.
    This class inherits from nn.Cell and implements the MLM head for generating predictions using the provided
    configuration and optional weight.

    Attributes:
        config: The configuration settings for the MLM head.
        transform: An instance of BridgeTowerPredictionHeadTransform for transforming the input.
        decoder: A fully connected neural network layer for generating MLM predictions.
        bias: The bias parameter for the MLM predictions.

    Methods:
        __init__(self, config, weight=None): Initializes the BridgeTowerMLMHead with the given configuration and optional weight.

        construct(self, x): Constructs the MLM predictions for the input tensor x by applying transformations and using the decoder with bias.
    """
    def __init__(self, config, weight=None):
        """
        Initializes the BridgeTowerMLMHead class.

        Args:
            self: The instance of the class.
            config: A configuration object containing the settings for the model.
            weight: (optional) The weight parameter for the decoder. Default value is None.

        Returns:
            None.

        Raises:
            ValueError: If the config parameter is not provided or is not of the expected type.
            TypeError: If the weight parameter is provided but is not of the expected type.
        """
        super().__init__()
        self.config = config
        self.transform = BridgeTowerPredictionHeadTransform(config)
        self.decoder = nn.Dense(config.hidden_size, config.text_config.vocab_size, has_bias=False)
        self.bias = Parameter(ops.zeros(config.text_config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def construct(self, x):
        """
        This method constructs the MLM score for the BridgeTowerMLMHead class.

        Args:
            self (object): The instance of the BridgeTowerMLMHead class.
            x (object): The input data for which the MLM score is to be calculated.

        Returns:
            None.

        Raises:
            None.
        """
        mlm_score = self.transform(x)
        mlm_score = self.decoder(mlm_score) + self.bias
        return mlm_score


class BridgeTowerITMHead(nn.Cell):

    """
    BridgeTowerITMHead is a class representing an ITM (Item-Transaction-Model) head for a Bridge Tower model.
    This class inherits from nn.Cell and contains methods for initializing the head with a specific
    hidden size and constructing the ITM score based on the input data.

    Attributes:
        fc (nn.Dense): A fully connected layer for computing the ITM score.

    Methods:
        __init__(hidden_size): Initializes the BridgeTowerITMHead with the specified hidden size.
        construct(x): Constructs the ITM score based on the input data x using the fully connected layer fc.
    """
    def __init__(self, hidden_size):
        """
        Initialize the BridgeTowerITMHead class with the specified hidden size.

        Args:
            self: The instance of the BridgeTowerITMHead class.
            hidden_size (int): The size of the hidden layer in the neural network.
                Specifies the number of neurons in the hidden layer.
                Should be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.fc = nn.Dense(hidden_size, 2)

    def construct(self, x):
        """
        Construct a ITM score based on the input data.

        Args:
            self (object): The instance of the class BridgeTowerITMHead.
            x (object): The input data for constructing the ITM score.

        Returns:
            None.

        Raises:
            None
        """
        itm_score = self.fc(x)
        return itm_score


class BridgeTowerForMaskedLM(BridgeTowerPreTrainedModel):

    """
    BridgeTowerForMaskedLM class represents a model for masked language modeling using the BridgeTower architecture.
    It inherits functionality from the BridgeTowerPreTrainedModel class.

    This class includes methods for initializing the model with configuration, getting and setting output embeddings,
    and constructing the model for inference.
    The 'construct' method takes various input tensors such as input_ids, attention_mask, token_type_ids, pixel_values,
    pixel_mask, etc., and returns masked language modeling outputs.
    It also supports optional labels for computing the masked language modeling loss.

    The class provides an example of how to use the model for masked language modeling tasks using images and text inputs.
    It showcases the process of preparing inputs, performing a forward pass, decoding model outputs, and printing the results.

    The BridgeTowerForMaskedLM class encapsulates the functionality for masked language modeling tasks using the BridgeTower architecture.
    """
    _tied_weights_keys = ["mlm_score.decoder.weight"]

    def __init__(self, config):
        """
        __init__

        Initializes an instance of the BridgeTowerForMaskedLM class.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing settings and parameters for the BridgeTowerForMaskedLM instance.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)
        self.mlm_score = BridgeTowerMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method returns the output embeddings for the Masked Language Model (MLM) decoder.

        Args:
            self (BridgeTowerForMaskedLM): The instance of the BridgeTowerForMaskedLM class.

        Returns:
            None: This method returns None, as it directly returns the output embeddings without any further processing.

        Raises:
            None
        """
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the BridgeTowerForMaskedLM model.

        Args:
            self (BridgeTowerForMaskedLM): The instance of the BridgeTowerForMaskedLM class.
            new_embeddings (Tensor): The new embeddings to be set as the output embeddings. It should be a tensor of shape (vocab_size, hidden_size).

        Returns:
            None.

        Raises:
            None.

        This method sets the output embeddings for the BridgeTowerForMaskedLM model by updating the decoder attribute
        of the mlm_score object. The new_embeddings parameter should be a tensor representing the new embeddings to be
        used as the output embeddings. The tensor should have a shape of (vocab_size, hidden_size) where vocab_size is
        the number of tokens in the vocabulary and hidden_size is the size of the hidden state of the model.

        Example:
            ```python
            >>> model = BridgeTowerForMaskedLM()
            >>> new_embeddings = torch.randn(model.vocab_size, model.hidden_size)
            >>> model.set_output_embeddings(new_embeddings)
            ```
        """
        self.mlm_score.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        pixel_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        image_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[MaskedLMOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            `Union[MaskedLMOutput, Tuple[mindspore.Tensor]]`

        Example:
            ```python
            >>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
            >>> from PIL import Image
            >>> import requests
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            >>> text = "a <mask> looking out of the window"
            ...
            >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
            >>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
            ...
            >>> # prepare inputs
            >>> encoding = processor(image, text, return_tensors="pt")
            ...
            >>> # forward pass
            >>> outputs = model(**encoding)
            ...
            >>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())
            ...
            >>> print(results)
            .a cat looking out of the window.
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mlm_logits = self.mlm_score(outputs.text_features if return_dict else outputs[0])
        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(mlm_logits.view(-1, self.config.text_config.vocab_size), labels.view(-1))

        if not return_dict:
            output = tuple(mlm_logits)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):

    """
    BridgeTowerForImageAndTextRetrieval is a class for performing image and text retrieval using the BridgeTower model.

    This class extends the BridgeTowerPreTrainedModel and provides methods for constructing the model and computing the image-text matching loss.

    Args:
        config (BridgeTowerConfig): Configuration for the model.

    Returns:
        SequenceClassifierOutput or Tuple[mindspore.Tensor]: The output of the model, including the image-text matching loss and the logits.

    Example:
        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]
        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> scores = dict()
        >>> for text in texts:
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, 1].item()
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the BridgeTowerForImageAndTextRetrieval class.

        Args:
            self: The instance of the class itself.
            config: A configuration object containing the necessary parameters for initialization.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)

        self.itm_score = BridgeTowerITMHead(config.hidden_size * 2)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        pixel_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        image_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, 1)`, *optional*):
                Labels for computing the image-text matching loss. 0 means the pairs don't match and 1 means they match.
                The pairs with 0 will be skipped for calculation.

        Returns:
            Union[SequenceClassifierOutput, Tuple[mindspore.Tensor]]

        Example:
            ```python
            >>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
            >>> import requests
            >>> from PIL import Image
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]
            ...
            >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
            >>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
            ...
            >>> # forward pass
            >>> scores = dict()
            >>> for text in texts:
            ...     # prepare inputs
            ...     encoding = processor(image, text, return_tensors="pt")
            ...     outputs = model(**encoding)
            ...     scores[text] = outputs.logits[0, 1].item()
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[2]

        logits = self.itm_score(pooler_output)

        itm_loss = None
        if labels is not None:
            itm_loss = ops.cross_entropy(logits, labels)

        if not return_dict:
            output = tuple(logits)
            return ((itm_loss,) + output) if itm_loss is not None else output

        return SequenceClassifierOutput(
            loss=itm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BridgeTowerContrastiveHead(nn.Cell):

    """
    This class represents a BridgeTowerContrastiveHead module for neural network operations.
    It inherits from nn.Cell and provides functionality for handling contrastive head operations within a neural network.

    Attributes:
        fc (nn.Dense): A fully connected layer for mapping input data from hidden_size to embed_size.

    Methods:
        __init__: Initializes the BridgeTowerContrastiveHead with the specified hidden_size and embed_size.
        construct: Applies the fully connected layer to the input data and returns the result.
    """
    def __init__(self, hidden_size, embed_size):
        """
        Initializes an instance of the BridgeTowerContrastiveHead class.

        Args:
            self (BridgeTowerContrastiveHead): The instance of the class.
            hidden_size (int): The size of the hidden layer.
            embed_size (int): The size of the embedding layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.fc = nn.Dense(hidden_size, embed_size)

    def construct(self, x):
        """Constructs the BridgeTowerContrastiveHead.

        This method constructs the BridgeTowerContrastiveHead by processing the input tensor 'x' through the fully connected layer 'fc'.

        Args:
            self (BridgeTowerContrastiveHead): An instance of the BridgeTowerContrastiveHead class.
            x (torch.Tensor): The input tensor to be processed through the fully connected layer.
                It should have a shape of (batch_size, features).

        Returns:
            None.

        Raises:
            None.
        """
        x = self.fc(x)
        return x


class BridgeTowerForContrastiveLearning(BridgeTowerPreTrainedModel):

    """
    Represents a BridgeTower model for contrastive learning.

    This class inherits from BridgeTowerPreTrainedModel and includes initialization and construction methods for
    contrastive learning. It contains methods for processing input data, calculating contrastive loss, and returning
    outputs for text and image embeddings.

    The `construct` method takes input tensors for text and image data, and optional parameters for attention, token
    types, and masks. It returns a BridgeTowerContrastiveOutput object containing the contrastive loss, logits,
    text embeddings, image embeddings, cross-modal embeddings, hidden states, and attentions.

    The example provided demonstrates the usage of the BridgeTowerForContrastiveLearning class for processing images
    and texts, calculating contrastive loss, and obtaining model outputs.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BridgeTowerForContrastiveLearning class.

        Args:
            self: The instance of the class.
            config: The configuration object containing various settings and parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)

        self.itc_text_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_image_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_cross_modal_head = BridgeTowerContrastiveHead(config.hidden_size * 2, config.contrastive_hidden_size)

        self.logit_scale = Parameter(mindspore.tensor(self.config.logit_scale_init_value))
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        pixel_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        image_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        return_loss: Optional[bool] = None,
    ) -> Union[BridgeTowerContrastiveOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            return_loss (`bool`, *optional*):
                Whether or not to return the contrastive loss.

        Returns:
            Union[BridgeTowerContrastiveOutput, Tuple[mindspore.Tensor]]

        Example:
            ```python
            >>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
            >>> import requests
            >>> from PIL import Image
            >>> import torch
            ...
            >>> image_urls = [
            ...     "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg",
            ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
            ... ]
            >>> texts = ["two dogs in a car", "two cats sleeping on a couch"]
            >>> images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
            ...
            >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
            >>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
            ...
            >>> inputs = processor(images, texts, padding=True, return_tensors="pt")
            >>> loss = model(**inputs, return_loss=True).loss
            ...
            >>> inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
            >>> loss_swapped = model(**inputs, return_loss=True).loss
            ...
            >>> print("Loss", round(loss.item(), 4))
            Loss 0.0019
            >>> print("Loss with swapped images", round(loss_swapped.item(), 4))
            Loss with swapped images 2.126
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[2]
        hidden_states_txt, hidden_states_img, hidden_states_cross_modal = (
            outputs.hidden_states if return_dict else outputs[3]
        )

        text_embeds = hidden_states_txt[-1]
        image_embeds = hidden_states_img[-1]

        image_embeds_with_ln = self.bridgetower.vision_model.visual.construct_post(image_embeds)
        image_token_type_embeddings = self.bridgetower.token_type_embeddings(
            ops.full((1,), 1, dtype=mindspore.int64)
        ).expand_as(image_embeds_with_ln)

        image_embeds = self.bridgetower.cross_modal_image_transform(image_embeds_with_ln) + image_token_type_embeddings

        # normalized features
        text_embeds = normalize(self.itc_text_head(text_embeds[:, 0, :]), dim=-1, p=2)
        image_embeds = normalize(self.itc_image_head(image_embeds[:, 0, :]), dim=-1, p=2)
        cross_embeds = normalize(self.itc_cross_modal_head(pooler_output), dim=-1, p=2)
        logits = ops.stack([text_embeds, image_embeds, cross_embeds], axis=-2)

        logit_scale = self.logit_scale.exp()
        logits_text_to_image = ops.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_text_to_cross = ops.matmul(text_embeds, cross_embeds.t()) * logit_scale
        logits_image_to_cross = ops.matmul(image_embeds, cross_embeds.t()) * logit_scale

        itc_loss = None

        if return_loss:
            labels = ops.arange(len(logits))
            text_to_image_loss = ops.cross_entropy(logits_text_to_image, labels)
            text_to_cross_loss = ops.cross_entropy(logits_text_to_cross, labels)
            image_to_cross_loss = ops.cross_entropy(logits_image_to_cross, labels)
            itc_loss = (text_to_image_loss + text_to_cross_loss + image_to_cross_loss) / 3.0

        if not return_dict:
            output = (logits, text_embeds, image_embeds, cross_embeds) + outputs[3:]
            return ((itc_loss,) + output) if itc_loss is not None else output

        return BridgeTowerContrastiveOutput(
            loss=itc_loss,
            logits=logits,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            cross_embeds=cross_embeds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "BridgeTowerForContrastiveLearning",
    "BridgeTowerForImageAndTextRetrieval",
    "BridgeTowerForMaskedLM",
    "BridgeTowerModel",
    "BridgeTowerPreTrainedModel",
]
