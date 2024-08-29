# coding=utf-8
# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.
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
""" MindSpore BLIP-2 model."""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import mindspore
from mindspore import Parameter
from mindspore.common.initializer import initializer, Normal, TruncatedNormal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ....utils import (
    ModelOutput,
    logging,
)
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/blip2-opt-2.7b"


@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`mindspore.Tensor`, *optional*, returned when `labels` is provided, `mindspore.Tensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """
    loss: Optional[Tuple[mindspore.Tensor]] = None
    logits: Optional[Tuple[mindspore.Tensor]] = None
    vision_outputs: Optional[mindspore.Tensor] = None
    qformer_outputs: Optional[Tuple[mindspore.Tensor]] = None
    language_model_outputs: Optional[Tuple[mindspore.Tensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        """
        Converts the Blip2ForConditionalGenerationModelOutput object to a tuple.
        
        Args:
            self: The instance of the Blip2ForConditionalGenerationModelOutput class.
        
        Returns:
            A tuple containing the attributes of the Blip2ForConditionalGenerationModelOutput object. The 'vision_outputs',
            'qformer_outputs', and 'language_model_outputs' attributes are recursively converted into tuples.
        
        Raises:
            None.
        """
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.blip.modeling_blip.BlipVisionEmbeddings with Blip->Blip2
class Blip2VisionEmbeddings(nn.Module):

    """
    This class represents the embedding module for the Blip2Vision model. It inherits from the nn.Module class.
    
    The Blip2VisionEmbeddings class initializes with a configuration object of type Blip2VisionConfig. It sets various attributes such as the embedding dimensions, image size, patch size, class embedding,
    patch embedding, number of patches, and position embedding.

    Attributes:
        config: The configuration object for Blip2Vision.
        embed_dim: The dimension of the embeddings.
        image_size: The size of the input image.
        patch_size: The size of the patches.
        class_embedding: The embedding for the class.
        patch_embedding: The embedding for the patches.
        num_patches: The total number of patches in the image.
        num_positions: The total number of positions in the image.
        position_embedding: The embedding for the positions.

    Methods:
        forward:
            Constructs the embeddings for the given pixel values.
            
            - pixel_values: A tensor containing the pixel values of the input image.
            - Returns: A tensor containing the embeddings for the input image.

    Note: This class assumes the availability of the following modules: nn, ops, mindspore.Tensor.
    """
    def __init__(self, config: Blip2VisionConfig):
        """
        Initializes the Blip2VisionEmbeddings class.

        Args:
            self: The instance of the Blip2VisionEmbeddings class.
            config (Blip2VisionConfig): An instance of the Blip2VisionConfig class containing configuration parameters.

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

        self.class_embedding = Parameter(ops.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = Parameter(ops.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs Blip2Vision embeddings based on the given pixel values.

        Args:
            self (Blip2VisionEmbeddings): The instance of the Blip2VisionEmbeddings class.
            pixel_values (mindspore.Tensor): A tensor containing pixel values of shape (batch_size, channels, height, width).
                The pixel values are used to generate patch embeddings for further processing.

        Returns:
            mindspore.Tensor: A tensor representing the forwarded Blip2Vision embeddings of shape (batch_size, total_embed_dim).
                The embeddings combine class embeddings and patch embeddings with positional encodings.

        Raises:
            ValueError: If the dtype of target_dtype is not compatible with the patch_embedding weights.
            IndexError: If the shape manipulation operations encounter indexing errors.
            RuntimeError: If there are issues with concatenating or adding embeddings during the process.
        """
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = ops.broadcast_to(self.class_embedding, (batch_size, 1, -1)).to(target_dtype)
        embeddings = ops.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.shape[1], :].to(target_dtype)
        return embeddings


class Blip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config):
        """
        Initializes an instance of the Blip2Attention class.

        Args:
            self: The instance of the class.
            config: 
                An object containing configuration parameters for the attention module.
                
                - Type: Any valid object
                - Purpose: Stores the configuration parameters for the attention module.
                - Restrictions: None

        Returns:
            None

        Raises:
            ValueError: 
                If embed_dim is not divisible by num_heads.

                - Reasons: The embed_dim parameter must be divisible by the num_heads parameter to ensure 
                proper attention calculation. If the condition is not met, this exception is raised.
        """
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
        self.dropout = nn.Dropout(p=config.attention_dropout)

        # small tweak here compared to CLIP, no bias here
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        if config.qkv_bias:
            q_bias = Parameter(ops.zeros(self.embed_dim))
            v_bias = Parameter(ops.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = ops.cat((q_bias, ops.zeros_like(v_bias), v_bias))
            self.qkv.bias = Parameter(qkv_bias)

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method, _shape, is defined within the Blip2Attention class and is used to reshape the input tensor for further processing.

        Args:
            self: Represents the instance of the class Blip2Attention.
            tensor (mindspore.Tensor): A tensor containing the input data to be reshaped.
            seq_len (int): An integer representing the sequence length of the input tensor.
            bsz (int): An integer representing the batch size of the input tensor.

        Returns:
            None: This method does not return any value; instead, it modifies the shape of the input tensor in-place.

        Raises:
            None
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.shape

        mixed_qkv = self.qkv(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_states, key_states.swapaxes(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.shape[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


# Copied from transformers.models.blip.modeling_blip.BlipMLP
class Blip2MLP(nn.Module):

    """
    Blip2MLP is a multi-layer perceptron (MLP) implemented as a subclass of the nn.Module class.
    It represents a feedforward neural network with two fully connected layers.

    The Blip2MLP class initializes with a configuration object, which contains various parameters for the MLP.
    It sets the configuration object to the 'config' attribute and assigns the activation function
    based on the 'hidden_act' parameter from the configuration object.

    The forward method takes a tensor 'hidden_states' as input and performs the following operations:

    1. Applies the first fully connected layer (fc1) to the 'hidden_states' tensor.
    2. Applies the activation function to the output of fc1.
    3. Applies the second fully connected layer (fc2) to the output of the activation function.
    4. Returns the resulting tensor.

    Note:
        This class assumes that ACT2FN is a dictionary mapping activation function names to their corresponding functions.

    Example usage:
        ```python
        >>> config = Configuration(hidden_act='relu', hidden_size=256, intermediate_size=128)
        >>> model = Blip2MLP(config)
        >>> input_tensor = mindspore.Tensor(...)
        >>> output_tensor = model.forward(input_tensor)
        ```

    Please refer to the nn.Module documentation for more information on how to use and train the Blip2MLP class.
    """
    def __init__(self, config):
        """
        Initializes an instance of the Blip2MLP class.

        Args:
            self (Blip2MLP): The current instance of the Blip2MLP class.
            config: A dictionary containing the configuration parameters for the MLP model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the Blip2MLP by applying linear transformations and activation functions to the input hidden states.

        Args:
            self (Blip2MLP): The instance of the Blip2MLP class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states of the Blip2MLP.
                It should be of type mindspore.Tensor.

        Returns:
            mindspore.Tensor: Returns the transformed hidden states as a mindspore.
                Tensor after applying linear transformations and activation functions.

        Raises:
            None
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.blip.modeling_blip.BlipEncoderLayer with Blip->Blip2
class Blip2EncoderLayer(nn.Module):

    """
        Blip2EncoderLayer represents a single layer in the Blip2 encoder, which consists of a self-attention mechanism
        followed by a feedforward neural network layer.

        This class inherits from nn.Module and is designed to be used within the Blip2 transformer model to process input
        sequences in the encoder stack.

        Attributes:
            embed_dim (int): The dimensionality of the hidden states in the encoder layer.
            self_attn (Blip2Attention): The self-attention mechanism used in the encoder layer.
            layer_norm1 (nn.LayerNorm): Layer normalization applied after the self-attention step.
            mlp (Blip2MLP): Feedforward neural network layer in the encoder.
            layer_norm2 (nn.LayerNorm): Layer normalization applied after the feedforward network.

        Methods:
            forward(hidden_states, attention_mask, output_attentions=False) -> Tuple[mindspore.Tensor]:
                Applies the encoder layer operations to the input hidden states.

                Args:

                - hidden_states (mindspore.Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
                - attention_mask (mindspore.Tensor): Attention mask of size `(batch, 1, tgt_len, src_len)` where padding
                elements are indicated by very large negative values.
                - output_attentions (bool, optional): Whether or not to return the attention tensors of all attention
                layers. Default is False.

                Returns:

                - Tuple[mindspore.Tensor]: A tuple containing the output hidden states of the encoder layer. If
                output_attentions is True, the tuple also includes the attention weights.
    """
    def __init__(self, config: Blip2Config):
        """Initialize a Blip2EncoderLayer object.

        Args:
            self (Blip2EncoderLayer): An instance of the Blip2EncoderLayer class.
            config (Blip2Config): An instance of the Blip2Config class representing the configuration settings.
                This parameter is used to initialize the hidden size attribute, embed_dim, of the Blip2EncoderLayer object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Blip2Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Blip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
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
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Blip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Blip2Config
    base_model_prefix = "blip"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Blip2Attention", "T5Block", "OPTDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(cell, (nn.Conv2d, nn.Linear, nn.Embedding)):
            cell.weight.set_data(initializer(Normal(factor), cell.weight.shape, cell.weight.dtype))
            if hasattr(cell, "bias") and cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        if isinstance(cell, Blip2VisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range

            cell.position_embedding.set_data(initializer(TruncatedNormal(factor), cell.position_embedding.shape, cell.position_embedding.dtype))
            cell.class_embedding.set_data(initializer(TruncatedNormal(factor), cell.class_embedding.shape, cell.class_embedding.dtype))

        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        elif isinstance(cell, nn.Linear) and cell.bias is not None:
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


# Copied from transformers.models.blip.modeling_blip.BlipEncoder with Blip->Blip2
class Blip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].

    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
    """
    def __init__(self, config: Blip2Config):
        """
        Initializes a new instance of the Blip2Encoder class.

        Args:
            self: The instance of the Blip2Encoder class.
            config (Blip2Config): An instance of Blip2Config class that holds the configuration parameters for the encoder.
                It is used to configure the encoder layers and gradient checkpointing.
                The config parameter must be of type Blip2Config.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
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
        for idx, encoder_layer in enumerate(self.layers):
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


# Copied from transformers.models.blip.modeling_blip.BlipVisionModel with Blip->Blip2, BLIP->BLIP_2
class Blip2VisionModel(Blip2PreTrainedModel):

    """
    This class represents a Blip2VisionModel which is designed for vision tasks using Blip2 models.
    The Blip2VisionModel class inherits from the Blip2PreTrainedModel class and provides functionality for processing pixel values,
    forwarding embeddings, and generating output for vision-related tasks.

    Attributes:
        config: An instance of Blip2VisionConfig containing configuration settings for the model.
        embeddings: An instance of Blip2VisionEmbeddings for creating embeddings from pixel values.
        encoder: An instance of Blip2Encoder for encoding input embeddings.
        post_layernorm: A LayerNorm module for applying layer normalization to the output.

    Methods:
        __init__: Initializes the Blip2VisionModel with the provided configuration.
        forward:
            Constructs the model by processing pixel values, generating embeddings, and producing output for vision tasks.
        get_input_embeddings: Retrieves the embeddings module used by the model for processing input pixel values.

    The Blip2VisionModel class provides a comprehensive solution for vision tasks by leveraging the Blip2 architecture
    and incorporating advanced features such as layer normalization and configurable output options.
    """
    main_input_name = "pixel_values"
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        """
        Initializes a new instance of the Blip2VisionModel class.

        Args:
            self: The object instance.
            config (Blip2VisionConfig): The configuration object containing various settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Blip2VisionEmbeddings(config)
        self.encoder = Blip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

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

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the Blip2VisionModel.

        Args:
            self (Blip2VisionModel): The instance of the Blip2VisionModel class.

        Returns:
            embeddings: This method returns the input embeddings.

        Raises:
            None.
        """
        return self.embeddings


class Blip2QFormerMultiHeadAttention(nn.Module):

    """
    This class represents a multi-head attention mechanism used in the Blip2QFormer model. It is designed to be compatible with the nn.Module class.

    Attributes:
        config: A configuration object containing various hyperparameters for the attention mechanism.
        is_cross_attention: A boolean indicating whether the attention mechanism is used for cross-attention or not.
        num_attention_heads: An integer specifying the number of attention heads.
        attention_head_size: An integer representing the size of each attention head.
        all_head_size: An integer representing the total size of all attention heads combined.
        query: A fully connected layer used to transform the input hidden states into query vectors.
        key: A fully connected layer used to transform the input hidden states into key vectors.
        value: A fully connected layer used to transform the input hidden states into value vectors.
        dropout: A dropout layer applied to the attention probabilities.
        position_embedding_type:
            A string specifying the type of position embedding used ('absolute', 'relative_key', 'relative_key_query').
        max_position_embeddings: An integer representing the maximum number of position embeddings.
        distance_embedding: An embedding layer used to calculate the relative distance between positions.
        save_attention: A boolean indicating whether to save the attention map during cross-attention.

    Methods:
        save_attn_gradients(attn_gradients): Saves the attention gradients.
        get_attn_gradients(): Retrieves the saved attention gradients.
        save_attention_map(attention_map): Saves the attention map during cross-attention.
        get_attention_map(): Retrieves the saved attention map.
        swapaxes_for_scores(x): Swaps axes of the input tensor to match the shape required for attention scores calculation.
        forward(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            Constructs the multi-head attention mechanism based on the given inputs and parameters.

    Note:
        The class forwardor (__init__ method) raises a ValueError if the hidden size is not a multiple of the number of attention heads.
    """
    def __init__(self, config, is_cross_attention=False):
        """
        This method initializes a Blip2QFormerMultiHeadAttention object.

        Args:
            self: The instance of the Blip2QFormerMultiHeadAttention class.
            config: A configuration object containing model-specific settings and hyperparameters.
            is_cross_attention (bool, optional): A flag indicating if the attention is cross-attention. Defaults to False.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads.
        """
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        """
        Saves the attention gradients for a specific instance of Blip2QFormerMultiHeadAttention.

        Args:
            self (Blip2QFormerMultiHeadAttention): An instance of the Blip2QFormerMultiHeadAttention class.
            attn_gradients: The attention gradients to be saved. This should be a tensor or array-like object.

        Returns:
            None.

        Raises:
            None.
        """
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        """
        Retrieve the attention gradients for the Blip2QFormerMultiHeadAttention.

        Args:
            self (Blip2QFormerMultiHeadAttention): An instance of the Blip2QFormerMultiHeadAttention class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        """
        Saves the attention map generated by the Blip2QFormerMultiHeadAttention model.

        Args:
            self: An instance of the Blip2QFormerMultiHeadAttention class.
            attention_map: A tensor representing the attention map generated by the model. The shape of the tensor is
               expected to be [batch_size, num_heads, max_seq_len, max_seq_len].

        Returns:
            None

        Raises:
            TypeError: If the attention map is not a tensor.
            ValueError: If the attention map is not of the expected shape.
        """
        self.attention_map = attention_map

    def get_attention_map(self):
        """
        Returns the attention map of the Blip2QFormerMultiHeadAttention object.

        Args:
            self (Blip2QFormerMultiHeadAttention): The instance of the Blip2QFormerMultiHeadAttention class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.attention_map

    def swapaxes_for_scores(self, x):
        """
        Performs the 'swapaxes_for_scores' operation in the Blip2QFormerMultiHeadAttention class.

        Args:
            self (Blip2QFormerMultiHeadAttention): An instance of the Blip2QFormerMultiHeadAttention class.
            x (Tensor): The input tensor to be processed. It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: The transformed tensor after applying the 'swapaxes_for_scores' operation. It has a shape of
            (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            None.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        This method 'forward' is defined in the class 'Blip2QFormerMultiHeadAttention' and
        is used to perform multi-head attention computation. It takes 8 parameters:

        Args:
            self: (object) The instance of the class.
            hidden_states: (tensor) The input tensor representing the hidden states.
            attention_mask: (tensor, optional) A tensor representing the attention mask. Default is None.
            head_mask: (tensor, optional) A tensor representing the head mask. Default is None.
            encoder_hidden_states: (tensor, optional) The hidden states of the encoder if cross-attention is performed. Default is None.
            encoder_attention_mask: (tensor, optional) The attention mask for the encoder if cross-attention is performed. Default is None.
            past_key_value: (tuple, optional) A tuple representing the past key and value. Default is None.
            output_attentions: (bool) A boolean flag indicating whether to output attentions.

        Returns:
            None.

        Raises:
            ValueError: If the position_embedding_type is not one of ('relative_key', 'relative_key_query').
            NotImplementedError: If the position_embedding_type is not recognized or implemented.
        """
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.swapaxes_for_scores(self.key(encoder_hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))
            key_layer = ops.cat([past_key_value[0], key_layer], dim=2)
            value_layer = ops.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            seq_length = hidden_states.shape[1]
            position_ids_l = ops.arange(seq_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(seq_length, dtype=mindspore.int64).view(1, -1)
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
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = ops.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Blip2QFormer
class Blip2QFormerSelfOutput(nn.Module):

    """
    This class represents the output of the Blip2QFormerSelf layer in a QFormer model.
    It applies a series of transformations to the input hidden states and returns the final output.

    This class inherits from the nn.Module class.

    Attributes:
        dense (nn.Linear): The dense layer used for linear transformation of the hidden states.
        LayerNorm (nn.LayerNorm): The layer normalization module used for normalizing the hidden states.
        dropout (nn.Dropout): The dropout module used for applying dropout to the hidden states.

    Methods:
        forward(hidden_states, input_tensor):
            Applies the transformations to the input hidden states and returns the final output.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the Blip2QFormerSelfOutput class.

        Args:
            self: The object instance.
            config:
                An instance of the configuration class which contains the configuration parameters for the model.

                - Type: object
                - Purpose: The configuration parameters for the Blip2QFormerSelfOutput class.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method to forward the self output of a Blip2QFormer model.

        Args:
            self (Blip2QFormerSelfOutput): Instance of the Blip2QFormerSelfOutput class.
            hidden_states (mindspore.Tensor): Tensor representing the hidden states.
                This tensor is passed through the dense layer and dropout before being combined with the input_tensor.
            input_tensor (mindspore.Tensor): Tensor representing the input to be combined with the hidden states.
                This tensor is added to the normalized hidden_states after passing through the dense layer and dropout.

        Returns:
            mindspore.Tensor: The forwarded tensor representing the self output of the Blip2QFormer model.
                This tensor is the result of processing the hidden_states and input_tensor through the designated layers.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Blip2QFormerAttention(nn.Module):

    """
    This class represents the attention mechanism used in the Blip2QFormer model. It is a subclass of the nn.Module class.

    The Blip2QFormerAttention class implements the attention mechanism that is responsible for attending to different
    parts of the input data. It consists of a multi-head attention layer and a self-output layer. The attention layer
    performs attention calculations on the input data, while the self-output layer processes the attention outputs.

    The class provides methods for pruning heads in the attention layer, which allows for reducing the computational
    complexity of the model. The forward method is used to perform the attention calculations and generate the final
    output.

    Methods:
        __init__(self, config, is_cross_attention=False): Initializes the Blip2QFormerAttention object. It takes a
            configuration object and a boolean flag indicating whether the attention is cross-attention or not. It
            initializes the attention and output layers, and sets up the pruned heads.

        prune_heads(self, heads): Prunes the specified heads from the attention layer. This method takes a list of heads
            to be pruned and updates the attention layer accordingly. The pruned heads are removed from the attention calculations,
            reducing the computational complexity of the model.

        forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
            encoder_attention_mask=None, past_key_value=None, output_attentions=False): Performs the attention
            calculations and generates the final output. This method takes the hidden states of the input, along with
            optional attention masks, head masks, encoder hidden states, encoder attention masks, past key-value
            pairs, and a flag indicating whether to output attention weights. It returns a tuple containing the
            attention output and any additional outputs.

    Attributes:
        attention: The multi-head attention layer used for attention calculations.
        output: The self-output layer that processes the attention outputs.
        pruned_heads: A set containing the indices of the pruned heads.

    Note:
        This class assumes the existence of helper functions and variables
        (e.g., find_pruneable_heads_and_indices, prune_linear_layer) that are not defined within the class itself.

    """
    def __init__(self, config, is_cross_attention=False):
        """
        Initializes a Blip2QFormerAttention instance.

        Args:
            self (Blip2QFormerAttention): The instance of the Blip2QFormerAttention class.
            config (object): The configuration object containing settings and parameters for the attention mechanism.
            is_cross_attention (bool, optional): Indicates if the attention mechanism is cross-attention.
                                                 Defaults to False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = Blip2QFormerMultiHeadAttention(config, is_cross_attention)
        self.output = Blip2QFormerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prune the attention heads in the Blip2QFormerAttention class.

        Args:
            self (Blip2QFormerAttention): An instance of the Blip2QFormerAttention class.
            heads (list): A list of integers representing the attention heads to be pruned.

        Returns:
            None

        Raises:
            None

        Description:
            This method prunes the attention heads specified in the 'heads' parameter.
            The attention heads are pruned by removing the corresponding weights and biases from the attention layers.

            - If the 'heads' list is empty, the method returns without performing any pruning.
            - The 'heads' list contains integers representing the attention heads to be pruned.
            - The 'self.attention' attribute refers to the attention layer of the Blip2QFormerAttention class.
            - The 'self.attention.query', 'self.attention.key', and 'self.attention.value' attributes represent
            the weight matrices of the attention layer.
            - The 'self.output.dense' attribute represents the weight matrix of the output layer.
            - The 'index' variable is obtained by calling the 'find_pruneable_heads_and_indices' function,
            which determines the indices of the pruneable attention heads based on the current configuration of the attention layer.
            - The 'prune_linear_layer' function is called to prune the attention heads by removing the corresponding
            weights and biases from the attention and output layers.
            - The 'self.attention.num_attention_heads' attribute is updated by subtracting the number of pruned heads
            from the current number of attention heads.
            - The 'self.attention.all_head_size' attribute is updated based on the new number of attention heads and
            the attention head size.
            - The 'self.pruned_heads' attribute is updated by adding the pruned heads to the set of previously pruned heads.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
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
        Constructs the attention mechanism of Blip2QFormer model.

        Args:
            self (Blip2QFormerAttention): The instance of the Blip2QFormerAttention class.
            hidden_states (mindspore.Tensor): The input hidden states.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor.
                Shape: (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]): The head mask tensor.
                Shape: (num_heads, sequence_length, sequence_length).
            encoder_hidden_states (Optional[mindspore.Tensor]): The encoder hidden states tensor.
                Shape: (batch_size, encoder_sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): The encoder attention mask tensor.
                Shape: (batch_size, encoder_sequence_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The tuple of past key-value tensors.
                Shape: ((past_key, past_value)).
            output_attentions (Optional[bool]): Flag to indicate whether to output attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor.
                Shape: (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        """
        self_outputs = self.attention(
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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Blip2QFormer
class Blip2QFormerIntermediate(nn.Module):

    """
    Blip2QFormerIntermediate represents an intermediate layer in a QFormer neural network model.

    This class inherits from nn.Module and is responsible for processing hidden states by applying a dense layer
    followed by an activation function specified in the configuration.

    Attributes:
        dense (nn.Linear): A dense layer with the specified hidden and intermediate sizes.
        intermediate_act_fn (function): The activation function to be applied to the hidden states.

    Methods:
        __init__: Initializes the Blip2QFormerIntermediate instance with the given configuration.
        forward: Processes the hidden states by
            applying the dense layer and the intermediate activation function, then returns the processed hidden states.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the Blip2QFormerIntermediate class.

        Args:
            self: The object itself.
            config:
                An instance of the configuration class containing the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str or function): The activation function for the hidden layer.

                    - If it is a string, it should be one of the supported activation function names.
                    - If it is a function, it should be a callable object representing the activation function.
                    - Default is None.

        Returns:
            None.

        Raises:
            None.

        Note:
            - The 'self' parameter is automatically passed and refers to the instance of the class.
            - The 'config' parameter should be an instance of the configuration class with the necessary attributes.
            - The 'hidden_act' attribute can be either a string representing a supported activation function or
            a callable function object that acts as the activation function.
            - The 'dense' attribute is initialized as an instance of the nn.Linear class with the 'hidden_size'
            and 'intermediate_size' attributes from the 'config' parameter.
            - The 'intermediate_act_fn' attribute is set based on the type of the 'hidden_act' attribute.
            If it is a string, it is mapped to the corresponding activation function from the ACT2FN dictionary.
            If it is a function, it is directly assigned to the 'intermediate_act_fn' attribute.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the intermediate representation of the Blip2QFormer model.

        Args:
            self (Blip2QFormerIntermediate): The instance of the Blip2QFormerIntermediate class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states of the model.
            It is expected to be a tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The output tensor representing the intermediate representation of the model.
                It is of the same shape as the input hidden_states tensor, with each value being transformed based on
                the operations performed within the method.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Blip2QFormer
class Blip2QFormerOutput(nn.Module):

    """
    The Blip2QFormerOutput class represents a custom output layer for a transformer model, specifically designed for a Blip2Q model.
    This class inherits from nn.Module and includes methods for initializing the layer and forwarding the output.

    Attributes:
        dense (nn.Linear): A fully connected layer to transform the hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module to normalize the hidden states.
        dropout (nn.Dropout): A dropout layer to apply dropout to the hidden states.

    Methods:
        __init__: Initializes the Blip2QFormerOutput layer with the provided configuration.
        forward: Constructs the output layer by applying dense transformation, dropout, layer normalization, and
            addition with input tensor.

    Note:
        The Blip2QFormerOutput class is designed for a specific transformer model architecture and should be used accordingly.
    """
    def __init__(self, config):
        """
        Initializes an instance of the Blip2QFormerOutput class.

        Args:
            self (Blip2QFormerOutput): An instance of the Blip2QFormerOutput class.
            config: An object of type 'config' containing the configuration options for the model.

        Returns:
            None

        Raises:
            None

        This method sets up the initial state of the Blip2QFormerOutput instance by initializing the following attributes:

        - dense (nn.Linear): A dense layer that maps the input features to the intermediate size specified in the config.
        - LayerNorm (nn.LayerNorm): A layer normalization module that normalizes the hidden size of the input tensor
            using the epsilon value specified in the config.
        - dropout (nn.Dropout): A dropout layer that applies dropout with the hidden dropout probability specified in the config.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output of the Blip2QFormer model.

        Args:
            self (Blip2QFormerOutput): An instance of the Blip2QFormerOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor.
                This tensor represents the output of the dense layer.
                Shape: (batch_size, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor.
                This tensor represents the input to be added to the hidden states tensor.
                Shape: (batch_size, hidden_size).

        Returns:
            mindspore.Tensor: The forwarded output tensor.
                This tensor represents the final output of the forward method.
                Shape: (batch_size, hidden_size).

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Blip2QFormerLayer(nn.Module):

    """
    This class represents a layer of a Blip2QFormer model, which is designed for efficient query processing in deep learning models.
    The Blip2QFormerLayer class contains methods for initializing the layer, forwarding the layer, and processing feed-forward operations.
    It inherits from nn.Module.

    Methods:
        __init__(self, config, layer_idx): Initializes the Blip2QFormerLayer instance with the given configuration
            and layer index.
        forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
            encoder_attention_mask=None, past_key_value=None, output_attentions=False, query_length=0):
            Constructs the layer by processing attention mechanisms and feed-forward operations.
        feed_forward_chunk(self, attention_output): Processes the feed-forward chunk operation for the layer.
        feed_forward_chunk_query(self, attention_output): Processes the feed-forward chunk operation specifically
            for query processing.

    Attributes:
        chunk_size_feed_forward: Size of the chunk used for feed-forward processing.
        seq_len_dim: Dimension for sequence length.
        attention: Blip2QFormerAttention instance for self-attention mechanisms.
        crossattention: Blip2QFormerAttention instance for cross-attention mechanisms if applicable.
        has_cross_attention: Flag indicating if the layer has cross-attention mechanisms.
        intermediate_query: Blip2QFormerIntermediate instance for intermediate query processing.
        output_query: Blip2QFormerOutput instance for output query processing.
        layer_idx: Index of the layer within the model.

    Note:
        This class is part of a larger Blip2QFormer model architecture and should be used in conjunction with
        other components for optimal performance.
    """
    def __init__(self, config, layer_idx):
        """
        Initializes a Blip2QFormerLayer object.

        Args:
            self: The instance of the Blip2QFormerLayer class.
            config:
                An object containing the configuration parameters for the Blip2QFormerLayer.

                - Type:
                - Purpose: The configuration object that contains the necessary settings for the layer.
            layer_idx: An integer representing the index of the layer.

                - Type: int
                - Purpose: The index of the layer within the model.
                - Restrictions: Must be a non-negative integer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(config)

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate_query = Blip2QFormerIntermediate(config)
        self.output_query = Blip2QFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        """
        Constructs a Blip2QFormerLayer.

        This method takes 9 parameters:

        - self: The instance of the Blip2QFormerLayer class.
        - hidden_states: The hidden states input tensor of shape (batch_size, sequence_length, hidden_size).
        - attention_mask (optional): The attention mask tensor of shape (batch_size, sequence_length).
        It indicates which tokens should be attended to and which ones should not.
        - head_mask (optional): The head mask tensor of shape (num_heads, sequence_length, sequence_length).
        It indicates which heads should be masked out.
        - encoder_hidden_states (optional): The hidden states output tensor from the encoder layer of shape
        (batch_size, sequence_length, hidden_size). Required for cross-attention layers.
        - encoder_attention_mask (optional): The attention mask tensor for the encoder layer of shape
        (batch_size, sequence_length). Required for cross-attention layers.
        - past_key_value (optional): The past key-value tensor of shape
        (2, batch_size, num_heads, past_sequence_length, hidden_size).
        It contains the previous key-value pairs. If not provided, it will be set to None.
        - output_attentions (optional): Whether to output attention weights. Defaults to False.
        - query_length (optional): The length of the query sequence.
        If greater than 0, cross-attention with the encoder layer will be applied. Defaults to 0.

        Returns:
            outputs:
                A tuple containing the layer output tensor, attention outputs (if output_attentions=True),
                and the present key-value tensor.
                The layer output tensor has shape (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If encoder_hidden_states is not provided for cross-attention layers.

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
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = ops.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        This method 'feed_forward_chunk' is part of the 'Blip2QFormerLayer' class and is used for processing
        the attention_output in a feed-forward manner.

        Args:
            self (Blip2QFormerLayer): The instance of the Blip2QFormerLayer class.
            attention_output (tensor): The attention output tensor to be processed by the feed-forward chunk.
                It is expected to be a tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method does not return any value. It processes the attention_output and updates internal layer states.

        Raises:
            None.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        """
        Feed forward chunk query method in the Blip2QFormerLayer class.

        This method applies a feed forward operation to the attention output of the Blip2QFormerLayer.
        It consists of two steps: intermediate query and output query.

        Args:
            self (Blip2QFormerLayer): An instance of the Blip2QFormerLayer class.
            attention_output: The attention output to be used for the feed forward operation.
                It should be a tensor or array-like object.

        Returns:
            None: This method does not return any value. It modifies the layer output in place.

        Raises:
            None.
        """
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class Blip2QFormerEncoder(nn.Module):

    """
    Blip2QFormerEncoder is a class representing an encoder for a Blip2QFormer neural network model.
    It consists of multiple layers of Blip2QFormerLayer instances and supports various functionalities
    such as gradient checkpointing, caching, and outputting hidden states and attentions.

    Attributes:
        config (dict): Configuration settings for the encoder.
        layer (nn.ModuleList): List of Blip2QFormerLayer instances representing the encoder layers.
        gradient_checkpointing (bool): Flag indicating whether gradient checkpointing is enabled.

    Methods:
        __init__(config): Initializes the Blip2QFormerEncoder with the given configuration.
        forward(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                  encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False,
                  output_hidden_states=False, return_dict=True, query_length=0):
                  Constructs the encoder by processing input hidden states through each layer
                  and optionally outputting hidden states and attentions.

    Returns:
        BaseModelOutputWithPastAndCrossAttentions: An object containing the final hidden states, past key values,
            hidden states of all layers, self-attentions of all layers, and cross-attentions of all layers.
    """
    def __init__(self, config):
        """
        __init__

        Initializes the Blip2QFormerEncoder.

        Args:
            self: Blip2QFormerEncoder
                The instance of the Blip2QFormerEncoder class.
            config: dict
                A dictionary containing the configuration parameters for the encoder.
                It includes settings such as the number of hidden layers and other configuration options.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        """
        This method forwards the Blip2QFormerEncoder by processing the input hidden states through multiple layers.

        Args:
            self (object): The instance of the Blip2QFormerEncoder class.
            hidden_states (torch.Tensor): The input hidden states to be processed.
            attention_mask (torch.Tensor, optional): Mask to avoid performing attention on padding tokens.
            head_mask (List[torch.Tensor], optional): Mask for attention heads in each layer.
            encoder_hidden_states (torch.Tensor, optional): Hidden states from the encoder.
            encoder_attention_mask (torch.Tensor, optional): Mask for encoder attention.
            past_key_values (List[torch.Tensor], optional): Cached key values from previous decoding steps.
            use_cache (bool, optional): Flag to indicate whether caching is used.
            output_attentions (bool): Flag to determine if attentions should be output.
            output_hidden_states (bool): Flag to determine if hidden states should be output.
            return_dict (bool): Flag to indicate whether the output should be returned as a dictionary.
            query_length (int): Length of the query.

        Returns:
            None.

        Raises:
            Warning: If `use_cache=True` is used in combination with gradient checkpointing,
                it raises a warning and sets `use_cache=False`.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
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
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
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


class Blip2QFormerModel(Blip2PreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """
    def __init__(self, config: Blip2QFormerConfig):
        """
        Initializes a new instance of the Blip2QFormerModel class.

        Args:
            self: The instance of the class.
            config (Blip2QFormerConfig): The configuration object containing the model settings.
                It should be an instance of Blip2QFormerConfig class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes the Blip2QFormerModel instance with the provided configuration.
            It sets the instance variables as follows:

            - self.config: The provided config object is stored as an instance variable.
            - self.layernorm: A LayerNorm module is created with the hidden size specified in the config.
            The epsilon value for LayerNorm is set to the value specified in the config.
            - self.dropout: A Dropout module is created with the dropout probability specified in the config.
            - self.encoder: An instance of Blip2QFormerEncoder class is created with the provided config.

        Note:
            After the initialization, self.post_init() method is called to perform any additional
            post-initialization steps.
        """
        super().__init__(config)
        self.config = config

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.encoder = Blip2QFormerEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the Blip2QFormerModel.

        Args:
            self: The reference to the current instance of the Blip2QFormerModel class.

        Returns:
            word_embeddings: The method returns the word embeddings for the input.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the Blip2QFormerModel.

        Args:
            self (Blip2QFormerModel): The instance of the Blip2QFormerModel class.
            value: The input embeddings to be set. It can be of any valid type.

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

    def get_extended_attention_mask(
        self,
        attention_mask: mindspore.Tensor,
        input_shape: Tuple[int],
        has_query: bool = False,
    ) -> mindspore.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`mindspore.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `mindspore.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Args:
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of:
                shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
                value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
                used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
                value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
                `(batch_size, sequence_length)`.
            use_cache (`bool`, `optional`):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.layernorm(query_embeds)
        embedding_output = self.dropout(embedding_output)

        input_shape = embedding_output.shape[:-1]
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = ops.ones(batch_size, seq_length + past_key_values_length)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].shape
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(*encoder_hidden_shape)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

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
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

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


class Blip2Model(Blip2PreTrainedModel):

    """
    A class representing the Blip2Model for multimodal language understanding tasks.

    Blip2Model is a multimodal transformer model that combines vision and text inputs to generate high-level
    representations and perform tasks such as image captioning, visual question answering, and multimodal
    language generation.

    This class extends the Blip2PreTrainedModel, which is the base class for all models in the Blip2 project.

    Attributes:
        vision_model (Blip2VisionModel): The vision model that processes the image inputs.
        query_tokens (Parameter): Query tokens used in the QFormer model.
        qformer (Blip2QFormerModel): The QFormer model that processes the query tokens and image embeddings.
        language_projection (nn.Linear): Projection layer that maps the QFormer output to the input size of the language model.
        language_model (Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]): The language model used for text processing.
        _tied_weights_keys (List[str]): List of tied weights keys for the language model.

    Methods:
        __init__: Initializes the Blip2Model with the given configuration.
        get_input_embeddings: Returns the input embeddings of the language model.
        set_input_embeddings: Sets the input embeddings of the language model.
        set_output_embeddings: Sets the output embeddings of the language model.
        get_output_embeddings: Returns the output embeddings of the language model.
        get_encoder: Returns the encoder of the language model.
        get_decoder: Returns the decoder of the language model.
        _tie_weights: Ties the weights of the encoder and decoder in the language model.
        get_text_features: Retrieves the text features from the language model.
        get_image_features: Retrieves the image features from the vision model.
        get_qformer_features: Retrieves the query transformer (QFormer) features from the vision model.
        forward: Constructs the Blip2Model with the given inputs and returns the model outputs.

    Example:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model
        ...
        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        ...
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> outputs = model.get_image_features(**inputs)
        ```
    """
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        """
        Initialize the Blip2Model with the specified configuration.

        Args:
            self: The instance of the Blip2Model class.
            config (Blip2Config): An instance of Blip2Config containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = Parameter(ops.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings from the Blip2Model.

        Args:
            self (Blip2Model): An instance of the Blip2Model class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the Blip2Model.

        Args:
            self (Blip2Model): The instance of the Blip2Model class.
            value: The input embeddings to be set for the language model.
                It should be of type torch.Tensor or any compatible type.

        Returns:
            None.

        Raises:
            TypeError: If the value provided is not of the expected type.
            ValueError: If the value provided is invalid or cannot be used for setting input embeddings.
        """
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the Blip2Model.

        Args:
            self (Blip2Model): The instance of the Blip2Model class.
            new_embeddings (object): The new output embeddings to be set for the language model.

        Returns:
            None.

        Raises:
            None.
        """
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        """
        Returns the output embeddings of the Blip2Model.

        Args:
            self: Blip2Model - The instance of the Blip2Model class.

        Returns:
            nn.Module: The output embeddings of the Blip2Model language model.

        Raises:
            None.
        """
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        """
        Method to retrieve the encoder from the language model within the Blip2Model class.

        Args:
            self (Blip2Model):
                The instance of the Blip2Model class.

                - This parameter refers to the current instance of the Blip2Model class.

        Returns:
            None:
                This method returns None as it retrieves the encoder from the language model.

        Raises:
            None.
        """
        return self.language_model.get_encoder()

    def get_decoder(self):
        """
        This method returns the decoder from the language model.

        Args:
            self: The instance of the Blip2Model class.

        Returns:
            None:
                This method returns the decoder obtained from the language model.

        Raises:
            None
        """
        return self.language_model.get_decoder()

    def _tie_weights(self):
        """
        Method to tie weights in the Blip2Model class.

        Args:
            self: The instance of the Blip2Model class.

        Returns:
            None: This method does not return any value.

        Raises:
            None.
        """
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def get_text_features(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""

        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(mindspore.Tensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.

        Example:
            ```python
            >>> import torch
            >>> from transformers import AutoTokenizer, Blip2Model
            ...
            >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
            >>> inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")
            >>> text_features = model.get_text_features(**inputs)
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        return text_outputs

    def get_image_features(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""

        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `mindspore.Tensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.

        Example:
            ```python
            >>> import torch
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, Blip2Model
            ...
            >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
            ...
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> inputs = processor(images=image, return_tensors="pt")
            >>> image_outputs = model.get_image_features(**inputs)
            ```
        """
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

        return vision_outputs

    def get_qformer_features(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""

        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `mindspore.Tensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.

        Example:
            ```python
            >>> import torch
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import Blip2Processor, Blip2Model
            ...
            >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> inputs = processor(images=image, return_tensors="pt")
            >>> qformer_outputs = model.get_qformer_features(**inputs)
            ```
        """
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

        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = ops.ones(*image_embeds.shape[:-1], dtype=mindspore.int64)

        query_tokens = ops.broadcast_to(self.query_tokens, (image_embeds.shape[0], -1, -1))
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return query_outputs

    def forward(
        self,
        pixel_values: mindspore.Tensor,
        input_ids: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""

        Returns:
            `Union[Tuple, Blip2ForConditionalGenerationModelOutput]`

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import Blip2Processor, Blip2Model
            >>> import torch
            ...
            ...
            >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", ms_dtype=torch.float16)
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> prompt = "Question: how many cats are there? Answer:"
            >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch.float16)
            ...
            >>> outputs = model(**inputs)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = ops.ones(*image_embeds.shape[:-1], dtype=mindspore.int64)

        query_tokens = ops.broadcast_to(self.query_tokens, (image_embeds.shape[0], -1, -1))
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = ops.ones(
            *language_model_inputs.shape[:-1], dtype=mindspore.int64
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = ops.cat([language_model_inputs, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        attention_mask = ops.cat([language_model_attention_mask, attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                logits = logits[:, -labels.shape[1] :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]

                # Flatten the tokens
                loss = F.cross_entropy(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )


class Blip2ForConditionalGeneration(Blip2PreTrainedModel):

    """
    The `Blip2ForConditionalGeneration` class is a model for image captioning and visual question answering. It is a conditional generator that takes as input an image and generates captions or answers to
    questions based on the image.

    The class inherits from the `Blip2PreTrainedModel` class.

    Example:
        ```python
        >>> from transformers import Blip2ForConditionalGeneration
        ...
        >>>model = Blip2ForConditionalGeneration()
        ```

    Methods:
        `__init__`: Initializes the Blip2ForConditionalGeneration model with the given configuration.
        `get_input_embeddings`: Returns the input embeddings of the language model.
        `set_input_embeddings`: Sets the input embeddings of the language model to the given value.
        `set_output_embeddings`: Sets the output embeddings of the language model to the given new embeddings.
        `get_output_embeddings`: Returns the output embeddings of the language model.
        `get_encoder`: Returns the encoder of the language model.
        `get_decoder`: Returns the decoder of the language model.
        `_tie_weights`: Ties the weights of the encoder and decoder if the model is not using a decoder-only language model.
        `forward`: Constructs the Blip2ForConditionalGeneration model with the given inputs and returns the output.
        `generate`: Generates captions or answers based on the given image and optionally the input sequence and attention mask.

    Please refer to the docstrings of each method for more detailed information.
    """
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        """
        This method initializes an instance of the Blip2ForConditionalGeneration class.

        Args:
            self: The instance of the Blip2ForConditionalGeneration class.
            config (Blip2Config): An object containing the configuration settings for the Blip2 model.
                It is used to initialize the various components of the model, such as vision model, query tokens, qformer,
                language projection, and language model. The config parameter is required and must be of type Blip2Config.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = Parameter(ops.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the Blip2 language model.

        Args:
            self: An instance of the Blip2ForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the underlying language model used in Blip2ForConditionalGeneration.
        It returns None since it directly calls the 'get_input_embeddings' method of the language model and does not modify
        or process the embeddings further.

        Please note that this method takes only one parameter, which is the instance of the Blip2ForConditionalGeneration class itself (self).
        """
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the Blip2ForConditionalGeneration model.

        Args:
            self (Blip2ForConditionalGeneration): The instance of the Blip2ForConditionalGeneration class.
            value (torch.Tensor): The input embeddings to be set for the language model.

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings for the underlying language model of Blip2ForConditionalGeneration.
        The input embeddings are used to represent the input tokens during the model's forward pass.
        The 'value' parameter should be a tensor of shape (vocab_size, embedding_dim), where 'vocab_size'
        represents the size of the vocabulary and 'embedding_dim' represents the dimensionality of the embedding space.
        The method updates the input embeddings of the language model with the provided 'value'.

        Example:
            ```python
            >>> model = Blip2ForConditionalGeneration()
            >>> embeddings = torch.randn(10000, 300)
            >>> model.set_input_embeddings(embeddings)
            ```
        """
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the Blip2ForConditionalGeneration class.

        Args:
            self (Blip2ForConditionalGeneration): The instance of the Blip2ForConditionalGeneration class.
            new_embeddings: The new embeddings to be set for the language model.

        Returns:
            None.

        Raises:
            None.
        """
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        """
        Get the output embeddings of the Blip2ForConditionalGeneration class.

        Args:
            self (Blip2ForConditionalGeneration): The instance of the Blip2ForConditionalGeneration class.

        Returns:
            nn.Module: The output embeddings of the language model.

        Raises:
            None.

        This method returns the output embeddings of the language model used in the Blip2ForConditionalGeneration class.
        The output embeddings are retrieved by calling the 'get_output_embeddings()' method of the language model.

        Note:
            The output embeddings are typically used to map the hidden states of the language model to the vocabulary
            space. They can be further processed or used in downstream tasks.
        """
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        """
        Method to retrieve the encoder from the language model for Blip2ForConditionalGeneration.

        Args:
            self:
                The instance of the Blip2ForConditionalGeneration class.

                - Type: Blip2ForConditionalGeneration
                - Purpose: To access the methods and attributes of the Blip2ForConditionalGeneration class.

        Returns:
            None:
                The method returns None as it retrieves the encoder from the language model
                and does not return any specific value.

                - Type: None
                - Purpose: Indicate that the method successfully retrieved the encoder from the language model.

        Raises:
            No specific exceptions are raised by this method.
        """
        return self.language_model.get_encoder()

    def get_decoder(self):
        """
        This method returns the decoder from the language model associated with the Blip2ForConditionalGeneration instance.

        Args:
            self (Blip2ForConditionalGeneration): The instance of the Blip2ForConditionalGeneration class.
                It is used to access the language model and retrieve the decoder.

        Returns:
            None:
                This method does not return any value directly.
                It retrieves the decoder from the language model associated with Blip2ForConditionalGeneration.

        Raises:
            None
        """
        return self.language_model.get_decoder()

    def _tie_weights(self):
        """
        Ties the weights between encoder and decoder in the Blip2ForConditionalGeneration model.

        Args:
            self: The instance of the Blip2ForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def forward(
        self,
        pixel_values: mindspore.Tensor,
        input_ids: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""

        Returns:
            Union[Tuple, Blip2ForConditionalGenerationModelOutput]

        Example:
            Prepare processor, model and image input

            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
            >>> import torch
            ...
            ...
            >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            >>> model = Blip2ForConditionalGeneration.from_pretrained(
            ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, ms_dtype=torch.float16
            ... )  # doctest: +IGNORE_RESULT
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ```

            Image captioning (without providing a text prompt):

            ```python
            >>> inputs = processor(images=image, return_tensors="pt").to(torch.float16)
            ...
            >>> generated_ids = model.generate(**inputs)
            >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            >>> print(generated_text)
            two cats laying on a couch
            ```

            Visual question answering (prompt = question):

            ```python
            >>> prompt = "Question: how many cats are there? Answer:"
            >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(dtype=torch.float16)
            ...
            >>> generated_ids = model.generate(**inputs)
            >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            >>> print(generated_text)
            two
            ```

            Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
            This greatly reduces the amount of memory used by the model while maintaining the same performance.

            ```python
            >>> model = Blip2ForConditionalGeneration.from_pretrained(
            ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, ms_dtype=torch.bfloat16
            ... )  # doctest: +IGNORE_RESULT
            ...
            >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(dtype=torch.bfloat16)
            ...
            >>> generated_ids = model.generate(**inputs)
            >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            >>> print(generated_text)
            two
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]
        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = ops.ones(*image_embeds.shape[:-1], dtype=mindspore.int64)

        query_tokens = ops.broadcast_to(self.query_tokens, (image_embeds.shape[0], -1, -1))
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = ops.ones(
            *language_model_inputs.shape[:-1], dtype=mindspore.int64
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = ops.cat([language_model_inputs, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        attention_mask = ops.cat([language_model_attention_mask.astype(mindspore.bool_), attention_mask.astype(mindspore.bool_)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                logits = logits[:, -labels.shape[1] :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]

                # Flatten the tokens
                loss = F.cross_entropy(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    def generate(
        self,
        pixel_values: mindspore.Tensor,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        **generate_kwargs,
    ) -> mindspore.Tensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`mindspore.Tensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`mindspore.Tensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`mindspore.Tensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

        image_attention_mask = ops.ones(*image_embeds.shape[:-1], dtype=mindspore.int64)

        query_tokens = ops.broadcast_to(self.query_tokens, (image_embeds.shape[0], -1, -1))
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = ops.ones(
            *language_model_inputs.shape[:-1], dtype=mindspore.int64
        )
        if input_ids is None:
            input_ids = (
                mindspore.Tensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
            )
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        attention_mask = ops.cat([language_attention_mask, attention_mask], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = ops.cat([language_model_inputs, inputs_embeds], dim=1)

        # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
        # -1 is to account for the prepended BOS after `generate.`
        # TODO (joao, raushan): refactor `generate` to avoid these operations with VLMs
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs

__all__ = [
    "Blip2Model",
    "Blip2QFormerModel",
    "Blip2PreTrainedModel",
    "Blip2ForConditionalGeneration",
    "Blip2VisionModel",
]
