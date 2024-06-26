# coding=utf-8
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" Blenderbot model configuration"""
from ...configuration_utils import PretrainedConfig
from ....utils import logging


logger = logging.get_logger(__name__)


class BlenderbotConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlenderbotModel`]. It is used to instantiate an
    Blenderbot model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Blenderbot
    [facebook/blenderbot-3B](https://huggingface.co/facebook/blenderbot-3B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the Blenderbot model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`BlenderbotModel`] or [`TFBlenderbotModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 128):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.

    Example:
        ```python
        >>> from transformers import BlenderbotConfig, BlenderbotModel
        ... 
        >>> # Initializing a Blenderbot facebook/blenderbot-3B style configuration
        >>> configuration = BlenderbotConfig()
        ... 
        >>> # Initializing a model (with random weights) from the facebook/blenderbot-3B style configuration
        >>> model = BlenderbotModel(configuration)
        ... 
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "blenderbot"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=8008,
        max_position_embeddings=128,
        encoder_layers=2,
        encoder_ffn_dim=10240,
        encoder_attention_heads=32,
        decoder_layers=24,
        decoder_ffn_dim=10240,
        decoder_attention_heads=32,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=2560,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=1,
        scale_embedding=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        encoder_no_repeat_ngram_size=3,
        forced_eos_token_id=2,
        **kwargs,
    ):
        """
        Initialize a BlenderbotConfig instance.
        
        Args:
            vocab_size (int, optional): The size of the vocabulary. Defaults to 8008.
            max_position_embeddings (int, optional): The maximum number of positional embeddings. Defaults to 128.
            encoder_layers (int, optional): The number of encoder layers. Defaults to 2.
            encoder_ffn_dim (int, optional): The dimension of the encoder's feedforward network. Defaults to 10240.
            encoder_attention_heads (int, optional): The number of attention heads in the encoder. Defaults to 32.
            decoder_layers (int, optional): The number of decoder layers. Defaults to 24.
            decoder_ffn_dim (int, optional): The dimension of the decoder's feedforward network. Defaults to 10240.
            decoder_attention_heads (int, optional): The number of attention heads in the decoder. Defaults to 32.
            encoder_layerdrop (float, optional): The probability of dropping a layer in the encoder. Defaults to 0.0.
            decoder_layerdrop (float, optional): The probability of dropping a layer in the decoder. Defaults to 0.0.
            use_cache (bool, optional): Whether to use cache during decoding. Defaults to True.
            is_encoder_decoder (bool, optional): Whether the model is an encoder-decoder architecture. Defaults to True.
            activation_function (str, optional): The activation function to use. Defaults to 'gelu'.
            d_model (int, optional): The dimension of the model. Defaults to 2560.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
            attention_dropout (float, optional): The dropout probability for attention layers. Defaults to 0.0.
            activation_dropout (float, optional): The dropout probability for activation layers. Defaults to 0.0.
            init_std (float, optional): The standard deviation for weight initialization. Defaults to 0.02.
            decoder_start_token_id (int, optional): The token id for the start of the decoder sequence. Defaults to 1.
            scale_embedding (bool, optional): Whether to scale the embeddings. Defaults to False.
            pad_token_id (int, optional): The token id for padding. Defaults to 0.
            bos_token_id (int, optional): The token id for the beginning of sequence. Defaults to 1.
            eos_token_id (int, optional): The token id for the end of sequence. Defaults to 2.
            encoder_no_repeat_ngram_size (int, optional): The size of the no repeat n-gram in the encoder. Defaults to 3.
            forced_eos_token_id (int, optional): The token id for the forced end of sequence. Defaults to 2.
            
        Returns:
            None.
        
        Raises:
            None
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

__all__ = ['BlenderbotConfig']
