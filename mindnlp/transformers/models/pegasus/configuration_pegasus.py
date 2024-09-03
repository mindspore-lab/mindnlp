# coding=utf-8
# Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd
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
""" PEGASUS model configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


__all__ = [
    "PegasusConfig",
    "PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP"
]


PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pegasus-large": "https://hf-mirror.com/google/pegasus-large/resolve/main/config.json"
}


class PegasusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PegasusModel`]. It is used to instantiate an
    PEGASUS model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PEGASUS
    [google/pegasus-large](https://hf-mirror.com/google/pegasus-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the PEGASUS model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`PegasusModel`] or [`TFPegasusModel`].
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
        max_position_embeddings (`int`, *optional*, defaults to 1024):
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
        forced_eos_token_id (`int`, *optional*, defaults to 1):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.

    Example:
        ```python
        >>> from transformers import PegasusConfig, PegasusModel
        ...
        >>> # Initializing a PEGASUS google/pegasus-large style configuration
        >>> configuration = PegasusConfig()
        ...
        >>> # Initializing a model (with random weights) from the google/pegasus-large style configuration
        >>> model = PegasusModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "pegasus"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        initializer_range=0.02,
        decoder_start_token_id=0,
        scale_embedding=False,
        pad_token_id=0,
        eos_token_id=1,
        forced_eos_token_id=1,
        **kwargs,
    ):
        """
        Initializes a new PegasusConfig object with the provided configuration parameters.

        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Default is 50265.
            max_position_embeddings (int, optional): The maximum number of tokens in a sequence. Default is 1024.
            encoder_layers (int, optional): The number of layers in the encoder. Default is 12.
            encoder_ffn_dim (int, optional): The dimension of the feedforward network in the encoder layers. Default is 4096.
            encoder_attention_heads (int, optional): The number of attention heads in the encoder layers. Default is 16.
            decoder_layers (int, optional): The number of layers in the decoder. Default is 12.
            decoder_ffn_dim (int, optional): The dimension of the feedforward network in the decoder layers. Default is 4096.
            decoder_attention_heads (int, optional): The number of attention heads in the decoder layers. Default is 16.
            encoder_layerdrop (float, optional): The probability of dropping a layer in the encoder. Default is 0.0.
            decoder_layerdrop (float, optional): The probability of dropping a layer in the decoder. Default is 0.0.
            use_cache (bool, optional): Whether to use caching for the model. Default is True.
            is_encoder_decoder (bool, optional): Whether the model is an encoder-decoder model. Default is True.
            activation_function (str, optional): The activation function to be used. Default is 'gelu'.
            d_model (int, optional): The dimension of the model. Default is 1024.
            dropout (float, optional): The dropout probability. Default is 0.1.
            attention_dropout (float, optional): The dropout probability for attention layers. Default is 0.0.
            activation_dropout (float, optional): The dropout probability for activation layers. Default is 0.0.
            init_std (float, optional): The standard deviation for weight initialization. Default is 0.02.
            initializer_range (float, optional): The range for weight initialization. Default is 0.02.
            decoder_start_token_id (int, optional): The token id for the start of the decoder sequence. Default is 0.
            scale_embedding (bool, optional): Whether to scale embeddings. Default is False.
            pad_token_id (int, optional): The token id for padding. Default is 0.
            eos_token_id (int, optional): The token id for end of sequence. Default is 1.
            forced_eos_token_id (int, optional): The token id for forced end of sequence. Default is 1.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            None.
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
        self.initializer_range = initializer_range
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

    @property
    def num_attention_heads(self) -> int:
        """
        Returns the number of attention heads in the Pegasus model's encoder.

        Args:
            self (PegasusConfig): The current instance of the PegasusConfig class.

        Returns:
            int: The number of attention heads used in the encoder of the Pegasus model.

        Raises:
            None.


        The `num_attention_heads` method returns an integer value representing the number of attention heads used
        in the encoder of the Pegasus model. Attention heads are a key component of transformer models, and they enable
        the model to focus on different parts of the input sequence during processing. By varying the number of
        attention heads, the model can capture different levels of information and dependencies in the input data.

        This method is a property, which means that it can be accessed as an attribute without needing to call
        it explicitly as a function. When accessed, it directly returns the number of attention heads specified in the
        `encoder_attention_heads` attribute of the current instance of the PegasusConfig class.

        Note that the `num_attention_heads` method does not take any additional parameters beyond the `self` parameter,
        as it is designed to provide information specific to the current instance of the class.

        Example:
            ```python
            >>> config = PegasusConfig()
            >>> num_heads = config.num_attention_heads
            >>> print(num_heads)
            12
            ```

        In this example, a new instance of the PegasusConfig class is created. The `num_attention_heads` property is
        accessed as an attribute (`config.num_attention_heads`), and the resulting number of attention heads
        (12 in this case) is printed.
        """
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the PegasusConfig object.
        
        Args:
            self: The PegasusConfig object.
        
        Returns:
            int: The hidden size of the PegasusConfig object.
                This value represents the size of the hidden state in the model.
        
        Raises:
            None.
        """
        return self.d_model
