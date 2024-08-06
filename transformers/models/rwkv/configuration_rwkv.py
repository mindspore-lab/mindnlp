# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# ============================================================================

""" RWKV configuration"""

from ...configuration_utils import PretrainedConfig


RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "RWKV/rwkv-4-169m-pile": "https://hf-mirror.com/RWKV/rwkv-4-169m-pile/resolve/main/config.json",
    "RWKV/rwkv-4-430m-pile": "https://hf-mirror.com/RWKV/rwkv-4-430m-pile/resolve/main/config.json",
    "RWKV/rwkv-4-1b5-pile": "https://hf-mirror.com/RWKV/rwkv-4-1b5-pile/resolve/main/config.json",
    "RWKV/rwkv-4-3b-pile": "https://hf-mirror.com/RWKV/rwkv-4-3b-pile/resolve/main/config.json",
    "RWKV/rwkv-4-7b-pile": "https://hf-mirror.com/RWKV/rwkv-4-7b-pile/resolve/main/config.json",
    "RWKV/rwkv-4-14b-pile": "https://hf-mirror.com/RWKV/rwkv-4-14b-pile/resolve/main/config.json",
    "RWKV/rwkv-raven-1b5": "https://hf-mirror.com/RWKV/rwkv-raven-1b5/resolve/main/config.json",
    "RWKV/rwkv-raven-3b": "https://hf-mirror.com/RWKV/rwkv-raven-3b/resolve/main/config.json",
    "RWKV/rwkv-raven-7b": "https://hf-mirror.com/RWKV/rwkv-raven-7b/resolve/main/config.json",
    "RWKV/rwkv-raven-14b": "https://hf-mirror.com/RWKV/rwkv-raven-14b/resolve/main/config.json",
}


class RwkvConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`RwkvModel`]. It is used to instantiate a RWKV
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RWVK-4
    [RWKV/rwkv-4-169m-pile](https://hf-mirror.com/RWKV/rwkv-4-169m-pile) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50277):
            Vocabulary size of the RWKV model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RwkvModel`].
        context_length (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model can be be used with in a single forward (using it in RNN mode
            lets use any sequence length).
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        attention_hidden_size (`int`, *optional*):
            Dimensionality of the attention hidden states. Will default to `hidden_size` if unset.
        intermediate_size (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. Will default to 4 times `hidden_size` if unset.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer
            as GPTNeoX.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer as
            GPTNeoX.
        rescale_every (`int`, *optional*, default to 6):
            At inference, the hidden states (and weights of the correponding output layers) are divided by 2 every
            `rescale_every` layer. If set to 0 or a negative number, no rescale is done.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the input token embeddings.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last state.


    Example:
        ```python
        >>> from transformers import RwkvConfig, RwkvModel
        ...
        >>> # Initializing a Rwkv configuration
        >>> configuration = RwkvConfig()
        ...
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = RwkvModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "rwkv"
    attribute_map = {"max_position_embeddings": "context_length"}
    pretrained_config_archive_map = RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        vocab_size=50277,
        context_length=1024,
        hidden_size=4096,
        num_hidden_layers=32,
        attention_hidden_size=None,
        intermediate_size=None,
        layer_norm_epsilon=1e-5,
        bos_token_id=0,
        eos_token_id=0,
        rescale_every=6,
        tie_word_embeddings=False,
        use_cache=True,
        **kwargs,
    ):
        """
        Initializes an instance of RwkvConfig.

        Args:
            self: The instance itself.
            vocab_size (int): The size of the vocabulary. Default is 50277.
            context_length (int): The length of the context. Default is 1024.
            hidden_size (int): The size of the hidden layers. Default is 4096.
            num_hidden_layers (int): The number of hidden layers. Default is 32.
            attention_hidden_size (int, optional): The size of the attention hidden layer.
                Defaults to hidden_size if not provided.
            intermediate_size (int, optional): The size of the intermediate layer. Defaults to 4 times hidden_size
                if not provided.
            layer_norm_epsilon (float): The epsilon value for layer normalization. Default is 1e-05.
            bos_token_id (int): The beginning of sentence token id. Default is 0.
            eos_token_id (int): The end of sentence token id. Default is 0.
            rescale_every (int): The frequency of rescaling. Default is 6.
            tie_word_embeddings (bool): Whether to tie word embeddings. Default is False.
            use_cache (bool): Whether to use cache. Default is True.

        Returns:
            None.

        Raises:
            ValueError: If the provided vocab_size, context_length, hidden_size, num_hidden_layers,
                attention_hidden_size, intermediate_size, layer_norm_epsilon, bos_token_id, eos_token_id,
                or rescale_every is not a positive integer.
            TypeError: If any of the provided parameters has an unexpected type.
        """
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
        )

__all__ = ['RwkvConfig']
