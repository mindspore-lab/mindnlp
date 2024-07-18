# coding=utf-8
# Copyright 2021- NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
""" MEGATRON_BERT model configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)

MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # See all MEGATRON_BERT models at https://hf-mirror.com/models?filter=bert
}


class MegatronBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegatronBertModel`]. It is used to instantiate a
    MEGATRON_BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MEGATRON_BERT
    [nvidia/megatron-bert-uncased-345m](https://hf-mirror.com/nvidia/megatron-bert-uncased-345m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 29056):
            Vocabulary size of the MEGATRON_BERT model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MegatronBertModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`MegatronBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Example:
        ```python
        >>> from transformers import MegatronBertConfig, MegatronBertModel
        ...
        >>> # Initializing a MEGATRON_BERT bert-base-uncased style configuration
        >>> configuration = MegatronBertConfig()
        ...
        >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
        >>> model = MegatronBertModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "megatron-bert"

    def __init__(
        self,
        vocab_size=29056,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
    ):
        """
        Initialize a MegatronBertConfig object with the provided parameters.

        Args:
            vocab_size (int): The size of the vocabulary used for tokenization.
            hidden_size (int): The size of the hidden layers in the model.
            num_hidden_layers (int): The number of hidden layers in the model.
            num_attention_heads (int): The number of attention heads in the model.
            intermediate_size (int): The size of the intermediate (feed-forward) layer.
            hidden_act (str): The activation function used in the hidden layers.
            hidden_dropout_prob (float): The dropout probability for the hidden layers.
            attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
            max_position_embeddings (int): The maximum length of input sequences.
            type_vocab_size (int): The size of the token type embeddings.
            initializer_range (float): The range for parameter initializations.
            layer_norm_eps (float): The epsilon value for layer normalization.
            pad_token_id (int): The ID of the padding token.
            position_embedding_type (str): The type of position embeddings used.
            use_cache (bool): Whether to use caching during inference.

        Returns:
            None.

        Raises:
            ValueError: If any argument is invalid or out of range.
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

__all__ = ['MegatronBertConfig']
