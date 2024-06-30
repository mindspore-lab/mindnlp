# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
""" XLM-RoBERTa configuration"""

from ...configuration_utils import PretrainedConfig

XLM_ROBERTA_SUPPORT_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large"
]

class XLMRobertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XLMRobertaModel`] or a [`TFXLMRobertaModel`]. It
    is used to instantiate a XLM-RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the XLMRoBERTa
    [xlm-roberta-base](https://hf-mirror.com/xlm-roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the XLM-RoBERTa model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`XLMRobertaModel`] or [`TFXLMRobertaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
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
            The vocabulary size of the `token_type_ids` passed when calling [`XLMRobertaModel`] or
            [`TFXLMRobertaModel`].
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
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Example:
        ```python
        >>> from transformers import XLMRobertaConfig, XLMRobertaModel
        ...
        >>> # Initializing a XLM-RoBERTa xlm-roberta-base style configuration
        >>> configuration = XLMRobertaConfig()
        ...
        >>> # Initializing a model (with random weights) from the xlm-roberta-base style configuration
        >>> model = XLMRobertaModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "xlm-roberta"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        """
        The __init__ method initializes an instance of the XLMRobertaConfig class.

        Args:
            self: The instance of the class.
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The size of the hidden layers.
            num_hidden_layers (int): The number of hidden layers.
            num_attention_heads (int): The number of attention heads.
            intermediate_size (int): The size of the intermediate layer in the transformer encoder.
            hidden_act (str): The activation function for the hidden layers.
            hidden_dropout_prob (float): The dropout probability for the hidden layers.
            attention_probs_dropout_prob (float): The dropout probability for the attention probabilities.
            max_position_embeddings (int): The maximum position for positional embeddings.
            type_vocab_size (int): The size of the type vocabulary.
            initializer_range (float): The range for weight initialization.
            layer_norm_eps (float): The epsilon value for layer normalization.
            pad_token_id (int): The id for padding tokens.
            bos_token_id (int): The id for the beginning of sequence tokens.
            eos_token_id (int): The id for the end of sequence tokens.
            position_embedding_type (str): The type of position embedding to use.
            use_cache (bool): Whether to use cache for intermediate computations.
            classifier_dropout (float): The dropout probability for the classifier. Default is None.

        Returns:
            None.

        Raises:
            ValueError: If vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size,
                max_position_embeddings, type_vocab_size are not positive integers.
            ValueError: If hidden_dropout_prob, attention_probs_dropout_prob, initializer_range, layer_norm_eps,
                classifier_dropout are not in the range [0.0, 1.0].
            ValueError: If position_embedding_type is not 'absolute' or 'relative'.
            ValueError: If pad_token_id, bos_token_id, eos_token_id are not non-negative integers.
            TypeError: If classifier_dropout is not a float or None.
        """
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

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
        self.classifier_dropout = classifier_dropout

__all__ = ['XLMRobertaConfig']
