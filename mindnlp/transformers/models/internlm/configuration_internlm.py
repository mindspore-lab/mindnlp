# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
""" InternLM model configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

INTERNLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


# Modified from transformers.model.llama.configuration_llama.LlamaConfig
class InternLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternLMModel`]. It is used to instantiate
    an InternLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the InternLM-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the InternLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`InternLMModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
    """
    model_type = "internlm"
    _auto_class = "AutoConfig"

    def __init__(  # pylint: disable=W0102
        self,
        vocab_size=103168,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        bias=True,
        rotary={"base": 10000, "type": "dynamic"},  # pylint: disable=W0102
        attn_implementation="eager",
        **kwargs,
    ):
        """
        This method initializes an instance of the InternLMConfig class with the provided configuration parameters.
        
        Args:
            vocab_size (int): The size of the vocabulary used in the language model.
            hidden_size (int): The size of the hidden layers in the model.
            intermediate_size (int): The size of the intermediate layers in the model.
            num_hidden_layers (int): The number of hidden layers in the model.
            num_attention_heads (int): The number of attention heads in the model.
            hidden_act (str): The activation function used in the hidden layers.
            max_position_embeddings (int): The maximum position index that can be used in the model.
            initializer_range (float): The range for weight initialization.
            rms_norm_eps (float): The epsilon value for RMS norm.
            use_cache (bool): Whether to use cache during model computation.
            pad_token_id (int): The token ID used for padding sequences.
            bos_token_id (int): The token ID used for the beginning of a sequence.
            eos_token_id (int): The token ID used for the end of a sequence.
            tie_word_embeddings (bool): Whether to tie the word embeddings.
            bias (bool): Whether to include bias in the model.
            rotary (dict): Dictionary with keys 'base' (int) and 'type' (str) defining rotary settings.
            attn_implementation (str): The implementation method for attention. If None, defaults to 'eager'.

        Returns:
            None.

        Raises:
            None
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.bias = bias
        self.rotary = rotary
        self.attn_implementation = attn_implementation
        if self.attn_implementation is None:
            self.attn_implementation = "eager"
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ['InternLMConfig']
