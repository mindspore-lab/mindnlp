# coding=utf-8
# Copyright 2024 The HuaWei Technologies Co., Microsoft Corporation.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
""" MPNet model configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)

MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/config.json",
}


class MPNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MPNetModel`] or a [`TFMPNetModel`]. It is used to
    instantiate a MPNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPNet
    [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) architecture.
    ```"""
    model_type = "mpnet"

    def __init__(
        self,
        vocab_size=30527,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        relative_attention_num_buckets=32,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        """Initializes a new instance of the MPNetConfig class.
        
        Args:
            vocab_size (int, optional): The size of the vocabulary. Defaults to 30527.
            hidden_size (int, optional): The size of the hidden states. Defaults to 768.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 12.
            intermediate_size (int, optional): The size of the intermediate layer in the feedforward network. Defaults to 3072.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'gelu'.
            hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): The dropout probability for the attention probabilities. Defaults to 0.1.
            max_position_embeddings (int, optional): The maximum number of positional embeddings. Defaults to 512.
            initializer_range (float, optional): The range for the random weight initialization. Defaults to 0.02.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-12.
            relative_attention_num_buckets (int, optional): The number of buckets for relative attention. Defaults to 32.
            pad_token_id (int, optional): The token ID for padding. Defaults to 1.
            bos_token_id (int, optional): The token ID for the beginning of sequence. Defaults to 0.
            eos_token_id (int, optional): The token ID for the end of sequence. Defaults to 2.
        
        Returns:
            None.
        
        Raises:
            None.
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
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.relative_attention_num_buckets = relative_attention_num_buckets


__all__ = ['MPNetConfig']
