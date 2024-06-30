# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
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
"""
MobileBERT model configuration
"""

from ...configuration_utils import PretrainedConfig


MOBILEBERT_SUPPORT_LIST = [
    "mobilebert-uncased",
]

class MobileBertConfig(PretrainedConfig):
    """
    MobileBertConfig
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=24,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        embedding_size=128,
        trigram_input=True,
        use_bottleneck=True,
        intra_bottleneck_size=128,
        use_bottleneck_attention=False,
        key_query_shared_bottleneck=True,
        num_feedforward_networks=4,
        normalization_type="no_norm",
        classifier_activation=True,
        classifier_dropout=None,
        **kwargs,
    ):
        """
        Initializes a new instance of the MobileBertConfig class.
        
        Args:
            self (MobileBertConfig): The current instance of the MobileBertConfig class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 30522.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 512.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 24.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 4.
            intermediate_size (int, optional): The size of the intermediate layer. Defaults to 512.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'relu'.
            hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Defaults to 0.0.
            attention_probs_dropout_prob (float, optional): The dropout probability for the attention layer. Defaults to 0.1.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 512.
            type_vocab_size (int, optional): The size of the type vocabulary. Defaults to 2.
            initializer_range (float, optional): The range for random weight initialization. Defaults to 0.02.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-12.
            pad_token_id (int, optional): The token ID for padding. Defaults to 0.
            embedding_size (int, optional): The size of the embeddings. Defaults to 128.
            trigram_input (bool, optional): Whether to use trigram input. Defaults to True.
            use_bottleneck (bool, optional): Whether to use a bottleneck layer. Defaults to True.
            intra_bottleneck_size (int, optional): The size of the bottleneck layer. Defaults to 128.
            use_bottleneck_attention (bool, optional): Whether to use bottleneck attention. Defaults to False.
            key_query_shared_bottleneck (bool, optional): Whether to share the bottleneck between key and query. Defaults to True.
            num_feedforward_networks (int, optional): The number of feedforward networks. Defaults to 4.
            normalization_type (str, optional): The type of normalization to apply. Defaults to 'no_norm'.
            classifier_activation (bool, optional): Whether to apply activation to the classifier layer. Defaults to True.
            classifier_dropout (float, optional): The dropout probability for the classifier layer. Defaults to None.
        
        Returns:
            None.
        
        Raises:
            None.
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
        self.embedding_size = embedding_size
        self.trigram_input = trigram_input
        self.use_bottleneck = use_bottleneck
        self.intra_bottleneck_size = intra_bottleneck_size
        self.use_bottleneck_attention = use_bottleneck_attention
        self.key_query_shared_bottleneck = key_query_shared_bottleneck
        self.num_feedforward_networks = num_feedforward_networks
        self.normalization_type = normalization_type
        self.classifier_activation = classifier_activation

        if self.use_bottleneck:
            self.true_hidden_size = intra_bottleneck_size
        else:
            self.true_hidden_size = hidden_size

        self.classifier_dropout = classifier_dropout

__all__ = ['MobileBertConfig']
