# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Bert Model config
"""

from ...configuration_utils import PretrainedConfig


BERT_SUPPORT_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking"
]


class BertConfig(PretrainedConfig):
    """
    Configuration for BERT-base
    """
    model_type = "bert"

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
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        """
        Initialize a BertConfig object with the specified parameters.
        
        Args:
            self (object): The object instance.
            vocab_size (int): The size of the vocabulary. Defaults to 30522.
            hidden_size (int): The size of the hidden layers. Defaults to 768.
            num_hidden_layers (int): The number of hidden layers. Defaults to 12.
            num_attention_heads (int): The number of attention heads. Defaults to 12.
            intermediate_size (int): The size of the intermediate layer in the transformer encoder. Defaults to 3072.
            hidden_act (str): The activation function for the hidden layers. Defaults to 'gelu'.
            hidden_dropout_prob (float): The dropout probability for the hidden layers. Defaults to 0.1.
            attention_probs_dropout_prob (float): The dropout probability for the attention probabilities. Defaults to 0.1.
            max_position_embeddings (int): The maximum position index. Defaults to 512.
            type_vocab_size (int): The size of the type vocabulary. Defaults to 2.
            initializer_range (float): The range for weight initialization. Defaults to 0.02.
            layer_norm_eps (float): The epsilon value for layer normalization. Defaults to 1e-12.
            pad_token_id (int): The token ID for padding. Defaults to 0.
            position_embedding_type (str): The type of position embeddings. Defaults to 'absolute'.
            use_cache (bool): Whether to use cache during inference. Defaults to True.
            classifier_dropout (float): The dropout probability for the classifier layer. Defaults to None.
        
        Returns:
            None.
        
        Raises:
            ValueError: If any of the input parameters are invalid or out of range.
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
        self.classifier_dropout = classifier_dropout


__all__ = ["BertConfig"]
