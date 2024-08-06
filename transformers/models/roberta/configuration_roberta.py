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
""" RoBERTa configuration"""
from mindnlp.utils import logging

from ...configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

class RobertaConfig(PretrainedConfig):
    """Roberta Config."""
    model_type = "roberta"

    def __init__(
        self,
        vocab_size=50265,
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
        This method initializes an instance of the RobertaConfig class.
        
        Args:
            vocab_size (int): The size of the vocabulary. Default is 50265.
            hidden_size (int): The size of the hidden layers and the size of the embeddings. Default is 768.
            num_hidden_layers (int): The number of hidden layers in the model. Default is 12.
            num_attention_heads (int): The number of attention heads for each layer. Default is 12.
            intermediate_size (int): The size of the "intermediate" (i.e., feed-forward) layer in the transformer.
                Default is 3072.
            hidden_act (str): The non-linear activation function for the hidden layers. Default is 'gelu'.
            hidden_dropout_prob (float): The dropout probability for all fully connected layers in the embeddings
                and transformer layers. Default is 0.1.
            attention_probs_dropout_prob (float): The dropout probability for the attention probabilities.
                Default is 0.1.
            max_position_embeddings (int): The maximum sequence length that this model might ever be used with.
                Default is 512.
            type_vocab_size (int): The size of the "type" vocabulary. Default is 2.
            initializer_range (float): The standard deviation of the truncated_normal_initializer for initializing
                all weight matrices. Default is 0.02.
            layer_norm_eps (float): The epsilon used by LayerNorm layers. Default is 1e-12.
            pad_token_id (int): The id of the padding token. Default is 1.
            bos_token_id (int): The id of the beginning of the sequence token. Default is 0.
            eos_token_id (int): The id of the end of the sequence token. Default is 2.
            position_embedding_type (str): The type of position embedding. Default is 'absolute'.
            use_cache (bool): Whether or not to use caching for the model. Default is True.
            classifier_dropout (float): The dropout probability for the classifier. Default is None.
            **kwargs: Additional keyword arguments.
        
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
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

__all__ = ['RobertaConfig']
