# Copyright 2024 Huawei Technologies Co., Ltd
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
Bge M3 Config.
"""
from ...configuration_utils import PretrainedConfig

# Copied from transformers.models.xlm_roberta.configuration_xlm_roberta.XLMRobertaConfig with XLMRoberta->BgeM3
class BgeM3Config(PretrainedConfig):

    """
    A class representing the configuration for a BgeM3 model. 
    
    This class inherits from the PretrainedConfig class and defines the configuration parameters for a BgeM3 model,
    including vocabulary size, hidden size, number of hidden layers, number of attention heads, intermediate size,
    activation function, dropout probabilities, maximum position embeddings, type vocabulary size, initializer range,
    layer normalization epsilon, padding token ID, beginning of sequence token ID, end of sequence token ID,
    position embedding type, cache usage, classifier dropout, Colbert dimension, sentence pooling method, and unused tokens.

    Parameters:
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The size of the hidden layers.
        num_hidden_layers (int): The number of hidden layers in the model.
        num_attention_heads (int): The number of attention heads in the model.
        intermediate_size (int): The size of the intermediate layer in the model.
        hidden_act (str): The activation function used in the hidden layers.
        hidden_dropout_prob (float): The dropout probability for the hidden layers.
        attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
        max_position_embeddings (int): The maximum position embeddings in the model.
        type_vocab_size (int): The size of the type vocabulary.
        initializer_range (float): The range for parameter initialization.
        layer_norm_eps (float): The epsilon value for layer normalization.
        pad_token_id (int): The ID for padding tokens.
        bos_token_id (int): The ID for the beginning of sequence tokens.
        eos_token_id (int): The ID for the end of sequence tokens.
        position_embedding_type (str): The type of position embedding used.
        use_cache (bool): Flag indicating whether caching is used.
        classifier_dropout (float): The dropout rate for the classifier layer.
        colbert_dim (int): The dimension of Colbert.
        sentence_pooling_method (str): The method used for sentence pooling.
        unused_tokens (list): A list of unused tokens.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The size of the hidden layers.
        num_hidden_layers (int): The number of hidden layers in the model.
        num_attention_heads (int): The number of attention heads in the model.
        hidden_act (str): The activation function used in the hidden layers.
        intermediate_size (int): The size of the intermediate layer in the model.
        hidden_dropout_prob (float): The dropout probability for the hidden layers.
        attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
        max_position_embeddings (int): The maximum position embeddings in the model.
        type_vocab_size (int): The size of the type vocabulary.
        initializer_range (float): The range for parameter initialization.
        layer_norm_eps (float): The epsilon value for layer normalization.
        position_embedding_type (str): The type of position embedding used.
        use_cache (bool): Flag indicating whether caching is used.
        classifier_dropout (float): The dropout rate for the classifier layer.
        colbert_dim (int): The dimension of Colbert.
        sentence_pooling_method (str): The method used for sentence pooling.
        unused_tokens (list): A list of unused tokens.
    """
    model_type = "bge-m3"

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
        colbert_dim=None,
        sentence_pooling_method='cls',
        unused_tokens=None,
        **kwargs,
    ):
        """
        This method initializes an instance of the BgeM3Config class with the given parameters.
        
        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Default is 30522.
            hidden_size (int, optional): The size of the hidden layers. Default is 768.
            num_hidden_layers (int, optional): The number of hidden layers. Default is 12.
            num_attention_heads (int, optional): The number of attention heads. Default is 12.
            intermediate_size (int, optional): The size of the intermediate layer in the transformer encoder. Default is 3072.
            hidden_act (str, optional): The activation function for the hidden layers. Default is 'gelu'.
            hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Default is 0.1.
            attention_probs_dropout_prob (float, optional): The dropout probability for the attention probabilities. Default is 0.1.
            max_position_embeddings (int, optional): The maximum number of positions for positional embeddings. Default is 512.
            type_vocab_size (int, optional): The size of the type vocabulary. Default is 2.
            initializer_range (float, optional): The range for parameter initializers. Default is 0.02.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Default is 1e-12.
            pad_token_id (int, optional): The token id for padding. Default is 1.
            bos_token_id (int, optional): The token id for the beginning of sequence. Default is 0.
            eos_token_id (int, optional): The token id for the end of sequence. Default is 2.
            position_embedding_type (str, optional): The type of position embedding to use. Default is 'absolute'.
            use_cache (bool, optional): Whether to use caching during decoding. Default is True.
            classifier_dropout (float, optional): The dropout probability for the classifier layer. Default is None.
            colbert_dim (int, optional): The dimensionality of the colbert layer. Default is None.
            sentence_pooling_method (str, optional): The method for pooling sentence representations. Default is 'cls'.
            unused_tokens (list, optional): A list of unused tokens. Default is None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None.
        
        Raises:
            ValueError: If any of the parameters are invalid or out of range.
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
        self.colbert_dim = colbert_dim
        self.sentence_pooling_method = sentence_pooling_method
        self.unused_tokens = unused_tokens

__all__ = ['BgeM3Config']
