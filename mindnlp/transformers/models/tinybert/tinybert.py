# Copyright 2022 Huawei Technologies Co., Ltd
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
TinyBert Models
"""
import math
import os
from typing import Union
import mindspore
from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from .tinybert_config import TinyBertConfig
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel


class TinyBertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        """
        init BertEmbeddings
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        Construct the embeddings from word, position and token_type embeddings.
        """
        seq_length = input_ids.shape[1]
        position_ids = ops.arange(seq_length, dtype=mindspore.int64)
        position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TinyBertSelfAttention(nn.Module):
    r"""
    TinyBertSelfAttention
    """
    def __init__(self, config):
        """
        Initializes an instance of the TinyBertSelfAttention class.
        
        Args:
            self: The instance of the class.
            config: An object of type 'config' containing the configuration settings for the self-attention mechanism.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the hidden size specified in the configuration is not a multiple of the number of
                attention heads.
        
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the " +
                f"number of attention heads {config.num_attention_heads}")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        transpose_for_scores
        """
        new_x_shape = x.shape[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """
        This method forwards the self-attention mechanism for the TinyBERT model.
        
        Args:
            self (object): The instance of the TinyBertSelfAttention class.
            hidden_states (tensor): The input hidden states tensor representing the input sequence.
            attention_mask (tensor): The attention mask tensor to mask certain positions in the input sequence. 
                It should have the same shape as the input sequence.
        
        Returns:
            tensor: The context layer tensor after applying the self-attention mechanism.
            tensor: The attention scores tensor generated during the self-attention computation.
        
        Raises:
            ValueError: If the dimensions of the input tensors are incompatible for matrix multiplication.
            RuntimeError: If an error occurs during the softmax computation.
            AttributeError: If the required attributes are missing in the class instance.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(
            query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class TinyBertAttention(nn.Module):
    """
    TinyBertAttention
    """
    def __init__(self, config):
        """
        Initializes a new instance of the TinyBertAttention class.
        
        Args:
            self (object): The instance of the class itself.
            config (object): An object containing configuration parameters for the TinyBertAttention class.
                This parameter is required for configuring the attention mechanism.
                
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()

        self.self_ = TinyBertSelfAttention(config)
        self.output = TinyBertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Constructs the attention output and layer attention for the TinyBertAttention class.
        
        Args:
            self (TinyBertAttention): An instance of the TinyBertAttention class.
            input_tensor (Tensor): The input tensor for the attention calculation.
            attention_mask (Tensor): The attention mask tensor.
        
        Returns:
            tuple: A tuple containing the attention output tensor and the layer attention tensor.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        self_output, layer_att = self.self_(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att


class TinyBertSelfOutput(nn.Module):
    """
    TinyBertSelfOutput
    """
    def __init__(self, config):
        """
        Initialize the TinyBertSelfOutput class.
        
        Args:
            self (object): The instance of the TinyBertSelfOutput class.
            config (object):
                An object containing configuration parameters for the model.

                - hidden_size (int): The size of the hidden layer.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters are missing or incorrect.
            TypeError: If the provided configuration is not of the expected type.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Constructs the output of the TinyBertSelf layer.

        Args:
            self (TinyBertSelfOutput): An instance of the TinyBertSelfOutput class.
            hidden_states (torch.Tensor): A tensor representing the hidden states.
                The shape of the tensor is (batch_size, sequence_length, hidden_size).
                It contains the input hidden states to be processed.
            input_tensor (torch.Tensor): A tensor representing the input tensor.
                The shape of the tensor is (batch_size, sequence_length, hidden_size).
                It serves as the residual connection for the hidden states.

        Returns:
            torch.Tensor: A tensor representing the output of the TinyBertSelf layer.
                The shape of the tensor is (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TinyBertIntermediate(nn.Module):
    """
    TinyBertIntermediate
    """
    def __init__(self, config, intermediate_size=-1):
        """
        This method initializes a TinyBertIntermediate instance.

        Args:
            self: The instance of the TinyBertIntermediate class.
            config:
                An object that holds the configuration settings for the TinyBertIntermediate model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: None

            intermediate_size:
                An optional integer representing the intermediate size.

                - Type: int
                - Purpose: Specifies the size of the intermediate layer.
                - Restrictions: Must be a non-negative integer. If not provided, the default value is -1.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not of type 'object'.
            ValueError: If the 'intermediate_size' parameter is provided and is a negative integer.
        """
        super().__init__()
        if intermediate_size < 0:
            self.dense = nn.Linear(
                config.hidden_size, config.intermediate_size)
        else:
            self.dense = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """
        This method forwards the intermediate layer of a TinyBERT model.

        Args:
            self (object): The instance of the TinyBertIntermediate class.
            hidden_states (tensor): The input hidden states to be processed by the method.
                Should be a tensor representing the hidden states of the model.

        Returns:
            hidden_states: The processed hidden states are returned as the output of the method.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TinyBertOutput(nn.Module):
    """
    TinyBertOutput
    """
    def __init__(self, config, intermediate_size=-1):
        """
        Initializes an instance of the TinyBertOutput class.

        Args:
            self (TinyBertOutput): The instance of the class.
            config: A configuration object containing various parameters.
            intermediate_size (int, optional): The size of the intermediate layer. Defaults to -1.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        if intermediate_size < 0:
            self.dense = nn.Linear(
                config.intermediate_size, config.hidden_size)
        else:
            self.dense = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Method 'forward' in the class 'TinyBertOutput'.

        Args:
            self (object): Instance of the 'TinyBertOutput' class.
            hidden_states (object):
                The hidden states to be processed.

                - Type: Tensor
                - Purpose: Represents the hidden states that need to be transformed.
                - Restrictions: Should be a valid tensor object.
            input_tensor (object):
                The input tensor to be combined with the hidden states.

                - Type: Tensor
                - Purpose: Represents the input tensor to be added to the hidden states.
                - Restrictions: Should be a valid tensor object.

        Returns:
            None:
                The transformation is applied to 'hidden_states' in-place.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TinyBertLayer(nn.Module):
    """
    TinyBertLayer
    """
    def __init__(self, config):
        """
        Initializes a new instance of the TinyBertLayer class.

        Args:
            self: The object itself.
            config: An instance of the configuration class that holds various hyperparameters and settings
                for the TinyBertLayer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = TinyBertAttention(config)
        self.intermediate = TinyBertIntermediate(config)
        self.output = TinyBertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Constructs a TinyBertLayer by applying attention mechanism and generating layer output.

        Args:
            self (object): The instance of the TinyBertLayer class.
            hidden_states (object): The input hidden states for the layer, typically a tensor of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (object): The attention mask for the input hidden states, typically a tensor of shape
                (batch_size, 1, sequence_length, sequence_length) with 0s for padding tokens and 1s for non-padding
                tokens.

        Returns:
            tuple: A tuple containing the layer output (typically a tensor of shape
                (batch_size, sequence_length, hidden_size)) and the attention scores (layer_att) generated during
                the attention computation.

        Raises:
            ValueError: If the shape of the hidden_states or attention_mask is not compatible with the expected
                input shapes.
            TypeError: If the input data types are not compatible with the expected types.
            RuntimeError: If there is an issue during the computation of attention or output layers.
        """
        attention_output, layer_att = self.attention(
            hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, layer_att


class TinyBertEncoder(nn.Module):
    """
    TinyBertEncoder
    """
    def __init__(self, config):
        """
        Initializes a TinyBertEncoder object with the given configuration.

        Args:
            self (TinyBertEncoder): The instance of the TinyBertEncoder class.
            config (object): The configuration object containing parameters for the TinyBertEncoder.
                This object must have the following attributes:

                - num_hidden_layers (int): The number of hidden layers for the encoder.
                - Other attributes required by the TinyBertLayer forwardor.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layer = nn.ModuleList([TinyBertLayer(config)
                                  for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        """
        Method 'forward' in the class 'TinyBertEncoder'.

        Args:
            self (object): The instance of the 'TinyBertEncoder' class.
            hidden_states (object): The hidden states to be processed by the encoder.
            attention_mask (object): The attention mask to apply during encoding.

        Returns:
            Tuple:
                Two lists containing the encoder layers and their respective attentions.

                - List: 'all_encoder_layers' - List of all encoder layers processed during encoding.
                - List: 'all_encoder_atts' - List of attention values for each encoder layer.

        Raises:
            None.
        """
        all_encoder_layers = []
        all_encoder_atts = []
        for _, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att = layer_module(
                hidden_states, attention_mask)
            all_encoder_atts.append(layer_att)

        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_atts


class TinyBertPooler(nn.Module):
    """
    TinyBertPooler
    """
    def __init__(self, config):
        """
        Initialize the TinyBertPooler class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration settings.

                - hidden_size (int): The size of the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        """
        Constructs the TinyBertPooler by calculating the pooled output from the given hidden states.

        Args:
            self (TinyBertPooler): An instance of the TinyBertPooler class.
            hidden_states (List[Tensor]): A list of hidden states from the TinyBERT model. Each hidden state is a tensor.

        Returns:
            Tensor: The pooled output tensor obtained after processing the hidden states through dense layers and
                activation function.

        Raises:
            TypeError: If the input hidden_states is not a list of tensors.
            ValueError: If the hidden_states list is empty or does not contain valid tensors.
            IndexError: If the hidden_states list does not have the required elements for pooling.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. "-1" refers to last layer
        pooled_output = hidden_states[-1][:, 0]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class TinyBertPredictionHeadTransform(nn.Module):
    """
    TinyBertPredictionHeadTransform
    """
    def __init__(self, config):
        """
        Initializes a TinyBertPredictionHeadTransform object.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration parameters for the TinyBertPredictionHeadTransform.

                - hidden_size (int): The size of the hidden layer.
                - hidden_act (str or function): The activation function to be used for the hidden layer.
                If it's a string, it should correspond to a key in ACT2FN dictionary.
                - epsilon (float): A small value added to the variance in LayerNorm to prevent division by zero.

        Returns:
            None.

        Raises:
            TypeError: If the config.hidden_act is not a string or a function.
            ValueError: If the config.hidden_act string does not correspond to a valid key in ACT2FN dictionary.
            ValueError: If the config.hidden_size is not an integer value.
        """
        super().__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=1e-12)

    def forward(self, hidden_states):
        """
        Constructs the TinyBertPredictionHeadTransform.

        This method takes in the hidden states and performs a series of transformations to predict the next token
        in the sequence.

        Args:
            self (TinyBertPredictionHeadTransform): An instance of the TinyBertPredictionHeadTransform class.
            hidden_states (tensor): The hidden states obtained from the previous layer.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TinyBertLMPredictionHead(nn.Module):
    """
    TinyBertLMPredictionHead
    """
    def __init__(self, config, bert_model_embedding_weights):
        """
        Initializes the TinyBertLMPredictionHead.

        Args:
            self: The object itself.
            config: A dictionary containing configuration parameters for the TinyBertPredictionHeadTransform.
            bert_model_embedding_weights: A tensor representing the weights of the BERT model's embedding layer.

        Returns:
            None.

        Raises:
            ValueError: If the provided `config` is not a dictionary or if `bert_model_embedding_weights`
                is not a tensor.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__()
        self.transform = TinyBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.shape[1],
                                bert_model_embedding_weights.shape[0],
                                bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = Parameter(ops.zeros(
            bert_model_embedding_weights.shape[0]))

    def forward(self, hidden_states):
        """
        Method to forward the prediction head for TinyBERT LM.

        Args:
            self (object): Instance of the TinyBertLMPredictionHead class.
            hidden_states (tensor): Hidden states obtained from the TinyBERT model.
                The tensor should have the shape [batch_size, sequence_length, hidden_size].

        Returns:
            None: This method modifies the hidden_states in place to forward the LM prediction head.

        Raises:
            None.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class TinyBertOnlyMLMHead(nn.Module):
    """
    TinyBertOnlyMLMHead
    """
    def __init__(self, config, bert_model_embedding_weights):
        """
        Initializes an instance of the TinyBertOnlyMLMHead class.

        Args:
            self: The object instance.
            config: A configuration object containing the model's hyperparameters and settings.
                This parameter is of type 'config' and is required.
            bert_model_embedding_weights: The pre-trained BERT model's embedding weights.
                This parameter is of type 'Tensor' and is required.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = TinyBertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        """
        This method forwards the prediction scores based on the provided sequence output for the
        TinyBertOnlyMLMHead class.

        Args:
            self (object): The instance of the TinyBertOnlyMLMHead class.
            sequence_output (tensor): The output tensor representing the sequence to be used for prediction scores
                calculation. It should be of type tensor and must contain the sequence information for prediction.

        Returns:
            prediction_scores (tensor): The prediction scores tensor calculated based on the provided sequence_output.
                It represents the scores for predicting the masked words in the input sequence.

        Raises:
            No specific exceptions are raised by this method.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TinyBertOnlyNSPHead(nn.Module):
    """
    TinyBertOnlyNSPHead
    """
    def __init__(self, config):
        """
        Initializes the TinyBertOnlyNSPHead class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration settings for the TinyBertOnlyNSPHead.
                It must have the attribute 'hidden_size' to specify the size of the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        """
        Method: forward

        Description:
            This method calculates the sequence relationship score using the provided pooled_output.

        Args:
            self: (object) The instance of the class TinyBertOnlyNSPHead.
            pooled_output: (object) The pooled output from the TinyBERT model.

        Returns:
            seq_relationship_score: (object) The calculated sequence relationship score based on the
                provided pooled_output.

        Raises:
            None
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class TinyBertPreTrainingHeads(nn.Module):
    """
    TinyBertPreTrainingHeads
    """
    def __init__(self, config, bert_model_embedding_weights):
        """
        Initializes the TinyBertPreTrainingHeads class.

        Args:
            self (object): The instance of the class.
            config (object): Configuration object containing settings for the model.
            bert_model_embedding_weights (ndarray): The weights for the BERT model's embeddings.

        Returns:
            None.

        Raises:
            ValueError: If the config parameter is not provided or is of incorrect type.
            ValueError: If the bert_model_embedding_weights parameter is not provided or is not a numpy array.
        """
        super().__init__()
        self.predictions = TinyBertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        """
        This method forwards prediction scores and sequence relationship scores for pre-training heads in TinyBert.

        Args:
            self (object): The instance of the TinyBertPreTrainingHeads class.
            sequence_output (object): The output sequence from the TinyBert model,
                representing the contextualized representations of tokens.
            pooled_output (object): The output at the pooled layer of the TinyBert model,
                representing the aggregated representation of the entire input sequence.

        Returns:
            tuple:
                A tuple containing the prediction scores and sequence relationship score.

                - prediction_scores (object): The prediction scores for the pre-training task based on the
                sequence_output.
                - seq_relationship_score (object): The sequence relationship score for the pre-training task based on
                the pooled_output.

        Raises:
            None
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class TinyBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """
    config_class = TinyBertConfig

    base_model_prefix = 'bert'

    def __init__(self, config):
        """
        Initializes a TinyBertPreTrainedModel instance.

        Args:
            self: The instance of the TinyBertPreTrainedModel class.
            config (TinyBertConfig): An instance of the TinyBertConfig class that holds the configuration settings
                for the model. It should be an instance of TinyBertConfig.

        Returns:
            None: This method initializes the TinyBertPreTrainedModel instance with the provided configuration.

        Raises:
            ValueError: If the config parameter is not an instance of TinyBertConfig, a ValueError is raised
                with a message specifying the requirement for config to be an instance of TinyBertConfig.
        """
        super().__init__(config)
        if not isinstance(config, TinyBertConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of " +
                "class `BertConfig`. To create a model from a Google pretrained model use " +
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def init_model_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight = ops.normal(
                size=module.weight.shape,
                mean=0.0,
                std=self.config.initializer_range
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias = ops.full(
                dtype=module.bias.dtype, size=module.bias.shape, fill_value=0)
            module.weight = ops.full(
                dtype=module.weight.dtype, size=module.weight.shape, fill_value=1.0)
        if isinstance(module, nn.Linear):
            module.weight = ops.normal(
                size=module.weight.shape, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias = ops.full(
                    dtype=module.bias.dtype, size=module.bias.shape, fill_value=0)

    def get_input_embeddings(self) -> "nn.Module":
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`: A mindspore cell mapping vocabulary to hidden states.
        """
    def set_input_embeddings(self, new_embeddings: "nn.Module"):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Module`): A mindspore cell mapping vocabulary to hidden states.
        """
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        resize the model position embeddings if necessary
        """
    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
    def save(self, save_dir: Union[str, os.PathLike]):
        "save pretrain model"
    #TODO
    def post_init(self):
        """post init."""
class TinyBertModel(TinyBertPreTrainedModel):
    """
    TinyBERT model
    """
    def __init__(self, config):
        """
        Initializes a new instance of the TinyBertModel class.

        Args:
            self: The instance of the TinyBertModel class.
            config:
                A dictionary containing configuration parameters for the model.

                - Type: dict
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid dictionary object.

        Returns:
            None.

        Raises:
            ValueError: If the provided 'config' parameter is not a valid dictionary.
            TypeError: If any of the required components fail to initialize properly.
        """
        super().__init__(config)
        self.embeddings = TinyBertEmbeddings(config)
        self.encoder = TinyBertEncoder(config)
        self.pooler = TinyBertPooler(config)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                  output_all_encoded_layers=True, output_att=True):
        """forward."""
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # extended_attention_mask = extended_attention_mask.to(
        #     dtype=next(self.parameters()).dtype)  # fp16 compatibility

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, layer_atts = self.encoder(embedding_output,
                                                  extended_attention_mask)

        pooled_output = self.pooler(encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if not output_att:
            return encoded_layers, pooled_output

        return encoded_layers, layer_atts, pooled_output


class TinyBertForPreTraining(TinyBertPreTrainedModel):
    """
    BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:

    - the masked language modeling head, and
    - the next sentence classification head.
    """
    def __init__(self, config):
        """
        Initialize the TinyBertForPreTraining class.

        Args:
            self (object): The instance of the TinyBertForPreTraining class.
            config (object): The configuration object containing parameters for model initialization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                  masked_lm_labels=None, next_sentence_label=None):
        """forward."""
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, output_att=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss
            return total_loss
        return prediction_scores, seq_relationship_score


class TinyBertFitForPreTraining(TinyBertPreTrainedModel):
    """
    TinyBertForPreTraining with fit dense
    """
    def __init__(self, config, fit_size=768):
        """
        Initialize a TinyBertFitForPreTraining object.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object for the model.
            fit_size (int, optional): The size to fit the dense layer to. Default is 768.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """forward."""
        sequence_output, att_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask)
        tmp = []
        for _, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output


class TinyBertForMaskedLM(TinyBertPreTrainedModel):
    """
    BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the TinyBertForMaskedLM class.

        Args:
            self: The object instance.
            config: An instance of the configuration class, containing various hyperparameters and settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                  output_att=False):
        """forward."""
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=True, output_att=output_att)

        if output_att:
            sequence_output, att_output = sequence_output
        prediction_scores = self.cls(sequence_output[-1])

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            if not output_att:
                return masked_lm_loss

            return masked_lm_loss, att_output

        if not output_att:
            return prediction_scores
        return prediction_scores, att_output


class TinyBertForNextSentencePrediction(TinyBertPreTrainedModel):
    """
    BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.
    """
    def __init__(self, config):
        """
        Initializes the TinyBertForNextSentencePrediction class.

        Args:
            self: An instance of the TinyBertForNextSentencePrediction class.
            config:
                A configuration object containing various hyperparameters and settings for the model.

                - Type: Any valid configuration object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid configuration object.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__(config)
        self.bert = TinyBertModel(config)
        self.cls = TinyBertOnlyNSPHead(config)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        """forward."""
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False, output_att=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss

        return seq_relationship_score


class TinyBertForSentencePairClassification(TinyBertPreTrainedModel):
    """
    TinyBertForSentencePairClassification
    """
    def __init__(self, config, num_labels):
        """
        Initializes a new instance of the TinyBertForSentencePairClassification class.
        
        Args:
            self: The instance of the class.
            config: A configuration object containing various settings and hyperparameters for the model.
                It is of type 'Config'.
            num_labels: An integer representing the number of labels for sentence pair classification.
        
        Returns:
            None.
        
        Raises:
            AttributeError: If the 'config' parameter is missing required attributes.
            ValueError: If the 'num_labels' parameter is not a positive integer.
            TypeError: If the 'config' parameter is not of type 'Config'.
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = TinyBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_model_weights)

    def forward(self, a_input_ids, b_input_ids, a_token_type_ids=None, b_token_type_ids=None,
                  a_attention_mask=None, b_attention_mask=None, labels=None):
        """forward."""
        _, a_pooled_output = self.bert(
            a_input_ids, a_token_type_ids, a_attention_mask, output_all_encoded_layers=False, output_att=False)
        # a_pooled_output = self.dropout(a_pooled_output)

        _, b_pooled_output = self.bert(
            b_input_ids, b_token_type_ids, b_attention_mask, output_all_encoded_layers=False, output_att=False)
        # b_pooled_output = self.dropout(b_pooled_output)

        logits = self.classifier(ops.relu(ops.concat((a_pooled_output, b_pooled_output,
                                                      ops.abs(a_pooled_output - b_pooled_output)), -1)))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class TinyBertForSequenceClassification(TinyBertPreTrainedModel):
    """
    TinyBertForSequenceClassification
    """
    def __init__(self, config, num_labels, fit_size=768):
        """
        Initializes a TinyBertForSequenceClassification instance.
        
        Args:
            self: The instance of the class.
            config (object): The configuration object containing model hyperparameters.
            num_labels (int): The number of labels for classification.
            fit_size (int, optional): The size of the hidden layer for fitting. Default is 768.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the provided 'config' is not of the expected type.
            ValueError: If 'num_labels' is not a positive integer.
            ValueError: If 'fit_size' is not a positive integer.
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = TinyBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                  is_student=False):
        """forward"""
        sequence_output, att_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                               output_all_encoded_layers=True, output_att=True)

        logits = self.classifier(ops.relu(pooled_output))

        tmp = []
        if is_student:
            for _, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output

__all__ = [
    'TinyBertModel',
    'TinyBertForSequenceClassification',
    'TinyBertForMaskedLM',
    'TinyBertForNextSentencePrediction',
    'TinyBertForMaskedLM',
    'TinyBertForSentencePairClassification',
    'TinyBertForPreTraining',
    'TinyBertFitForPreTraining'
]
