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

"""roberta model, base on bert."""
import mindspore
from mindspore.common.initializer import initializer

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import functional as F
from .configuration_roberta import RobertaConfig
from ..bert.modeling_bert import BertModel, BertPreTrainedModel


class MSRobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        """
        This method initializes the MSRobertaEmbeddings class.
        
        Args:
            self (object): The instance of the MSRobertaEmbeddings class.
            config (object): An object containing configuration parameters for the embeddings.
                It should include the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - max_position_embeddings (int): The maximum allowed position for embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.
                - position_embedding_type (str, optional): The type of position embedding. Defaults to 'absolute'.
                - pad_token_id (int): The token id for padding.

        Returns:
            None.

        Raises:
            AttributeError: If the 'config' object does not contain the required attributes.
            ValueError: If the provided configuration parameters are invalid or inconsistent.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).view((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        """
        Construct the embeddings for the MSRoberta model.

        Args:
            self (MSRobertaEmbeddings): The instance of the MSRobertaEmbeddings class.
            input_ids (Tensor, optional): The input tensor containing token ids. Default is None.
            token_type_ids (Tensor, optional): The input tensor containing token type ids. Default is None.
            position_ids (Tensor, optional): The input tensor containing position ids. Default is None.
            inputs_embeds (Tensor, optional): The input tensor containing embeddings. Default is None.
            past_key_values_length (int, optional): The length of past key values. Default is 0.

        Returns:
            Tensor: The forwarded embeddings for the model.

        Raises:
            ValueError: If input_ids and inputs_embeds are both None.
            ValueError: If position_ids is None and input_ids is also None.
            ValueError: If token_type_ids is None and self does not have 'token_type_ids'.
            ValueError: If input_shape does not contain the sequence length information.
            ValueError: If inputs_embeds is None and self does not have 'word_embeddings'.
            ValueError: If self.position_embedding_type is not 'absolute'.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in forwardor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ops.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64
        )
        return position_ids.unsqueeze(0).broadcast_to(input_shape)


class MSRobertaPreTrainedModel(BertPreTrainedModel):
    """Roberta Pretrained Model."""
    config_class = RobertaConfig
    base_model_prefix = "roberta"

class MSRobertaModel(BertModel):
    """Roberta Model"""
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new instance of the MSRobertaModel class.

        Args:
            self: The object itself.
            config (RobertaConfig): An instance of the RobertaConfig class containing the model configuration settings.
            add_pooling_layer (bool, optional): Specifies whether to add a pooling layer on top of the model output.
                Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = MSRobertaEmbeddings(config)

class MSRobertaLMHead(nn.Module):
    """RobertaLMHead"""
    def __init__(self, config):
        """
        Initializes the MSRobertaLMHead class with the given configuration.

        Args:
            self (MSRobertaLMHead): The instance of the MSRobertaLMHead class.
            config (Config): An object containing the configuration parameters for the model.
                It includes the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - vocab_size (int): The size of the vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            ValueError: If the config parameter is not of type Config.
            TypeError: If any of the attributes in the config object are missing or have incorrect types.
            RuntimeError: If there is an issue with initializing any of the neural network layers or parameters.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = Parameter(initializer('zeros', config.vocab_size), 'bias')
        self.gelu = nn.GELU()

    def forward(self, features):
        """
        Constructs the output of the MSRobertaLMHead model.

        Args:
            self (MSRobertaLMHead): An instance of the MSRobertaLMHead class.
            features: The input features to be processed. This should be a tensor of shape (batch_size, feature_size).

        Returns:
            None.

        Raises:
            None.
        """
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x) + self.bias
        return x

class MSRobertaClassificationHead(nn.Module):
    """RobertaClassificationHead"""
    def __init__(self, config):
        """
        Initializes an instance of the MSRobertaClassificationHead class.

        Args:
            self: The instance of the class.
            config: An object of the configuration class containing the necessary parameters for initialization.
                It must have the following attributes:

                - hidden_size (int): The size of the hidden state.
                - hidden_dropout_prob (float): The dropout probability for the hidden state.
                - num_labels (int): The number of output labels.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        """
        This method forwards the classification head for the MSRoberta model.

        Args:
            self (MSRobertaClassificationHead): The instance of the MSRobertaClassificationHead class.
            features (Tensor): The input features for classification, expected to be a 3D tensor of shape
                (batch_size, sequence_length, feature_dim).

        Returns:
            Tensor: The output tensor after applying the classification head operations.

        Raises:
            ValueError: If the input features tensor is not in the expected format or shape.
            RuntimeError: If an error occurs during the forwardion of the classification head.
        """
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = ops.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MSRobertaForMaskedLM(MSRobertaPreTrainedModel):
    """RobertaForMaskedLM"""
    def __init__(self, config, *args, **kwargs):
        """
        Initializes an instance of the MSRobertaForMaskedLM class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration parameters for the model.
                It should be an instance of the MSRobertaConfig class or a subclass of it.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config, *args, **kwargs)
        self.roberta = MSRobertaModel(config)
        self.lm_head = MSRobertaLMHead(config)
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight
        self.vocab_size = self.config.vocab_size

    def get_output_embeddings(self):
        """
        This method returns the output embeddings of the model's decoder layer.

        Args:
            self: An instance of the MSRobertaForMaskedLM class.
                This parameter is used to access the decoder layer of the model.

        Returns:
            None:
                The method does not return any specific value but provides access to the output embeddings of
                the decoder layer.

        Raises:
            None:
                This method does not raise any exceptions.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the MSRobertaForMaskedLM model.

        Args:
            self (MSRobertaForMaskedLM): The instance of the MSRobertaForMaskedLM class.
                This parameter represents the current instance of the model.
            new_embeddings (object): The new embeddings to be set as the output embeddings.
                This should be an object representing the new embeddings to replace the current ones.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head.decoder = new_embeddings

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  masked_lm_labels=None):
        """
        Constructs the masked language model (MLM) outputs for the MSRobertaForMaskedLM model.

        Args:
            self (MSRobertaForMaskedLM): The instance of the MSRobertaForMaskedLM class.
            input_ids (Tensor): The input tensor representing the tokenized input sequence.
            attention_mask (Tensor, optional): An optional tensor representing the attention mask.
                It specifies which tokens should be attended to and which should be ignored. Defaults to None.
            token_type_ids (Tensor, optional): An optional tensor representing the type of each token.
                Defaults to None.
            position_ids (Tensor, optional): An optional tensor representing the position of each token.
                Defaults to None.
            head_mask (Tensor, optional): An optional tensor representing the mask for the attention heads.
                Defaults to None.
            masked_lm_labels (Tensor, optional): An optional tensor representing the masked language model labels.
                Defaults to None.

        Returns:
            Tuple: A tuple containing the MLM prediction scores, and other outputs from the model.

        Raises:
            TypeError: If the input_ids tensor is not provided.
            ValueError: If the input_ids tensor is empty.
            ValueError: If the masked_lm_labels tensor is not provided.
            ValueError: If the masked_lm_labels tensor is empty.
        """
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, self.vocab_size),
                                               masked_lm_labels.view(-1), ignore_index=-1)
            outputs = (masked_lm_loss,) + outputs

        return outputs

class MSRobertaForSequenceClassification(MSRobertaPreTrainedModel):
    """MSRobertaForSequenceClassification"""
    def __init__(self, config, *args, **kwargs):
        """
        Initializes an instance of the MSRobertaForSequenceClassification class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing settings for the model.
                This parameter is required for initializing the model and must be of type 'object'.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config, *args, **kwargs)
        self.num_labels = config.num_labels
        self.roberta = MSRobertaModel(config, add_pooling_layer=False)
        self.classifier = MSRobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  labels=None):
        """
        Constructs the model architecture for sequence classification using the MSRoberta model.

        Args:
            self (MSRobertaForSequenceClassification): An instance of the MSRobertaForSequenceClassification class.
            input_ids (torch.Tensor): The input sequence token IDs. Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor, optional): The attention mask that identifies padding tokens.
                Shape: (batch_size, sequence_length). Defaults to None.
            token_type_ids (torch.Tensor, optional): The token type IDs. Shape: (batch_size, sequence_length).
                Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs. Shape: (batch_size, sequence_length).
                Defaults to None.
            head_mask (torch.Tensor, optional): The head mask. Shape: (num_heads,) or (num_layers, num_heads).
                Defaults to None.
            labels (torch.Tensor, optional): The target labels for sequence classification.
                Shape: (batch_size,) or (batch_size, num_labels). Defaults to None.

        Returns:
            tuple:
                A tuple containing the output logits and additional outputs.

                - logits (torch.Tensor): The output logits for sequence classification.
                Shape: (batch_size, num_labels) or (batch_size,) if num_labels equals 1.
                - sequence_output (torch.Tensor): The output tensor from the MSRoberta model.
                Shape: (batch_size, sequence_length, hidden_size).
                - additional outputs (tuple): Any additional outputs from the MSRoberta model.

        Raises:
            ValueError: If the input_ids tensor shape is not (batch_size, sequence_length).
            ValueError: If the attention_mask tensor shape is not (batch_size, sequence_length).
            ValueError: If the token_type_ids tensor shape is not (batch_size, sequence_length).
            ValueError: If the position_ids tensor shape is not (batch_size, sequence_length).
            ValueError: If the head_mask tensor shape is not (num_heads,) or (num_layers, num_heads).
            ValueError: If the labels tensor shape is not (batch_size,) or (batch_size, num_labels).
        """
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class MSRobertaForMultipleChoice(MSRobertaPreTrainedModel):
    """RobertaForMultipleChoice"""
    def __init__(self, config, *args, **kwargs):
        """
        Initializes an instance of the 'MSRobertaForMultipleChoice' class.

        Args:
            self: The instance of the class.
            config (RobertaConfig): The configuration object for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config, *args, **kwargs)
        self.roberta = MSRobertaModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                  position_ids=None, head_mask=None):
        """
        Constructs the multiple choice model for MSRoberta.

        Args:
            self (MSRobertaForMultipleChoice): The instance of the MSRobertaForMultipleChoice class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, num_choices, sequence_length)
                containing the input IDs.
            token_type_ids (torch.Tensor, optional): The tensor of shape (batch_size, num_choices, sequence_length)
                containing the token type IDs. Default: None.
            attention_mask (torch.Tensor, optional): The tensor of shape (batch_size, num_choices, sequence_length)
                containing the attention mask. Default: None.
            labels (torch.Tensor, optional): The tensor of shape (batch_size,) containing the labels for the multiple
                choice questions. Default: None.
            position_ids (torch.Tensor, optional): The tensor of shape (batch_size, num_choices, sequence_length)
                containing the position IDs. Default: None.
            head_mask (torch.Tensor, optional): The tensor of shape (num_hidden_layers, num_attention_heads)
                containing the head mask. Default: None.

        Returns:
            tuple: A tuple of output tensors. The first element is reshaped_logits of shape (batch_size, num_choices),
                representing the logits for each choice. The remaining elements are the same as the outputs of the
                Roberta model.
        
        Raises:
            TypeError: If the input_ids is not a torch.Tensor.
            ValueError: If the input_ids shape is not (batch_size, num_choices, sequence_length).
            TypeError: If the token_type_ids is not a torch.Tensor or None.
            ValueError: If the token_type_ids shape is not (batch_size, num_choices, sequence_length) when not None.
            TypeError: If the attention_mask is not a torch.Tensor or None.
            ValueError: If the attention_mask shape is not (batch_size, num_choices, sequence_length) when not None.
            TypeError: If the labels is not a torch.Tensor or None.
            ValueError: If the labels shape is not (batch_size,) when not None.
            TypeError: If the position_ids is not a torch.Tensor or None.
            ValueError: If the position_ids shape is not (batch_size, num_choices, sequence_length) when not None.
            TypeError: If the head_mask is not a torch.Tensor or None.
            ValueError: If the head_mask shape is not (num_hidden_layers, num_attention_heads) when not None.
        """
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = F.cross_entropy(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (ops.cumsum(mask, dim=1).astype(mask.dtype) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


__all__ = ['MSRobertaModel', 'MSRobertaPreTrainedModel',
           'MSRobertaForMaskedLM', 'MSRobertaForMultipleChoice',
           'MSRobertaForSequenceClassification']
