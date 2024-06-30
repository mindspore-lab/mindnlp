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
""" ConvBERT model."""

from typing import Optional, Tuple, Union
import math
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, Normal

from ...activations import ACT2FN, get_activation
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from .convbert_config import ConvBertConfig
from ...ms_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "conv-bert-base",
    "conv-bert-medium-small",
    "conv-bert-small",
]


class ConvBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes the ConvBertEmbeddings object.
        
        Args:
            self (ConvBertEmbeddings): The current instance of the ConvBertEmbeddings class.
            config:
                An object containing the configuration parameters for the ConvBert model.

                - vocab_size (int): The size of the vocabulary.
                - embedding_size (int): The size of the word embeddings.
                - pad_token_id (int): The index of the padding token in the vocabulary.
                - max_position_embeddings (int): The maximum number of positions in the input sequence.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the embeddings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.embedding_size, epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to(
            (1, -1)
        )

        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=ms.int64)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """
        Constructs the embeddings for ConvBert model.

        Args:
            self (ConvBertEmbeddings): The instance of the ConvBertEmbeddings class.
            input_ids (Optional[ms.Tensor]): The input tensor containing the token indices. Default is None.
            token_type_ids (Optional[ms.Tensor]): The input tensor containing the token type indices. Default is None.
            position_ids (Optional[ms.Tensor]): The input tensor containing the position indices. Default is None.
            inputs_embeds (Optional[ms.Tensor]): The input tensor containing the embedded representation of the input. Default is None.

        Returns:
            ms.Tensor: The constructed embeddings tensor.

        Raises:
            None.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=ms.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ConvBertConfig
    base_model_prefix = "convbert"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight = ms.Parameter(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.bias is not None:
                cell.bias = ms.Parameter(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            cell.weight = ms.Parameter(
                initializer(
                    Normal(mean=0.0, sigma=self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            # if cell.padding_idx is not None:
            #     cell.weight.data[cell.padding_idx].set_data(
            #         initializer('zeros',
            #                     cell.weight.data[cell.padding_idx].shape,
            #                     cell.weight.data[cell.padding_idx].dtype)
            #     )
        elif isinstance(cell, nn.LayerNorm):
            cell.bias = ms.Parameter(
                initializer("zeros", cell.bias.shape, cell.bias.dtype)
            )
            cell.weight = ms.Parameter(
                initializer("ones", cell.weight.shape, cell.weight.dtype)
            )


class SeparableConv1D(nn.Cell):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""
    def __init__(self, config, input_filters, output_filters, kernel_size):
        """
        Initializes a SeparableConv1D instance.

        Args:
            self: The instance of the class.
            config: An object containing configuration settings.
            input_filters: An integer indicating the number of input filters.
            output_filters: An integer indicating the number of output filters.
            kernel_size: An integer specifying the size of the kernel.

        Returns:
            None.

        Raises:
            ValueError: If input_filters is not an integer.
            ValueError: If output_filters is not an integer.
            ValueError: If kernel_size is not an integer.
            ValueError: If config.initializer_range is not a valid value.
            ValueError: If pad_mode is not 'pad'.
            ValueError: If the dimensions of the weights for depthwise and pointwise convolutions do not match.
        """
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            group=input_filters,
            pad_mode="pad",
            padding=kernel_size // 2,
            has_bias=False,
        )
        self.pointwise = nn.Conv1d(
            input_filters, output_filters, kernel_size=1, has_bias=False
        )
        self.bias = ms.Parameter(ops.zeros((output_filters, 1)))

        self.depthwise.weight = ms.Parameter(
            initializer(
                Normal(sigma=config.initializer_range, mean=0.0),
                self.depthwise.weight.shape,
                self.depthwise.weight.dtype,
            )
        )
        self.pointwise.weight = ms.Parameter(
            initializer(
                Normal(sigma=config.initializer_range, mean=0.0),
                self.pointwise.weight.shape,
                self.depthwise.weight.dtype,
            )
        )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        """
        Constructs a separable 1D convolution operation.

        Args:
            self (SeparableConv1D): An instance of the SeparableConv1D class.
            hidden_states (ms.Tensor): The input hidden states tensor to be convolved.
                Expected to be of shape (batch_size, input_channels, sequence_length).

        Returns:
            ms.Tensor: The output tensor after applying depthwise and pointwise convolutions, and adding bias.
                The shape of the output tensor is determined by the convolution operations performed.

        Raises:
            TypeError: If the input hidden_states is not a ms.Tensor object.
            ValueError: If the dimensions of the hidden_states tensor are not valid for convolution operations.
        """
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias
        return x


class ConvBertSelfAttention(nn.Cell):
    """
    ConvBertSelfAttention
    """
    def __init__(self, config):
        '''
        Initializes a new instance of the ConvBertSelfAttention class.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing the settings for the self-attention mechanism.

        Returns:
            None

        Raises:
            ValueError:
                If the hidden size is not divisible by the number of attention heads or if the hidden size
                is not a multiple of the number of attention heads.

        '''
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        new_num_attention_heads = config.num_attention_heads // config.head_ratio
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio

        self.conv_kernel_size = config.conv_kernel_size
        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size should be divisible by num_attention_heads")

        self.attention_head_size = (config.hidden_size // self.num_attention_heads) // 2
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.key_conv_attn_layer = SeparableConv1D(
            config, config.hidden_size, self.all_head_size, self.conv_kernel_size
        )
        self.conv_kernel_layer = nn.Dense(
            self.all_head_size, self.num_attention_heads * self.conv_kernel_size
        )
        self.conv_out_layer = nn.Dense(config.hidden_size, self.all_head_size)
        self.unfold = nn.Unfold(
            ksizes=[1, self.conv_kernel_size, 1, 1],
            rates=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding="same",
        )
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def swapaxes_for_scores(self, x):
        """swapaxes for scores"""
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        '''
        The `construct` method in the `ConvBertSelfAttention` class performs the construction of self-attention
        mechanism using convolutional operations.

        Args:
            self: The instance of the ConvBertSelfAttention class.
            hidden_states (ms.Tensor):
                The input tensor of shape [batch_size, sequence_length, hidden_size] representing the hidden states
                of the input sequence.
            attention_mask (Optional[ms.Tensor]):
                An optional tensor of shape [batch_size, 1, sequence_length, sequence_length] containing attention mask
                for the input sequence. Default is None.
            head_mask (Optional[ms.Tensor]):
                An optional tensor of shape [num_attention_heads] representing the mask for attention heads.
                Default is None.
            encoder_hidden_states (Optional[ms.Tensor]):
                An optional tensor of shape [batch_size, sequence_length, hidden_size] representing the hidden states
                of the encoder. Default is None.
            output_attentions (Optional[bool]): Whether to output attention probabilities. Default is False.

        Returns:
            Tuple[ms.Tensor, Optional[ms.Tensor]]:
                A tuple containing the context layer tensor of shape [batch_size, sequence_length, hidden_size]
                and the optional attention probabilities tensor of shape
            [batch_size, num_attention_heads, sequence_length, sequence_length].

        Raises:
            None.
        '''
        mixed_query_layer = self.query(hidden_states)
        batch_size = hidden_states.shape[0]
        # If this is instantiated as a cross-attention cell, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        mixed_key_conv_attn_layer = self.key_conv_attn_layer(
            hidden_states.swapaxes(1, 2)
        )
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.swapaxes(1, 2)

        query_layer = self.swapaxes_for_scores(mixed_query_layer)
        key_layer = self.swapaxes_for_scores(mixed_key_layer)
        value_layer = self.swapaxes_for_scores(mixed_value_layer)
        conv_attn_layer = ops.multiply(mixed_key_conv_attn_layer, mixed_query_layer)

        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = ops.reshape(
            conv_kernel_layer, [-1, self.conv_kernel_size, 1]
        )
        conv_kernel_layer = ops.softmax(conv_kernel_layer, axis=1)

        conv_out_layer = self.conv_out_layer(hidden_states)
        conv_out_layer = ops.reshape(
            conv_out_layer, [batch_size, -1, self.all_head_size]
        )
        conv_out_layer = conv_out_layer.swapaxes(1, 2).unsqueeze(-1)
        conv_out_layer = ops.unfold(
            conv_out_layer,
            kernel_size=[self.conv_kernel_size, 1],
            dilation=1,
            padding=[(self.conv_kernel_size - 1) // 2, 0],
            stride=1,
        )
        conv_out_layer = conv_out_layer.swapaxes(1, 2).reshape(
            batch_size, -1, self.all_head_size, self.conv_kernel_size
        )
        conv_out_layer = ops.reshape(
            conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size]
        )
        conv_out_layer = ops.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = ops.reshape(conv_out_layer, [-1, self.all_head_size])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ConvBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)

        conv_out = ops.reshape(
            conv_out_layer,
            [batch_size, -1, self.num_attention_heads, self.attention_head_size],
        )
        context_layer = ops.cat([context_layer, conv_out], 2)

        # conv and context
        new_context_layer_shape = context_layer.shape[:-2] + (
            self.num_attention_heads * self.attention_head_size * 2,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class ConvBertSelfOutput(nn.Cell):
    """
    ConvBertSelfOutput
    """
    def __init__(self, config):
        """
        Initializes an instance of the ConvBertSelfOutput class.

        Args:
            self (ConvBertSelfOutput): The instance of the ConvBertSelfOutput class.
            config: The configuration object that holds various hyperparameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        """
        Constructs the output of the ConvBertSelfOutput layer.

        Args:
            self (ConvBertSelfOutput): An instance of the ConvBertSelfOutput class.
            hidden_states (ms.Tensor): The hidden states tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the output of the previous layer.
            input_tensor (ms.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the input to the layer.

        Returns:
            ms.Tensor: The output tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the constructed output of the ConvBertSelfOutput layer.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ConvBertAttention(nn.Cell):
    """
    ConvBertAttention
    """
    def __init__(self, config):
        """
        Initializes an instance of the ConvBertAttention class.

        Args:
            self (ConvBertAttention): The instance of the ConvBertAttention class.
            config: The configuration parameters for the ConvBertAttention class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.self = ConvBertSelfAttention(config)
        self.output = ConvBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """prune heads"""
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """
        This method constructs the output of ConvBertAttention.

        Args:
            self: The instance of ConvBertAttention.
            hidden_states (ms.Tensor): The input hidden states for the attention layer.
            attention_mask (Optional[ms.Tensor]): Optional tensor specifying which elements in the input sequence should be attended to.
            head_mask (Optional[ms.Tensor]): Optional tensor specifying the mask to be applied to the attention heads.
            encoder_hidden_states (Optional[ms.Tensor]): Optional tensor representing the hidden states of the encoder.
            output_attentions (Optional[bool]): Optional flag indicating whether to output the attention weights. Default is False.

        Returns:
            Tuple[ms.Tensor, Optional[ms.Tensor]]:
                A tuple containing the attention output tensor and optionally the attention weights.

        Raises:
            None.
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class GroupedLinearLayer(nn.Cell):
    """
    GroupedLinearLayer
    """
    def __init__(self, input_size, output_size, num_groups):
        """
        Initializes a GroupedLinearLayer object.

        Args:
            self (GroupedLinearLayer): The instance of the GroupedLinearLayer class.
            input_size (int): The size of the input tensor.
            output_size (int): The size of the output tensor.
            num_groups (int): The number of groups to divide the input and output tensors into.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups
        self.weight = nn.Parameter(
            ops.zeros((self.num_groups, self.group_in_dim, self.group_out_dim))
        )
        self.bias = nn.Parameter(ops.zeros(output_size))

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        """Constructs a grouped linear layer.

        Args:
            self (GroupedLinearLayer): The instance of the GroupedLinearLayer class.
            hidden_states (ms.Tensor): The input tensor of shape [batch_size, input_size].

        Returns:
            ms.Tensor: The output tensor of shape [batch_size, output_size].

        Raises:
            TypeError: If `hidden_states` is not of type `ms.Tensor`.
            ValueError: If the shape of `hidden_states` is not compatible with the expected shape [batch_size, input_size].
            ValueError: If `self.weight` is not of shape [num_groups, group_in_dim, output_size].
            ValueError: If `self.bias` is not of shape [output_size].

        Note:
            The `hidden_states` tensor represents the input to the grouped linear layer. It is expected to have a shape
            of [batch_size, input_size].

            The grouped linear layer applies a linear transformation to the input tensor by grouping the input features
            into `num_groups` groups. The `group_in_dim` represents the number of features in each group. The output tensor
            has a shape of [batch_size, output_size].

            The linear transformation is performed by reshaping the input tensor to a shape of [batch_size * num_groups,
            group_in_dim], permuting the dimensions to [num_groups, batch_size, group_in_dim], and performing matrix
            multiplication with the weight tensor of shape [num_groups, group_in_dim, output_size]. The result tensor is
            then reshaped back to [batch_size, -1, output_size] and added with the bias tensor of shape [output_size].

            The grouped linear layer is typically used in neural network architectures to introduce non-linearity and
            increase model capacity by learning more complex relationships between input and output features.
        """
        batch_size = list(hidden_states.shape)[0]
        x = ops.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim])
        x = x.permute(1, 0, 2)
        x = ops.matmul(x, self.weight)
        x = x.permute(1, 0, 2)
        x = ops.reshape(x, [batch_size, -1, self.output_size])
        x = x + self.bias
        return x


class ConvBertIntermediate(nn.Cell):
    """
    ConvBertIntermediate
    """
    def __init__(self, config):
        """
        Initializes an instance of the ConvBertIntermediate class with the provided configuration.

        Args:
            self (ConvBertIntermediate): The current instance of the ConvBertIntermediate class.
            config (object):
                An object containing configuration parameters for the intermediate layer.

                - num_groups (int): The number of groups for the intermediate layer.
                Must be an integer greater than or equal to 1.
                - hidden_size (int): The size of the hidden layer.
                Must be an integer specifying the size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                Must be an integer specifying the size of the intermediate layer.
                - hidden_act (str or function): The activation function for the hidden layer.
                Must be a string representing a predefined activation function or a custom activation function.

        Returns:
            None.

        Raises:
            ValueError: If the num_groups parameter is not an integer greater than or equal to 1.
            ValueError: If the hidden_size parameter is not an integer.
            ValueError: If the intermediate_size parameter is not an integer.
            ValueError: If the hidden_act parameter is not a valid string or function.
        """
        super().__init__()
        if config.num_groups == 1:
            self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        else:
            self.dense = GroupedLinearLayer(
                input_size=config.hidden_size,
                output_size=config.intermediate_size,
                num_groups=config.num_groups,
            )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        '''
        This method constructs the intermediate layer in the ConvBert model.

        Args:
            self (ConvBertIntermediate): The instance of the ConvBertIntermediate class.
            hidden_states (ms.Tensor): The input tensor containing the hidden states.

        Returns:
            ms.Tensor: Returns the tensor representing the constructed intermediate layer.

        Raises:
            None
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ConvBertOutput(nn.Cell):
    """
    ConvBertOutput
    """
    def __init__(self, config):
        """
        Initializes a new instance of the ConvBertOutput class.

        Args:
            self: The object instance.
            config:
                An object of type 'config' containing the configuration parameters for the ConvBertOutput class.

                - num_groups (int): The number of groups. If equal to 1, a Dense layer is used.
                Otherwise, a GroupedLinearLayer is used. (Restrictions: Must be a positive integer)
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        if config.num_groups == 1:
            self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        else:
            self.dense = GroupedLinearLayer(
                input_size=config.intermediate_size,
                output_size=config.hidden_size,
                num_groups=config.num_groups,
            )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        """
        Constructs the output tensor for the ConvBertOutput class.

        Args:
            self (ConvBertOutput): The instance of the ConvBertOutput class.
            hidden_states (ms.Tensor): The input tensor representing the hidden states.
            input_tensor (ms.Tensor): The input tensor.

        Returns:
            ms.Tensor: The output tensor representing the constructed hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ConvBertLayer(nn.Cell):
    """
    ConvBertLayer
    """
    def __init__(self, config):
        """
        Initializes a new instance of the ConvBertLayer class.

        Args:
            self: The current object instance.
            config: An object of type 'config' containing the configuration settings for the ConvBertLayer.

        Returns:
            None.

        Raises:
            TypeError: Raised if the 'add_cross_attention' flag is set to True but the ConvBertLayer is not used as a decoder model.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ConvBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = ConvBertAttention(config)
        self.intermediate = ConvBertIntermediate(config)
        self.output = ConvBertOutput(config)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """
        Constructs a ConvBertLayer.

        This method applies the ConvBertLayer transformation to the input hidden states and returns the
        transformed output. It also supports cross-attention if `encoder_hidden_states` are provided.

        Args:
            self (ConvBertLayer): An instance of the ConvBertLayer class.
            hidden_states (ms.Tensor): The input hidden states of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[ms.Tensor]):
                The attention mask of shape (batch_size, seq_len) or (batch_size, seq_len, seq_len). Defaults to None.
            head_mask (Optional[ms.Tensor]):
                The head mask of shape (num_heads,) or (num_layers, num_heads). Defaults to None.
            encoder_hidden_states (Optional[ms.Tensor]):
                The hidden states of the encoder if cross-attention is enabled. Defaults to None.
            encoder_attention_mask (Optional[ms.Tensor]):
                The attention mask of the encoder if cross-attention is enabled. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.

        Returns:
            Tuple[ms.Tensor, Optional[ms.Tensor]]: A tuple containing the transformed output tensor and
                optional attention tensors.

        Raises:
            AttributeError:
                If `encoder_hidden_states` are passed, `self` has to be instantiated with cross-attention layers
                by setting `config.add_cross_attention=True`.
        """
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            cross_attention_outputs = self.crossattention(
                attention_output,
                encoder_attention_mask,
                head_mask,
                encoder_hidden_states,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        """feed forward chunk"""
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ConvBertEncoder(nn.Cell):
    """
    ConvBertEncoder
    """
    def __init__(self, config):
        """
        __init__(self, config)

        Initializes a ConvBertEncoder instance.

        Args:
            self (object): The instance of the ConvBertEncoder class.
            config (object): An object containing configuration parameters for the ConvBertEncoder.
                The config object should have attributes related to the encoder's configuration,
                such as the number of hidden layers, and other relevant settings.
                It should be an instance of a compatible configuration class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList(
            [ConvBertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        """
        This method constructs the ConvBertEncoder by processing the input hidden states through a series of layers.

        Args:
            self: The instance of the ConvBertEncoder class.
            hidden_states (ms.Tensor): The input hidden states to be processed by the encoder.
            attention_mask (Optional[ms.Tensor]): An optional tensor specifying the attention mask for the input.
            head_mask (Optional[ms.Tensor]): An optional tensor providing mask for heads in the multi-head attention mechanism.
            encoder_hidden_states (Optional[ms.Tensor]): An optional tensor representing hidden states from an encoder.
            encoder_attention_mask (Optional[ms.Tensor]): An optional tensor specifying the attention mask for the encoder hidden states.
            output_attentions (Optional[bool]): A flag indicating whether to output attention tensors.
            output_hidden_states (Optional[bool]): A flag indicating whether to output hidden states at each layer.
            return_dict (Optional[bool]): A flag indicating whether to return the output as a dictionary.

        Returns:
            Union[Tuple, BaseModelOutputWithCrossAttentions]: The output of the method which can be a tuple of relevant
            values or a BaseModelOutputWithCrossAttentions object containing the processed hidden states, attentions,
            and cross-attentions.

        Raises:
            None.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        for i, layer_cell in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_cell.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_cell(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ConvBertPredictionHeadTransform(nn.Cell):
    """
    ConvBertPredictionHeadTransform
    """
    def __init__(self, config):
        """
        Initializes a ConvBertPredictionHeadTransform object.

        Args:
            self (ConvBertPredictionHeadTransform): The instance of the ConvBertPredictionHeadTransform class.
            config:
                An object containing configuration parameters for the transformation.

                - Type: Any
                - Purpose: Specifies the configuration settings for the transformation.
                - Restrictions: Must contain the following attributes:

                    - hidden_size: Integer representing the size of the hidden layer.
                    - hidden_act: Activation function for the hidden layer. Can be a string or a callable.
                    - layer_norm_eps: Epsilon value for LayerNorm.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is of an unexpected type.
            AttributeError: If the config object does not contain the required attributes.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        """
        This method constructs the prediction head transformation for ConvBert.

        Args:
            self (ConvBertPredictionHeadTransform): The instance of ConvBertPredictionHeadTransform.
            hidden_states (ms.Tensor): The input tensor representing hidden states.

        Returns:
            ms.Tensor: The transformed hidden states tensor.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ConvBertModel(ConvBertPreTrainedModel):
    """
    ConvBertModel
    """
    def __init__(self, config):
        """
        Initializes the ConvBertModel class.

        Args:
            self (object): The instance of the ConvBertModel class.
            config (object): An object containing configuration parameters for the model.
                This object should include settings such as embedding size, hidden size, etc.
                It is used to configure the model's parameters and behavior.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.embeddings = ConvBertEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Dense(
                config.embedding_size, config.hidden_size
            )

        self.encoder = ConvBertEncoder(config)
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the ConvBertModel.

        Args:
            self (ConvBertModel): The object instance of the ConvBertModel class.

        Returns:
            word_embeddings: The method returns the word embeddings from the input embeddings.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the ConvBertModel.

        Args:
            self (ConvBertModel): The instance of the ConvBertModel class.
            new_embeddings (Tensor): The new embeddings to be set for input.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        '''
        Construct method in ConvBertModel class.

        Args:
            self (ConvBertModel): The instance of the ConvBertModel class.
            input_ids (Optional[ms.Tensor]): Input tensor containing the indices of input sequence tokens in the vocabulary.
            attention_mask (Optional[ms.Tensor]): Mask tensor showing which elements of the input sequence should be attended to.
            token_type_ids (Optional[ms.Tensor]): Tensor containing the type embeddings of the input tokens.
            position_ids (Optional[ms.Tensor]): Tensor containing the position embeddings of the input tokens.
            head_mask (Optional[ms.Tensor]): Tensor to mask heads of the attention mechanism.
            inputs_embeds (Optional[ms.Tensor]): Input embeddings for the sequence.
            output_attentions (Optional[bool]): Whether to return attentions tensors.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary of outputs in addition to the traditional tuple output.

        Returns:
            Union[Tuple, BaseModelOutputWithCrossAttentions]:
                A tuple or BaseModelOutputWithCrossAttentions object, containing the hidden states, attentions,
                and/or other model outputs.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified simultaneously.
            ValueError: If neither input_ids nor inputs_embeds are specified.
        '''
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=ms.int64)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states


class ConvBertGeneratorPredictions(nn.Cell):
    """Prediction cell for the generator, made up of two dense layers."""
    def __init__(self, config):
        """
        Initializes an instance of the ConvBertGeneratorPredictions class.

        Args:
            self (ConvBertGeneratorPredictions): The current instance of the ConvBertGeneratorPredictions class.
            config: A configuration object containing various settings for the ConvBertGeneratorPredictions.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.activation = get_activation("gelu")
        self.LayerNorm = nn.LayerNorm(
            config.embedding_size, epsilon=config.layer_norm_eps
        )
        self.dense = nn.Dense(config.hidden_size, config.embedding_size)

    def construct(self, generator_hidden_states: ms.Tensor) -> ms.Tensor:
        """
        Constructs the generator predictions based on the given generator hidden states.

        Args:
            self: An instance of the ConvBertGeneratorPredictions class.
            generator_hidden_states (ms.Tensor): The hidden states generated by the generator.
                It should be a tensor of shape (batch_size, sequence_length, hidden_size).
                The hidden_size is the dimensionality of the hidden states.

        Returns:
            ms.Tensor: The constructed generator predictions.
                It is a tensor of shape (batch_size, sequence_length, hidden_size).
                The hidden_size is the same as the input hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class ConvBertForMaskedLM(ConvBertPreTrainedModel):
    """
    ConvBertForMaskedLM
    """
    _tied_weights_keys = ["generator.lm_head.weight"]

    def __init__(self, config):
        """Initialize a ConvBertForMaskedLM object.

        Args:
            self (ConvBertForMaskedLM): An instance of the ConvBertForMaskedLM class.
            config (object): The configuration object that contains the model's hyperparameters and settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.convbert = ConvBertModel(config)
        self.generator_predictions = ConvBertGeneratorPredictions(config)

        self.generator_lm_head = nn.Dense(config.embedding_size, config.vocab_size)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the ConvBertForMaskedLM model.

        Args:
            self: The instance of the ConvBertForMaskedLM class.

        Returns:
            None: This method returns the generator_lm_head attribute from the ConvBertForMaskedLM model,
                which contains the output embeddings.

        Raises:
            None.
        """
        return self.generator_lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the ConvBertForMaskedLM model.

        Args:
            self (ConvBertForMaskedLM): The instance of the ConvBertForMaskedLM class.
            new_embeddings: The new embeddings to be set for the output.
                This should be of the same type and shape as the current embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.generator_lm_head = new_embeddings

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        generator_hidden_states = self.convbert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        generator_sequence_output = generator_hidden_states[0]

        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)

        loss = None
        # Masked language modeling softmax layer
        if labels is not None:
            labels = labels.to(ms.int32)
            loss = ops.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )


class ConvBertClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        """
        Initializes an instance of the ConvBertClassificationHead class.

        Args:
            self (ConvBertClassificationHead): The instance of the class.
            config: The configuration object that contains the necessary parameters for initialization.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Dense(config.hidden_size, config.num_labels)

        self.config = config

    def construct(self, hidden_states: ms.Tensor, **kwargs) -> ms.Tensor:
        """
        This method constructs a classification head for ConvBert model.

        Args:
            self: The instance of the ConvBertClassificationHead class.
            hidden_states (ms.Tensor):
                The input tensor containing the hidden states from the ConvBert model.
                It is expected to have a shape of [batch_size, sequence_length, hidden_size].

        Returns:
            ms.Tensor: A tensor representing the output of the classification head. It has a shape of [batch_size, num_labels].

        Raises:
            None
        """
        x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ConvBertForSequenceClassification(ConvBertPreTrainedModel):
    """
    ConvBertForSequenceClassification
    """
    def __init__(self, config):
        """
        Initializes a new instance of ConvBertForSequenceClassification.

        Args:
            self (ConvBertForSequenceClassification): The current instance of the ConvBertForSequenceClassification class.
            config (ConvBertConfig):
                The configuration object for ConvBertForSequenceClassification.

                - num_labels (int): The number of labels for classification.
                - ... (other configuration parameters)

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.convbert = ConvBertModel(config)
        self.classifier = ConvBertClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.convbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(ms.int32)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in [ms.int64, ms.int32]:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ConvBertForMultipleChoice(ConvBertPreTrainedModel):
    """
    ConvBertForMultipleChoice
    """
    def __init__(self, config):
        """
        Initialize the ConvBertForMultipleChoice class.

        Args:
            self (object): The instance of the ConvBertForMultipleChoice class.
            config (object): The configuration object containing various parameters for the model initialization.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)

        self.convbert = ConvBertModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        input_ids = (
            input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        )
        attention_mask = (
            attention_mask.view(-1, attention_mask.shape[-1])
            if attention_mask is not None
            else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.shape[-1])
            if token_type_ids is not None
            else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.shape[-1])
            if position_ids is not None
            else None
        )
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.convbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        pooled_output = self.sequence_summary(sequence_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            labels = labels.to(ms.int32)
            loss = ops.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ConvBertForTokenClassification(ConvBertPreTrainedModel):
    """
    ConvBertForTokenClassification
    """
    def __init__(self, config):
        """
        Initializes an instance of the ConvBertForTokenClassification class.

        Args:
            self (ConvBertForTokenClassification): The object itself.
            config: The configuration object containing various settings for the ConvBert model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.convbert = ConvBertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.convbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(ms.int32)
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ConvBertForQuestionAnswering(ConvBertPreTrainedModel):
    """
    ConvBertForQuestionAnswering
    """
    def __init__(self, config):
        """
        Initializes an instance of ConvBertForQuestionAnswering.

        Args:
            self (object): The instance of the class.
            config (object):
                Configuration object containing the model's settings.

                - num_labels (int): The number of labels for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convbert = ConvBertModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
        Constructs the ConvBertForQuestionAnswering model.

        Args:
            self: An instance of the ConvBertForQuestionAnswering class.
            input_ids (Optional[ms.Tensor]): The input token IDs. Default is None.
            attention_mask (Optional[ms.Tensor]): The attention mask. Default is None.
            token_type_ids (Optional[ms.Tensor]): The token type IDs. Default is None.
            position_ids (Optional[ms.Tensor]): The position IDs. Default is None.
            head_mask (Optional[ms.Tensor]): The head mask. Default is None.
            inputs_embeds (Optional[ms.Tensor]): The input embeddings. Default is None.
            start_positions (Optional[ms.Tensor]): The start positions for question answering. Default is None.
            end_positions (Optional[ms.Tensor]): The end positions for question answering. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary output. Default is None.

        Returns:
            Union[Tuple, QuestionAnsweringModelOutput]:
                The model output.

                - If return_dict is False, returns a tuple containing the start logits, end logits, and additional outputs.
                - If return_dict is True, returns a QuestionAnsweringModelOutput object containing the loss,
                start logits, end logits, hidden states, and attentions.
        
        Raises:
            None.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.convbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index).to(ms.int32)
            end_positions = end_positions.clamp(0, ignored_index).to(ms.int32)

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "ConvBertModel",
    "ConvBertForMaskedLM",
    "ConvBertForMultipleChoice",
    "ConvBertForSequenceClassification",
    "ConvBertGeneratorPredictions",
    "ConvBertForTokenClassification",
    "ConvBertForQuestionAnswering",
]
