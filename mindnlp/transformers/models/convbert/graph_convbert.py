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
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.nn import CrossEntropyLoss
from mindspore.common.initializer import initializer, Normal

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
)
from ...modeling_utils import PreTrainedModel
from .convbert_config import ConvBertConfig
from ...ms_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

MSCONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "conv-bert-base",
]


class MSConvBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initialize the MSConvBertEmbeddings class with the provided configuration.
        
        Args:
            self: The instance of the MSConvBertEmbeddings class.
            config:
                An object containing configuration parameters for the embeddings.

                - Type: Custom configuration object.
                - Purpose: Specifies various settings for initializing embeddings.
                - Restrictions: Must be properly configured with required parameters.

        Returns:
            None.

        Raises:
            None.
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

        self.LayerNorm = nn.LayerNorm(
            config.embedding_size, epsilon=config.layer_norm_eps
        )
        self.dropout_p = config.hidden_dropout_prob
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to(
            (1, -1)
        )

        self.token_type_ids = ops.zeros(
            self.position_ids.shape, dtype=ms.int64)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        '''
        Construct embeddings for the MSConvBert model.

        Args:
            self (MSConvBertEmbeddings): The instance of the MSConvBertEmbeddings class.
            input_ids (Optional[ms.Tensor]): A 2D tensor containing the input token IDs. Default is None.
            token_type_ids (Optional[ms.Tensor]): A 2D tensor containing the token type IDs. Default is None.

        Returns:
            ms.Tensor: A 3D tensor representing the constructed embeddings.

        Raises:
            TypeError: If the input_ids or token_type_ids are not of type ms.Tensor.
            ValueError: If the sequence length derived from input_ids.shape is not valid.
        '''
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = ops.dropout(embeddings, p=self.dropout_p)
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
        Initializes a separable 1D convolutional layer.

        Args:
            self: The instance of the class.
            config: The configuration object containing initialization parameters.
            input_filters (int): The number of input filters.
            output_filters (int): The number of output filters.
            kernel_size (int): The size of the convolutional kernel.

        Returns:
            None.

        Raises:
            ValueError: If input_filters or output_filters is not a positive integer.
            ValueError: If kernel_size is not a positive odd integer.
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
        Method to construct a separable 1D convolution operation on the given hidden states.

        Args:
            self (SeparableConv1D): An instance of the SeparableConv1D class.
            hidden_states (ms.Tensor): The input hidden states tensor on which the separable convolution is applied.
                Must be a Tensor of shape compatible with the convolution operation.

        Returns:
            ms.Tensor:
                Returns a Tensor representing the output of the separable convolution operation on the input hidden states.

        Raises:
            ValueError: If the hidden_states parameter is not a valid Tensor object.
            RuntimeError: If any runtime error occurs during the execution of the convolution operation.
        """
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias
        return x


class MSConvBertSelfAttention(nn.Cell):
    """
    MSConvBertSelfAttention
    """
    def __init__(self, config):
        """Initialize the MSConvBertSelfAttention class.

        Args:
            self: The instance of the MSConvBertSelfAttention class.
            config: An object containing configuration parameters for the self-attention mechanism.
                It should have the following attributes:

                - hidden_size (int): The size of the hidden state.
                - num_attention_heads (int): The number of attention heads.
                - head_ratio (int): The ratio to reduce the number of attention heads.
                - conv_kernel_size (int): The size of the convolutional kernel.
                - embedding_size (optional, int): The size of the embedding.
                If not provided, the hidden_size should be divisible by num_attention_heads.

        Returns:
            None.

        Raises:
            ValueError:
                - If the hidden size is not a multiple of the number of attention heads and the config does not have the attribute 'embedding_size'.
                - If the hidden_size is not divisible by num_attention_heads.
        """
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
            raise ValueError(
                "hidden_size should be divisible by num_attention_heads")

        self.attention_head_size = (
            config.hidden_size // self.num_attention_heads) // 2
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
        self.dropout_p = config.attention_probs_dropout_prob

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
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """
        Constructs the self-attention mechanism for the MSConvBert model.

        Args:
            self (MSConvBertSelfAttention): An instance of the MSConvBertSelfAttention class.
            hidden_states (ms.Tensor): The input hidden states of shape [batch_size, sequence_length, hidden_size].
            attention_mask (Optional[ms.Tensor]): An optional attention mask of shape [batch_size, sequence_length] with
                0s in positions corresponding to padding tokens and 1s elsewhere. Defaults to None.

        Returns:
            context_layer: The output context layer of shape [batch_size, sequence_length, hidden_size].
            attention_mask: An optional attention mask of shape [batch_size, sequence_length] with 0s in positions
                corresponding to padding tokens and 1s elsewhere. This is returned only if attention_mask is not None.

        Raises:
            None.
        """
        mixed_query_layer = self.query(hidden_states)
        batch_size = hidden_states.shape[0]
        # If this is instantiated as a cross-attention cell, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_key_conv_attn_layer = self.key_conv_attn_layer(
            hidden_states.swapaxes(1, 2)
        )
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.swapaxes(1, 2)

        query_layer = self.swapaxes_for_scores(mixed_query_layer)
        key_layer = self.swapaxes_for_scores(mixed_key_layer)
        value_layer = self.swapaxes_for_scores(mixed_value_layer)
        conv_attn_layer = ops.multiply(
            mixed_key_conv_attn_layer, mixed_query_layer)

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
            conv_out_layer, [-1, self.attention_head_size,
                             self.conv_kernel_size]
        )
        conv_out_layer = ops.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = ops.reshape(conv_out_layer, [-1, self.all_head_size])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / \
            ops.sqrt(ms.Tensor(self.attention_head_size))

        # Apply the attention mask is (precomputed for all layers in ConvBertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = ops.dropout(attention_probs, p=self.dropout_p)

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
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        return outputs


class MSConvBertSelfOutput(nn.Cell):
    """
    MSConvBertSelfOutput
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSConvBertSelfOutput class.

        Args:
            self (MSConvBertSelfOutput): The current instance of the MSConvBertSelfOutput class.
            config:
                An object containing the configuration parameters for the MSConvBertSelfOutput class.

                - hidden_size (int): The size of the hidden state.
                - layer_norm_eps (float): The epsilon value used in layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden state.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout_p = config.hidden_dropout_prob

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        '''
        This method constructs the self output for the MSConvBert model.

        Args:
            self (MSConvBertSelfOutput): The instance of the MSConvBertSelfOutput class.
            hidden_states (ms.Tensor): The hidden states tensor representing the output of the model's self-attention mechanism.
            input_tensor (ms.Tensor): The input tensor representing the input to the self output layer.

        Returns:
            ms.Tensor: A tensor representing the constructed self output.

        Raises:
            None.
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout_p)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MSConvBertAttention(nn.Cell):
    """
    MSConvBertAttention
    """
    def __init__(self, config):
        """
        Initializes a new instance of the MSConvBertAttention class.

        Args:
            self: The instance of the class.
            config: The configuration object that contains the settings for the attention module.
                This parameter is of type 'config' and is used to initialize the attention module and output module.
                The config object must have the necessary attributes required by the MSConvBertSelfAttention and MSConvBertSelfOutput classes.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = MSConvBertSelfAttention(config)
        self.output = MSConvBertSelfOutput(config)
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
        self.output.dense = prune_linear_layer(
            self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - \
            len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """
        Constructs the attention mechanism for the MSConvBert model.

        Args:
            self (MSConvBertAttention): The instance of the MSConvBertAttention class.
            hidden_states (ms.Tensor): The input hidden states to be used for attention computation.
            attention_mask (Optional[ms.Tensor]): An optional tensor representing the attention mask. Defaults to None.

        Returns:
            Tuple[ms.Tensor, Optional[ms.Tensor]]: A tuple containing the attention output tensor and an optional tensor.

        Raises:
            None.
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MSConvBertIntermediate(nn.Cell):
    """
    MSConvBertIntermediate
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSConvBertIntermediate class.

        Args:
            self: The instance of the class.
            config (object):
                An object containing configuration settings.

                - Type: Any object.
                - Purpose: Configuration settings for the intermediate layer.
                - Restrictions: Must contain 'hidden_size' and 'intermediate_size' attributes.

        Returns:
            None.

        Raises:
            KeyError: If the 'hidden_act' attribute in the config object does not match any key in ACT2FN dictionary.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        """
        Constructs the intermediate layer for MSConvBert model.

        Args:
            self: Instance of the MSConvBertIntermediate class.
            hidden_states (ms.Tensor): Input tensor representing the hidden states from the previous layer.

        Returns:
            ms.Tensor: Transformed tensor after passing through the intermediate layer.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MSConvBertOutput(nn.Cell):
    """
    MSConvBertOutput
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSConvBertOutput class.

        Args:
            self (MSConvBertOutput): The instance of the MSConvBertOutput class.
            config (object): The configuration object containing parameters for initialization.
                This object should have the following attributes:

                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for LayerNorm.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            ValueError: If any of the required attributes (intermediate_size, hidden_size, layer_norm_eps,
                hidden_dropout_prob) are missing in the config object.
            ValueError: If the values of hidden_size or intermediate_size are not valid integers.
            ValueError: If the values of layer_norm_eps or hidden_dropout_prob are not valid floats.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout_p = config.hidden_dropout_prob

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        """
        The 'construct' method in the 'MSConvBertOutput' class constructs a tensor output based on the provided
        hidden states and input tensor.

        Args:
            self (MSConvBertOutput): The instance of the MSConvBertOutput class.
            hidden_states (ms.Tensor): The hidden states tensor to be processed.
                This tensor should represent the internal states of the model.
            input_tensor (ms.Tensor): The input tensor to be combined with the processed hidden states.
                This tensor should contain the input data to be integrated into the output.

        Returns:
            ms.Tensor: A tensor representing the output constructed based on the hidden states and input tensor.
                The output tensor reflects the combined information from the hidden states and input.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If any input parameter does not meet the required restrictions.
            RuntimeError: If there are issues during the tensor processing or combination steps.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout_p)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MSConvBertLayer(nn.Cell):
    """
    MSConvBertLayer
    """
    def __init__(self, config):
        """
        Initializes an instance of MSConvBertLayer.

        Args:
            self (object): The instance of the MSConvBertLayer class.
            config (object): An object containing configuration parameters for the layer.
                This parameter is used to configure the behavior of the layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.seq_len_dim = 1
        self.attention = MSConvBertAttention(config)
        self.intermediate = MSConvBertIntermediate(config)
        self.output = MSConvBertOutput(config)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """
        Constructs a Convolutional BERT layer.

        Args:
            self (MSConvBertLayer): The instance of the MSConvBertLayer class.
            hidden_states (ms.Tensor): The hidden states of the input sequence. Shape: [batch_size, sequence_length, hidden_size].
            attention_mask (ms.Tensor): The attention mask to avoid performing attention on padding tokens. Shape: [batch_size, sequence_length].

        Returns:
            layer_output (ms.Tensor): The output of the convolutional BERT layer. Shape: [batch_size, sequence_length, hidden_size].
            outputs (Optional[ms.Tensor]): Additional outputs from the attention mechanism.

        Raises:
            None
        """
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class MSConvBertEncoder(nn.Cell):
    """
    MSConvBertEncoder
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSConvBertEncoder class.

        Args:
            self (MSConvBertEncoder): The instance of the class.
            config (object): The configuration object for the MSConvBertEncoder.
                It contains various parameters and settings for the encoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList(
            [MSConvBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        '''
        Constructs the MSConvBertEncoder by processing the hidden states with attention mask.

        Args:
            self: The instance of the MSConvBertEncoder class.
            hidden_states (ms.Tensor): The input hidden states to be processed by the encoder.
            attention_mask (ms.Tensor): The attention mask to be applied during the encoding process.

        Returns:
            Union[Tuple, BaseModelOutputWithCrossAttentions]:
                Returns the processed hidden states, which could be a Tuple or BaseModelOutputWithCrossAttentions.

        Raises:
            None.
        '''
        for i, layer_cell in enumerate(self.layer):
            layer_outputs = layer_cell(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class MSConvBertModel(ConvBertPreTrainedModel):
    """
    MSConvBertModel
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSConvBertModel class.

        Args:
            self: The instance of the MSConvBertModel class.
            config: A dictionary containing configuration parameters for the model initialization.
                This dictionary must include the necessary settings for the model to be properly configured.
                Expected keys may include settings related to embeddings, encoder, and other model specifics.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not a dictionary.
            ValueError: If required keys are missing in the 'config' dictionary.
        """
        super().__init__(config)
        self.embeddings = MSConvBertEmbeddings(config)
        self.encoder = MSConvBertEncoder(config)
        self.config = config
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the MSConvBertModel.

        Args:
            self (MSConvBertModel): The instance of the MSConvBertModel class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings of the MSConvBertModel.

        Args:
            self (MSConvBertModel): The instance of the MSConvBertModel class.
            new_embeddings (torch.Tensor): The new word embeddings to be set for the model.

        Returns:
            None: The method modifies the self.embeddings.word_embeddings attribute directly.

        Raises:
            None.

        Description:
            This method allows for the setting of new word embeddings for the MSConvBertModel.
            The new_embeddings parameter should be a tensor containing the new word embeddings.
            The method updates the self.embeddings.word_embeddings attribute of the MSConvBertModel instance with
            the provided new_embeddings.

        Note:
            The new_embeddings tensor should have the same shape as the existing word_embeddings tensor.
            It is important to ensure that the dimensions of the new_embeddings tensor match the word_embeddings tensor
            of the model, otherwise unexpected behavior may occur.
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
        input_ids: ms.Tensor,
        attention_mask: ms.Tensor,
        token_type_ids: ms.Tensor,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        """
        Constructs the MSConvBertModel by processing the input data.
        
        Args:
            self: The object instance.
            
            input_ids (ms.Tensor): The input tensor containing the token indices. 
                Shape: (batch_size, sequence_length).
                
            attention_mask (ms.Tensor): The attention mask tensor to avoid attending to padding tokens.
                Shape: (batch_size, sequence_length).
                
            token_type_ids (ms.Tensor): The token type ids tensor to differentiate between different sequences in the input.
                Shape: (batch_size, sequence_length).
        
        Returns:
            Union[Tuple, BaseModelOutputWithCrossAttentions]: The output hidden states of the MSConvBertModel.
            
        Raises:
            None.
        """
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.shape

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
        )

        return hidden_states


class MSConvBertForQuestionAnswering(ConvBertPreTrainedModel):
    """
    MSConvBertForQuestionAnswering
    """
    def __init__(self, config):
        """
        Initialize the MSConvBertForQuestionAnswering class.
        
        Args:
            self: The instance of the MSConvBertForQuestionAnswering class.
            config: An instance of the configuration class containing the model configuration.
                It must include the number of labels (num_labels) for the model.
                
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convbert = MSConvBertModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
        Constructs the Question Answering model for MSConvBert.
        
        Args:
            self: The object instance.
            input_ids (Optional[ms.Tensor]): The input tensor of token indices. Default is None.
            attention_mask (Optional[ms.Tensor]): The attention mask tensor. Default is None.
            token_type_ids (Optional[ms.Tensor]): The token type ids tensor. Default is None.
            start_positions (Optional[ms.Tensor]): The tensor of start positions for the answer span. Default is None.
            end_positions (Optional[ms.Tensor]): The tensor of end positions for the answer span. Default is None.
        
        Returns:
            Union[Tuple, QuestionAnsweringModelOutput]:
                The total loss value or a tuple containing the total loss value and the QuestionAnsweringModelOutput.
        
        Raises:
            None.
        """
        outputs = self.convbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.shape[1]
        start_positions = start_positions.clamp(0, ignored_index).to(ms.int32)
        end_positions = end_positions.clamp(0, ignored_index).to(ms.int32)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        return total_loss


__all__ = [
    "MSCONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MSConvBertForQuestionAnswering",
]
