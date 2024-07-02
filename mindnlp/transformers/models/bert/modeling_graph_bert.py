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

"""MindNLP bert model"""
import mindspore.common.dtype as mstype
from mindspore import nn, ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, Normal
from mindnlp.modules.functional import make_causal_mask, finfo
from .configuration_bert import BertConfig
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel


class MSBertEmbeddings(nn.Cell):
    """
    Embeddings for BERT, include word, position and token_type
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertEmbeddings class.
        
        Args:
            self: The object instance.
            config: An object of the config class containing the configuration parameters for the embeddings.
            
        Returns:
            None.
            
        Raises:
            None.
        
        This method initializes the MSBertEmbeddings object by setting up the word embeddings, position embeddings,
        token type embeddings, layer normalization, and dropout. The configuration parameters are used to determine
        the size of the embeddings and other properties.
        
        - The 'word_embeddings' attribute is an instance of the nn.Embedding class, which represents a lookup table
        for word embeddings. It takes the vocabulary size (config.vocab_size) and hidden size (config.hidden_size) as 
        arguments.
        - The 'position_embeddings' attribute is an instance of the nn.Embedding class, which represents a lookup table
        for position embeddings. It takes the maximum position embeddings (config.max_position_embeddings) and hidden size 
        (config.hidden_size) as arguments.
        - The 'token_type_embeddings' attribute is an instance of the nn.Embedding class, which represents a lookup table
        for token type embeddings. It takes the token type vocabulary size (config.type_vocab_size) and hidden size 
        (config.hidden_size) as arguments.
        - The 'LayerNorm' attribute is an instance of the nn.LayerNorm class, which applies layer normalization to the
        input embeddings. It takes the hidden size (config.hidden_size) and epsilon (config.layer_norm_eps) as arguments.
        - The 'dropout' attribute is an instance of the nn.Dropout class, which applies dropout regularization to the
        input embeddings. It takes the dropout probability (config.hidden_dropout_prob) as an argument.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids, position_ids):
        """
        This method constructs the embeddings for MSBert model.
        
        Args:
            self (object): The object instance of MSBertEmbeddings class.
            input_ids (tensor): The input tensor containing the token ids for the input sequence.
            token_type_ids (tensor): The token type ids to distinguish different sentences in the input sequence.
            position_ids (tensor): The position ids to indicate the position of each token in the input sequence.
        
        Returns:
            tensor: The constructed embeddings for the input sequence represented as a tensor.
        
        Raises:
            None
        """
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MSBertSelfAttention(nn.Cell):
    """
    Self attention layer for BERT.
    """
    def __init__(self, config, causal, init_cache=False):
        """Initializes an instance of the MSBertSelfAttention class.
        
        Args:
            self: The instance of the class.
            config:
                A configuration object containing various parameters.

                - Type: Object
                - Purpose: Specifies the configuration parameters for the attention mechanism.
                - Restrictions: None

            causal:
                A boolean value indicating whether the attention mechanism is causal or not.

                - Type: bool
                - Purpose: Determines if the attention mechanism is restricted to attend to previous positions only.
                - Restrictions: None

            init_cache:
                A boolean value indicating whether to initialize the cache or not.

                - Type: bool
                - Purpose: Determines if the cache for attention weights and values should be initialized.
                - Restrictions: None

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads.

        Notes:
            - This method is called when creating an instance of the MSBertSelfAttention class.
            - The attention mechanism is responsible for computing self-attention weights and values based on the input.
            - The method initializes various instance variables and parameters required for the attention mechanism.
            - If the hidden size is not divisible by the number of attention heads, a ValueError is raised.
            - The method also initializes the cache variables if `init_cache` is True, otherwise sets them to None.
            - The method creates dense layers for query, key, and value projections.
            - The method initializes dropout and softmax layers for attention probabilities computation.
            - The method creates a causal mask if `causal` is True, otherwise uses a mask of ones.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )
        self.key = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )
        self.value = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(-1)

        self.causal = causal
        self.init_cache = init_cache

        self.causal_mask = make_causal_mask(
            ops.ones((1, config.max_position_embeddings), dtype=mstype.bool_),
            dtype=mstype.bool_,
        )

        if not init_cache:
            self.cache_key = None
            self.cache_value = None
            self.cache_index = None
        else:
            self.cache_key = Parameter(
                initializer(
                    "zeros",
                    (
                        config.max_length,
                        config.max_batch_size,
                        config.num_attention_heads,
                        config.attention_head_size,
                    ),
                )
            )
            self.cache_value = Parameter(
                initializer(
                    "zeros",
                    (
                        config.max_length,
                        config.max_batch_size,
                        config.num_attention_heads,
                        config.attention_head_size,
                    ),
                )
            )
            self.cache_index = Parameter(Tensor(0, mstype.int32))

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        Concatenates the given key, value, query, and attention mask to the cache in the MSBertSelfAttention class.

        Args:
            self (MSBertSelfAttention): An instance of the MSBertSelfAttention class.
            key (Tensor): The key tensor to be concatenated to the cache.
                Shape: (batch_size, num_updated_cache_vectors, hidden_size).
            value (Tensor): The value tensor to be concatenated to the cache.
                Shape: (batch_size, num_updated_cache_vectors, hidden_size).
            query (Tensor): The query tensor. Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor): The attention mask tensor. Shape: (batch_size, sequence_length).

        Returns:
            tuple: A tuple containing the updated key, value, and attention mask tensors.

        Raises:
            None.
        """
        if self.init_cache:
            batch_size = query.shape[0]
            num_updated_cache_vectors = query.shape[1]
            max_length = self.cache_key.shape[0]
            indices = ops.arange(
                self.cache_index, self.cache_index + num_updated_cache_vectors
            )
            key = ops.scatter_update(self.cache_key, indices, key.swapaxes(0, 1))
            value = ops.scatter_update(self.cache_value, indices, value.swapaxes(0, 1))

            self.cache_index += num_updated_cache_vectors

            pad_mask = ops.broadcast_to(
                ops.arange(max_length) < self.cache_index,
                (batch_size, 1, num_updated_cache_vectors, max_length),
            )
            attention_mask = ops.logical_and(attention_mask, pad_mask)

        return key, value, attention_mask

    def transpose_for_scores(self, input_x):
        r"""
        transpose for scores
        """
        new_x_shape = input_x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        input_x = input_x.view(*new_x_shape)
        return input_x.transpose(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        """
        Constructs the self-attention layer for the MSBert model.

        Args:
            self (MSBertSelfAttention): The instance of the MSBertSelfAttention class.
            hidden_states (Tensor):
                The input tensor of shape (batch_size, seq_length, hidden_size) representing the hidden states.
            attention_mask (Tensor, optional):
                The attention mask tensor of shape (batch_size, seq_length) or (batch_size, seq_length, seq_length)
                to mask out certain positions from the attention computation.
                Defaults to None.
            head_mask (Tensor, optional):
                The tensor of shape (num_attention_heads,) representing the mask for the attention heads.
                Defaults to None.

        Returns:
            outputs (tuple): A tuple containing the context layer tensor of shape (batch_size, seq_length, hidden_size)
                and the attention probabilities tensor of shape (batch_size, num_attention_heads, eq_length, seq_length)
                if self.output_attentions is True, else only the context layer tensor is returned.

        Raises:
            None.
        """
        batch_size = hidden_states.shape[0]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_states = self.transpose_for_scores(mixed_query_layer)
        key_states = self.transpose_for_scores(mixed_key_layer)
        value_states = self.transpose_for_scores(mixed_value_layer)

        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = ops.slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = ops.broadcast_to(
                causal_mask, (batch_size,) + causal_mask.shape[1:]
            )
        else:
            causal_mask = None

        if attention_mask is not None and self.causal:
            attention_mask = ops.broadcast_to(
                attention_mask.expand_dims(-2).expand_dims(-3), causal_mask.shape
            )
            attention_mask = ops.logical_and(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = attention_mask.expand_dims(-2).expand_dims(-3)

        if self.causal and self.init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            # attention_bias = ops.select(
            #     attention_mask > 0,
            #     ops.full(attention_mask.shape, 0.0).astype(hidden_states.dtype),
            #     ops.full(attention_mask.shape, finfo(hidden_states.dtype, "min")).astype(
            #         hidden_states.dtype
            #     ),
            # )
            attention_bias = ops.select(
                attention_mask > 0,
                ops.zeros_like(attention_mask).astype(hidden_states.dtype),
                (ops.ones_like(attention_mask) * finfo(hidden_states.dtype, "min")).astype(
                    hidden_states.dtype
                ),
            )
        else:
            attention_bias = None

        # Take the dot product between "query" snd "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_states, key_states.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(
            Tensor(self.attention_head_size, mstype.float32)
        )
        # Apply the attention mask is (precommputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_bias

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class MSBertSelfOutput(nn.Cell):
    r"""
    Bert Self Output
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertSelfOutput class.

        Args:
            self: The instance of the MSBertSelfOutput class.
            config: An object containing configuration parameters for the MSBertSelfOutput class.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required configuration parameters.
            RuntimeError: If there is an issue with initializing the dense, LayerNorm, or dropout attributes.
        """
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        This method 'construct' is a part of the 'MSBertSelfOutput' class and is responsible for
        processing the hidden states and input tensor.

        Args:
            self: The instance of the class.

            hidden_states (tensor): The hidden states to be processed. It is expected to be a tensor.

            input_tensor (tensor): The input tensor to be incorporated into the hidden states. It is expected to be a tensor.

        Returns:
            tensor: The processed hidden states with the input tensor incorporated.

        Raises:
            This method does not raise any exceptions.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MSBertAttention(nn.Cell):
    r"""
    Bert Attention
    """
    def __init__(self, config, causal, init_cache=False):
        """
        Initializes an instance of MSBertAttention.

        Args:
            self: The instance of the class itself.
            config (object): The configuration object containing various settings.
            causal (bool): Flag indicating whether the attention mechanism is causal.
            init_cache (bool, optional): Flag indicating whether to initialize cache. Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = MSBertSelfAttention(config, causal, init_cache)
        self.output = MSBertSelfOutput(config)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        """
        Constructs the attention mechanism for a multi-head self-attention layer in MSBertAttention.

        Args:
            self (MSBertAttention): The instance of the MSBertAttention class.
            hidden_states (torch.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the sequence of hidden states for each token in the input sequence.
            attention_mask (torch.Tensor, optional): An optional tensor of shape (batch_size, sequence_length) indicating
                which tokens should be attended to and which should be ignored. The value 1 indicates to attend to the token,
                while 0 indicates to ignore it. If not provided, all tokens are attended to.
            head_mask (torch.Tensor, optional): An optional tensor of shape (num_heads,) or (num_layers, num_heads) indicating
                which heads or layers to mask. 1 indicates to include the head/layer, while 0 indicates to mask it.
                If not provided, all heads/layers are included.

        Returns:
            Tuple[torch.Tensor]:
                A tuple containing:

                - attention_output (torch.Tensor): The output tensor of shape (batch_size, sequence_length, hidden_size),
                  which represents the attended hidden states for each token in the input sequence.
                - self_outputs[1:] (tuple): A tuple of length `num_layers` containing tensors representing intermediate
                  outputs of the self-attention mechanism.

        Raises:
            None.
        """
        self_outputs = self.self(hidden_states, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MSBertIntermediate(nn.Cell):
    r"""
    Bert Intermediate
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertIntermediate class.

        Args:
            self: The instance of the MSBertIntermediate class.
            config: An object representing the configuration for the MSBertIntermediate model.
                It contains the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str): The activation function for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the correct type.
            ValueError: If the config object does not contain the required attributes.
        """
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.intermediate_size,
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        """
        Constructs the intermediate layer of the MSBert model.

        Args:
            self: An instance of the MSBertIntermediate class.
            hidden_states (Tensor): The input hidden states.
                Should be a tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: The output hidden states after passing through the intermediate layer.
                Has the same shape as the input hidden states.

        Raises:
            None.

        This method takes in the input hidden states and applies the intermediate layer transformations.
        It first passes the hidden states through a dense layer, then applies an activation function.
        The resulting hidden states are returned as the output.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MSBertOutput(nn.Cell):
    r"""
    Bert Output
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertOutput class.

        Args:
            self: The instance of the class.
            config: An object of type 'config' that contains the configuration parameters for the MSBertOutput.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(
            config.intermediate_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        This method constructs the output of the MSBert model.

        Args:
            self: The instance of the MSBertOutput class.
            hidden_states (tensor): The hidden states from the MSBert model.
                This tensor contains the encoded information from the input.
            input_tensor (tensor): The input tensor to be added to the hidden states.
                This tensor represents the original input to the MSBert model.

        Returns:
            tensor: The constructed output tensor representing the final hidden states.
            This tensor is the result of processing the hidden states and input tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MSBertLayer(nn.Cell):
    r"""
    Bert Layer
    """
    def __init__(self, config, init_cache=False):
        """
        Initializes an instance of the MSBertLayer class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing various settings and parameters.
            init_cache (bool, optional): Whether to initialize the cache. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.attention = MSBertAttention(config, causal=config.is_decoder, init_cache=init_cache)
        self.intermediate = MSBertIntermediate(config)
        self.output = MSBertOutput(config)
        if config.add_cross_attention:
            self.crossattention = MSBertAttention(config, causal=False, init_cache=init_cache)

    def construct(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states = None,
                encoder_attention_mask = None):
        """
        Constructs the MSBertLayer.

        Args:
            self: The instance of the MSBertLayer class.
            hidden_states: The input hidden states (tensor) of shape (batch_size, sequence_length, hidden_size).
            attention_mask:
                Optional attention mask (tensor) of shape (batch_size, sequence_length) or (batch_size, 1, 1, sequence_length).
                Defaults to None.
            head_mask: Optional head mask (tensor) of shape (num_heads,) or (num_layers, num_heads). Defaults to None.
            encoder_hidden_states:
                Optional encoder hidden states (tensor) of shape (batch_size, sequence_length, hidden_size).
                Defaults to None.
            encoder_attention_mask:
                Optional encoder attention mask (tensor) of shape (batch_size, sequence_length) or (batch_size, 1, 1, sequence_length).
                Defaults to None.

        Returns:
            tuple:
                A tuple containing the layer output (tensor) of shape (batch_size, sequence_length, hidden_size)
                and any additional attention outputs.

        Raises:
            None.
        """
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
            )
            attention_output = cross_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class MSBertEncoder(nn.Cell):
    r"""
    Bert Encoder
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertEncoder class.

        Args:
            self (MSBertEncoder): The instance of the class itself.
            config:
                An object containing the configuration parameters for the MSBertEncoder.

                - output_attentions (bool): Whether to output attentions weights.
                - output_hidden_states (bool): Whether to output all hidden states.
                - layer (nn.CellList): List of MSBertLayer instances.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList(
            [MSBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def _set_recompute(self):
        """
        Sets the recompute flag for each layer in the MSBertEncoder.

        Args:
            self: An instance of the MSBertEncoder class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method iterates over each layer within the MSBertEncoder instance and sets the recompute flag
            for each layer by calling the 'recompute()' method of the layer.
            The recompute flag is used to indicate whether the layer needs to be recomputed during the
            forward pass of the encoder.
            By setting the recompute flag, it allows for dynamic computation of the layer based on the input.

        Example:
            ```python
            >>> encoder = MSBertEncoder()
            >>> encoder._set_recompute()
            ```

        Note:
            This method is typically called internally within the MSBertEncoder class
            and does not need to be called externally.
        """
        for layer in self.layer:
            layer.recompute()

    def construct(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states = None,
                encoder_attention_mask = None):
        """
        Constructs the MSBertEncoder.

        Args:
            self: An instance of the MSBertEncoder class.
            hidden_states (Tensor): The input hidden states of the encoder.
                Shape: (batch_size, sequence_length, hidden_size)
            attention_mask (Tensor, optional): The attention mask for the input hidden states.
                If provided, the attention mask should have the same shape as the hidden states.
                Each element of the mask should be 0 or 1, where 0 indicates the position is padded/invalid and 1
                indicates the position is not padded/valid.
                Defaults to None.
            head_mask (Tensor, optional): The head mask for the attention mechanism.
                If provided, the head mask should have the same shape as the number of layers in the encoder.
                Each element of the mask should be 0 or 1, where 0 indicates the head is masked and 1 indicates the head is not masked.
                Defaults to None.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder.
                Shape: (batch_size, sequence_length, hidden_size)
                Defaults to None.
            encoder_attention_mask (Tensor, optional): The attention mask for the encoder hidden states.
                If provided, the attention mask should have the same shape as the encoder hidden states.
                Each element of the mask should be 0 or 1, where 0 indicates the position is padded/invalid
                and 1 indicates the position is not padded/valid.
                Defaults to None.

        Returns:
            outputs (Tuple):
                A tuple containing the following elements:

                - hidden_states (Tensor): The output hidden states of the encoder.
                    Shape: (batch_size, sequence_length, hidden_size)
                - all_hidden_states (Tuple[Tensor]): A tuple of hidden states of all layers.
                    Each element of the tuple has the shape (batch_size, sequence_length, hidden_size).
                    This will be included if the 'output_hidden_states' flag is set to True.
                - all_attentions (Tuple[Tensor]): A tuple of attention scores of all layers.
                    Each element of the tuple has the shape (batch_size, num_heads, sequence_length, sequence_length).
                    This will be included if the 'output_attentions' flag is set to True.

        Raises:
            None.
        """
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask
                )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions += (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs += (all_hidden_states,)
        if self.output_attentions:
            outputs += (all_attentions,)
        return outputs


class MSBertPooler(nn.Cell):
    r"""
    Bert Pooler
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertPooler class.

        Args:
            self (MSBertPooler): The instance of the MSBertPooler class.
            config:
                An object containing configuration parameters.

                - Type: Any
                - Purpose: Holds the configuration settings for the MSBertPooler.
                - Restrictions: Must be compatible with the expected configuration format.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        """
        This method constructs a pooled output from the given hidden states.

        Args:
            self (MSBertPooler): The instance of the MSBertPooler class.
            hidden_states (torch.Tensor): A tensor containing the hidden states.
                It is expected to have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: The pooled output tensor obtained by applying dense
                and activation functions to the first token tensor from the hidden_states.

        Raises:
            TypeError: If the input hidden_states is not a torch.Tensor.
            ValueError: If the hidden_states tensor does not have the expected shape of
                (batch_size, sequence_length, hidden_size).
        """
        # We "pool" the model by simply taking the hidden state corresponding.
        # to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MSBertPredictionHeadTransform(nn.Cell):
    r"""
    Bert Prediction Head Transform
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertPredictionHeadTransform class.

        Args:
            self: An instance of the MSBertPredictionHeadTransform class.
            config: An object containing configuration settings for the transformation.
                It is expected to have the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - hidden_act (str): The activation function to be used for the hidden layer.
                - layer_norm_eps (float): The epsilon value for LayerNorm.

        Returns:
            None: This method initializes the dense layer, activation function, and LayerNorm parameters for the transformation.

        Raises:
            TypeError: If the config parameter is not provided.
            ValueError: If the config parameter is missing any required attributes.
            KeyError: If the hidden activation function specified in the config is not found in the ACT2FN dictionary.
        """
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )

    def construct(self, hidden_states):
        """
        This method 'construct' is part of the 'MSBertPredictionHeadTransform' class and is used to perform transformations on hidden states.

        Args:
            self:
                The instance of the 'MSBertPredictionHeadTransform' class.

                - Type: MSBertPredictionHeadTransform
                - Purpose: Represents the current instance of the class.
                - Restrictions: None

            hidden_states:
                The input hidden states that need to undergo transformations.

                - Type: Any
                - Purpose: Represents the hidden states to be processed.
                - Restrictions: Should be compatible with the operations performed within the method.

        Returns:
            hidden_states:

                - Type: None
                - Purpose: To return the processed hidden states for further usage.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MSBertLMPredictionHead(nn.Cell):
    r"""
    Bert LM Prediction Head
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSBertLMPredictionHead class.

        Args:
            self: The object instance.
            config:
                An instance of the configuration class that contains the model's configuration settings.

                - Type: Any
                - Purpose: This parameter is used to configure the MSBertLMPredictionHead instance.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.transform = MSBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(
            config.hidden_size,
            config.vocab_size,
            has_bias=False,
        )

        self.bias = Parameter(initializer("zeros", config.vocab_size), "bias")

    def construct(self, hidden_states, masked_lm_positions):
        """
        Constructs the MSBertLMPredictionHead.

        This method takes in the hidden states and masked language model positions,
        and applies a series of operations to compute the final hidden states for the MSBertLMPredictionHead.
        The resulting hidden states are then transformed and decoded to produce the final output.

        Args:
            self (MSBertLMPredictionHead): An instance of the MSBertLMPredictionHead class.
            hidden_states (Tensor): A tensor of shape (batch_size, seq_len, hidden_size) containing the hidden states.
            masked_lm_positions (Tensor): A tensor of shape (batch_size, num_masked_lm_positions)
                containing the positions of the masked language model tokens. If None, no masking is applied.

        Returns:
            Tensor:
                A tensor of shape (batch_size, seq_len, hidden_size) containing
                the final hidden states for the MSBertLMPredictionHead.

        Raises:
            None.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        if masked_lm_positions is not None:
            flat_offsets = ops.arange(batch_size) * seq_len
            flat_position = (masked_lm_positions + flat_offsets.reshape(-1, 1)).reshape(
                -1
            )
            flat_sequence_tensor = hidden_states.reshape(-1, hidden_size)
            hidden_states = ops.gather(flat_sequence_tensor, flat_position, 0)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class MSBertPreTrainingHeads(nn.Cell):
    r"""
    Bert PreTraining Heads
    """
    def __init__(self, config):
        """
        Initialize the MSBertPreTrainingHeads class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration settings.

                - Type: Custom class
                - Purpose: Provides configuration parameters for the pre-training heads.
                - Restrictions: Must be compatible with the MSBertLMPredictionHead and nn.Dense classes.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = MSBertLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output, masked_lm_positions):
        """
        Construct method in the MSBertPreTrainingHeads class.

        Args:
            self (object): The instance of the class.
            sequence_output (tensor): The output tensor from the pre-trained model for the input sequence.
            pooled_output (tensor): The output tensor obtained by applying pooling to the sequence_output.
            masked_lm_positions (tensor): The positions of the masked language model tokens in the input sequence.

        Returns:
            Tuple: A tuple containing the prediction_scores (tensor) and seq_relationship_score (tensor)
                calculated based on the inputs.

        Raises:
            None: This method does not raise any exceptions.
        """
        prediction_scores = self.predictions(sequence_output, masked_lm_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MSBertPreTrainedModel(PreTrainedModel):
    """BertPretrainedModel"""
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_recompute = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(
                initializer(
                    Normal(self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.has_bias:
                cell.bias.set_data(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            weight = initializer(
                Normal(self.config.initializer_range),
                cell.weight.shape,
                cell.weight.dtype,
            )
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer("ones", cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))


class MSBertModel(MSBertPreTrainedModel):
    r"""
    Bert Model
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes the MSBertModel class with the provided configuration and optional pooling layer.

        Args:
            self (MSBertModel): The current instance of the MSBertModel class.
            config (object): The configuration object containing settings for the model.
            add_pooling_layer (bool): Flag indicating whether to add a pooling layer to the model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.embeddings = MSBertEmbeddings(config)
        self.encoder = MSBertEncoder(config)
        self.pooler = MSBertPooler(config) if add_pooling_layer else None
        self.num_hidden_layers = config.num_hidden_layers

    def get_input_embeddings(self):
        """
        This method returns the input embeddings of the MSBertModel.

        Args:
            self: The instance of the MSBertModel class.

        Returns:
            word_embeddings:
                This method returns the input embeddings of the MSBertModel.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the MSBertModel.

        Args:
            self (MSBertModel): The MSBertModel instance.
            new_embeddings (object): The new input embeddings to be set. This could be of any type, such as a tensor or an array.

        Returns:
            None.

        Raises:
            None
        """
        self.embeddings.word_embeddings = new_embeddings

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states = None,
        encoder_attention_mask = None
    ):
        """
        Construct method in the MSBertModel class.

        Args:
            self: MSBertModel object.
            input_ids (Tensor): The input tensor containing the token ids for the input sequence.
            attention_mask (Tensor, optional):
                A mask tensor indicating which tokens should be attended to and which should be ignored.
            token_type_ids (Tensor, optional): A tensor indicating the token types for each token in the input sequence.
            position_ids (Tensor, optional): A tensor specifying the position ids for each token in the input sequence.
            head_mask (Tensor, optional): A mask tensor applied to the attention scores in the self-attention mechanism.
            encoder_hidden_states (Tensor, optional): Hidden states from the encoder.
            encoder_attention_mask (Tensor, optional): A mask tensor indicating which encoder tokens should be attended to in the self-attention mechanism.

        Returns:
            Tuple:
                A tuple containing the following:

                - sequence_output (Tensor): The output tensor from the encoder for each token in the input sequence.
                - pooled_output (Tensor): The pooled output tensor from the pooler layer, if available.
                - Additional encoder outputs.

        Raises:
            ValueError: If the dimensions of the head_mask tensor are incompatible.
        """
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)
        if position_ids is None:
            position_ids = ops.broadcast_to(ops.arange(ops.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = (
                    head_mask.expand_dims(0)
                    .expand_dims(0)
                    .expand_dims(-1)
                    .expand_dims(-1)
                )
                head_mask = ops.broadcast_to(
                    head_mask, (self.num_hidden_layers, -1, -1, -1, -1)
                )
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class MSBertForPretraining(MSBertPreTrainedModel):
    r"""
    Bert For Pretraining
    """
    def __init__(self, config, *args, **kwargs):
        """
        __init__

        Initialize the MSBertForPretraining class.

        Args:
            self: The instance of the MSBertForPretraining class.
            config: The configuration for the MSBertForPretraining,
                containing various parameters and settings for model initialization.
                It should be an instance of the configuration class specific to the MSBertForPretraining model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config, *args, **kwargs)
        self.bert = MSBertModel(config)
        self.cls = MSBertPreTrainingHeads(config)
        self.vocab_size = config.vocab_size

        self.cls.predictions.decoder.weight = (
            self.bert.embeddings.word_embeddings.weight
        )

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_positions=None,
    ):
        """
        This method constructs the pretraining model for MSBertForPretraining.

        Args:
            self (MSBertForPretraining): The instance of the MSBertForPretraining class.
            input_ids (Tensor): The input tensor containing the token ids.
            attention_mask (Tensor, optional): A tensor representing the attention mask. Default is None.
            token_type_ids (Tensor, optional): A tensor representing the token type ids. Default is None.
            position_ids (Tensor, optional): A tensor representing the position ids. Default is None.
            head_mask (Tensor, optional): A tensor representing the head mask. Default is None.
            masked_lm_positions (List[int]): A list of integer positions of masked language model tokens.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the prediction scores and sequence relationship score.

        Raises:
            None
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_lm_positions
        )

        outputs = (
            prediction_scores,
            seq_relationship_score,
        ) + outputs[2:]
        # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

        return outputs


class MSBertForSequenceClassification(MSBertPreTrainedModel):
    """Bert Model for classification tasks"""
    def __init__(self, config):
        """
        Initializes an instance of the MSBertForSequenceClassification class.

        Args:
            self: The instance of the class.
            config (object): A configuration object containing the settings for the model.
                It should include the following attributes:

                - num_labels (int): The number of labels for sequence classification.
                - classifier_dropout (float, optional): The dropout probability for the classifier layer.
                If not provided, the value will default to config.hidden_dropout_prob.
        
        Returns:
            None: This method initializes the instance with the provided configuration.
        
        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            ValueError: If the num_labels attribute is not present in the config object.
            AttributeError: If the config object does not contain the necessary attributes for model configuration.
            RuntimeError: If an error occurs during model initialization.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = MSBertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.classifier = nn.Dense(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(p=classifier_dropout)

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        **kwargs
    ):
        """
        Constructs the MSBertForSequenceClassification model for a given input.
        
        Args:
            self (MSBertForSequenceClassification): The instance of the MSBertForSequenceClassification class.
            input_ids (Tensor): The input tensor containing the indices of input tokens.
            attention_mask (Tensor, optional): An optional tensor containing the attention mask for the input.
            token_type_ids (Tensor, optional): An optional tensor containing the token type ids.
            position_ids (Tensor, optional): An optional tensor containing the position ids.
            head_mask (Tensor, optional): An optional tensor containing the head mask.
            
        Returns:
            tuple: A tuple containing the logits for the classification and additional outputs from the model.
        
        Raises:
            None
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]

        return output


__all__ = [
    "MSBertEmbeddings",
    "MSBertAttention",
    "MSBertEncoder",
    "MSBertIntermediate",
    "MSBertLayer",
    "MSBertModel",
    "MSBertForPretraining",
    "MSBertLMPredictionHead",
    "MSBertForSequenceClassification",
]
