""" MindSpore MegatronBERT model."""
# pylint: disable=too-many-lines
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=import-outside-toplevel
# pylint: disable=arguments-differ
# pylint: disable=logging-fstring-interpolation
# pylint: disable=relative-beyond-top-level
# pylint: disable=no-else-raise


from typing import Tuple, Union
import os
import mindspore
from mindspore import nn, ops, Tensor, numpy, log as logger
from mindspore.common.initializer import Normal, initializer
from mindspore.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from mindnlp.abc import PreTrainedModel
from mindnlp.models.utils.utils import find_pruneable_heads_and_indices, prune_linear_layer, \
    apply_chunking_to_forward
from mindnlp.models.utils.activations import ACT2FN
from .megatron_bert_config import MegatronBertConfig


def load_tf_weights_in_megatron_bert(model, tf_checkpoint_path):
    """Load tf checkpoints in a mindspore model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in MindSpore, requires TensorFlow to be installed. "
            "Please see https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1",
                     "global_step"] for n in name):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
            if m_name[-11:] == "_embeddings":
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
        if pointer.shape != array.shape:
            raise ValueError(
                f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        logger.info(f"Initialize MindSpore weight {name}")
        pointer.data = Tensor.from_numpy(array)
    return model


class MegatronBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and
        # be able to load any TensorFlow checkpoint file

        # In Megatron, layer-norm is applied after the 1st dropout.
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def construct(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0
    ) -> Tensor:
        """ The forward algorithm for the embedding layer. """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, mindspore.int32)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Megatron BERT moves that layer norm after the drop-out (and to each layer).
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MegatronBertSelfAttention(nn.Cell):
    """ MegatronBert SelfAttention layer """

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of "
                f"the number of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, matrix_x: Tensor) -> Tensor:
        """Transpose the score Tensor"""
        new_x_shape = matrix_x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        matrix_x = matrix_x.view(new_x_shape)
        return matrix_x.permute(0, 2, 1, 3)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False
    ) -> Tuple[Tensor]:
        """Forward algorithms for the self-attention layer"""
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = ops.concat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder)
            # save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states.
            # Further calls to uni-directional self-attention can concat previous decoder
            # key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        if self.position_embedding_type == ("relative_key", "relative_key_query"):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = Tensor(key_length - 1, dtype=mindspore.int64).view(-1, 1)
            else:
                position_ids_l = mindspore.numpy.arange(query_length, dtype=Tensor.long).view(-1, 1)
            position_ids_r = mindspore.numpy.arange(key_length, dtype=Tensor.long).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            # fp16 compatibility
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = ops.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / ops.sqrt(Tensor(self.attention_head_size, mindspore.float32))
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MegatronBertModel
            # forward() function)
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

        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class MegatronBertSelfOutput(nn.Cell):
    """
    Based transformers.models.bert.modeling_bert.BertSelfOutput.
    Moved LayerNorm to MegatronBertAttention below.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, residual: Tensor) -> Tensor:
        """Construct algorithms for the self-output layer"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class MegatronBertAttention(nn.Cell):
    """ MegatronBert Attention layer """

    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.self = MegatronBertSelfAttention(config)
        self.output = MegatronBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prune the Attention heads.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads
        )
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ) -> Tuple[Tensor]:
        ln_outputs = Tensor(self.layer_norm(hidden_states))
        self_outputs = self.self(
            ln_outputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->MegatronBert
class MegatronBertIntermediate(nn.Cell):
    """ MegatronBert Intermediate layer """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MegatronBertOutput(nn.Cell):
    """
    Based on transformers.models.bert.modeling_bert.BertOutput.
    Moved LayerNorm to MegatronBertLayer below.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return input_tensor + hidden_states


class MegatronBertLayer(nn.Cell):
    """
    Based on transformers.models.bert.modeling_bert.BertLayer. Added LayerNorm.
    """

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MegatronBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model "
                                f"if cross attention is added")
            self.crossattention = MegatronBertAttention(config)
        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.intermediate = MegatronBertIntermediate(config)
        self.output = MegatronBertOutput(config)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ) -> Tuple[Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        present_key_value = None
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # add self attentions if we output attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    f"layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        The feed_forward_chunk function takes the input tensor
        (usually the output tensor of the self-attention Layer)
        and performs a non-linear transformation of the input through two fully connected layers,
        then adjusts and normalizes the transformation results
        using residual connection and Layer Normalization.
        Output the fully connected layer.
        """
        ln_output = self.layer_norm(attention_output)
        intermediate_output = self.intermediate(ln_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MegatronBertEncoder(nn.Cell):
    """
    The main function of MegatronBertEncoder is to map the input sequence
    into a high-dimensional spaceand extract task-related feature information.
    It is able to automatically learn how to encode the input during training
    and capture the semantic information in the input sequence
    by stacking multilayer neural networks.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([MegatronBertLayer(config) for _ in range(config.num_hidden_layers)])

        # The final layer norm. We removed the 1st LN, moved LN to each hidden layer and this one
        # is simply the final LN (Transformer's BERT has it attached to each hidden layer).
        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ) -> Union[Tuple, Tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            # goto else branch
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         return module(*inputs, past_key_value, output_attentions)
                #
                #     return custom_forward

                # layer_outputs = torch.utils.checkpoint.checkpoint(
                #     create_custom_forward(layer_module),
                #     hidden_states,
                #     attention_mask,
                #     layer_head_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                # )
                layer_outputs = ()  # this definition is for pass pylint check
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # Because we moved the layer-norm at the end of the hidden layer, we have
            # non-normalized data here. If that's really needed, we must apply LN to
            # match Transformer's BERT.

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Finalize the hidden states.
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ] if v is not None
            )

        return (hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions)


class MegatronBertPooler(nn.Cell):
    """
        Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->MegatronBert
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MegatronBertPredictionHeadTransform(nn.Cell):
    """
    Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform
    with Bert->MegatronBert
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.ln_2 = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.ln_2(hidden_states)
        return hidden_states


class MegatronBertLMPredictionHead(nn.Cell):
    """
    Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->MegatronBert
    """

    def __init__(self, config):
        super().__init__()
        self.transform = MegatronBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly
        # resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MegatronBertOnlyMLMHead(nn.Cell):
    """
    Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->MegatronBert
    """

    def __init__(self, config):
        super().__init__()
        self.predictions = MegatronBertLMPredictionHead(config)

    def construct(self, sequence_output: Tensor) -> Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MegatronBertOnlyNSPHead(nn.Cell):
    """
    Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->MegatronBert
    """

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class MegatronBertPreTrainingHeads(nn.Cell):
    """
    Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->MegatronBert
    """

    def __init__(self, config):
        super().__init__()
        self.predictions = MegatronBertLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MegatronBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for
    downloading and loading pretrained models.
    """

    config_class = MegatronBertConfig
    load_tf_weights = load_tf_weights_in_megatron_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the embedding_tables"""
        if isinstance(module, nn.Embedding):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization https://github.com/pytorch/pytorch/pull/5617
            module.embedding_table = initializer(Normal(mean=0.0, sigma=self.config.initializer_range),
                                                 shape=module.embedding_table.shape)
        elif isinstance(module, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization https://github.com/pytorch/pytorch/pull/5617
            module.weight = initializer(Normal(mean=0.0, sigma=self.config.initializer_range),
                                        shape=module.weight.shape)
            # module.weight.data.normal_(mean=0.0, sigma=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            zeroslike = ops.ZerosLike()
            module.beta = zeroslike(Tensor(module.beta.data))

            fill = ops.Fill()
            fill(mindspore.float32, module.gamma.shape, 1.0)
        if isinstance(module, nn.Dense) and module.bias is not None:
            zeroslike = ops.ZerosLike()
            module.bias = zeroslike(module.bias.data)

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Optimizing the memory usage during training of large neural networks.
        """
        if isinstance(module, MegatronBertEncoder):
            module.gradient_checkpointing = value

    def get_position_embeddings(self):
        """
        rewrite abstract method
        """

    def init_model_weights(self):
        """
        rewrite abstract method
        """

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        rewrite abstract method
        """

    def save(self, save_dir: Union[str, os.PathLike]):
        """
        rewrite abstract method
        """

    def set_input_embeddings(self, new_embeddings: "nn.Cell"):
        """
        rewrite abstract method
        """

    def post_init(self):
        """
        rewrite abstract method
        """

    def get_input_embeddings(self):
        """
        rewrite abstract method
        """


class MegatronBertForPreTrainingOutput(nn.Cell):
    """
    Output type of [`MegatronBertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `mindspore.float32` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence
            prediction (classification) loss.
        prediction_logits (`mindspore.float32` of shape `(batch_size, sequence_length,
            config.vocab_size)`): Prediction scores of the language modeling head (scores for each
            vocabulary token before SoftMax).
        seq_relationship_logits (`mindspore.float32` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of
            True/False continuation before SoftMax).
        hidden_states (`tuple(mindspore.float32)`, *optional*, returned when
            `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.float32` (one for the output of the embeddings + one for the
            output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding
            outputs.
        attentions (`tuple(mindspore.float32)`, *optional*, returned when `output_attentions=True`
            is passed or when `config.output_attentions=True`): Tuple of `mindspore.float32` (one
            for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.
    """

    loss = None
    prediction_logits = None
    seq_relationship_logits = None
    hidden_states = None
    attentions = None

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class MegatronBertModel(MegatronBertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder,
    in which case a layer of cross-attention is added between the self-attention layers,
    following the architecture described in [Attention is all you need]
    (https://arxiv.org/abs/1706.03762)
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
    Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of
    the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized
    with both `is_decoder` argument and `add_cross_attention` set to `True`;
    an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = MegatronBertEmbeddings(config)
        self.encoder = MegatronBertEncoder(config)

        self.pooler = MegatronBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        get word-embeddings
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        set word-embeddings
        """
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of
        {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        encoder_hidden_states  (`mindspore.float32` of shape `(batch_size, sequence_length,
            hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if the model is configured as a decoder.
        encoder_attention_mask (`mindspore.float32` of shape `(batch_size, sequence_length)`,
            *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(mindspore.float32))` of length `config.n_layers` with each
        tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1,
            embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last
            `decoder_input_ids` (those that don't have their past key value states given
            to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids`
            of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be
            used to speed up decoding (see `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, seq_length + past_key_values_length))
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length,
        # to_seq_length], ourselves in which case we just need to make it broadcastable
        # to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] and
        # head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return (
            sequence_output,
            pooled_output,
            encoder_outputs.past_key_values,
            encoder_outputs.hidden_states,
            encoder_outputs.attentions,
            encoder_outputs.cross_attentions)


class MegatronBertForPreTraining(MegatronBertPreTrainedModel):
    """
    Its function is to fine-tune the various natural language processing tasks
    by pre-training and learning the context of the language.
    """
    _keys_to_ignore_on_load_missing = ["cls.predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        self.cls = MegatronBertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        get output embeddings
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        set output embeddings
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        labels (`mindspore.int32` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with
            indices set to `-100` are ignored (masked), the loss is only computed for the
            tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`mindspore.int32` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss.
            Input should be a sequence pair (see `input_ids` docstring) Indices should
            be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return (
            total_loss,
            prediction_scores,
            seq_relationship_score,
            outputs.hidden_states,
            outputs.attentions,
        )


class MegatronBertForCausalLM(MegatronBertPreTrainedModel):
    """
    causal lm
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"cls.predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `MegatronBertForCausalLM` as a standalone, add `is_decoder=True.`")

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.cls = MegatronBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        get output embeddings
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        set output embeddings
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        encoder_hidden_states  (`mindspore.float32` of shape
            `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if the model is configured as a decoder.
        encoder_attention_mask (`mindspore.float32` of shape
            `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`mindspore.int32` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring)
            Tokens with indices set to `-100` are ignored (masked), the loss is only computed for
            the tokens with labels n `[0, ..., config.vocab_size]` past_key_values
            (`tuple(tuple(mindspore.float32))` of length `config.n_layers` with each tuple having 4
            tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last
            `decoder_input_ids` (those that don't have their past key value states given to this
            model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to
            speed up decoding (see `past_key_values`).

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return (
            lm_loss,
            prediction_scores,
            outputs.past_key_values,
            outputs.hidden_states,
            outputs.attentions,
            outputs.cross_attentions,
        )

    # def prepare_inputs_for_generation(
    #         self,
    #         input_ids,
    #         past_key_values=None,
    #         attention_mask=None,
    # ):
    #     """
    #     prepare_inputs_for_generation is used for generating sequences of text using the model.
    #
    #     The method creates a decoder_input_ids tensor by shifting the input_ids tensor one
    #     position to the right.
    #     This is done to prepare the decoder input for autoregressive generation, where the model
    #     generates one token at a time based on the previously generated tokens.
    #
    #     inputs:
    #         input_ids ('Tensor'):
    #             A tensor of token ids representing the input sequence.
    #         attention_mask ('Tensor'):
    #             A tensor that indicates which tokens should be attended to and which should not.
    #             This tensor is used to mask out padding tokens.
    #         past_key_values ('tuple'):
    #             A tuple of past key-value states for the Transformer blocks. The past states are
    #             cached states from the previous forward pass that are used to speed up generation.
    #         **model_kwargs:
    #             Additional arguments that are passed to the MegatronBertForCausalLM model.
    #     return:
    #         input_ids (Tensor):
    #             The input_ids tensor.
    #         attention_mask (Tensor):
    #             The attention_mask tensor.
    #         decoder_input_ids (Tensor):
    #             The decoder_input_ids tensor.
    #         past_key_values ('tuple'):
    #             The past key-value states for the Transformer blocks.
    #     """
    #     input_shape = input_ids.shape
    #     # if model is used as a decoder in encoder-decoder model,
    #     # the decoder attention mask is created on the fly
    #     if attention_mask is None:
    #         attention_mask = input_ids.new_ones(input_shape)
    #
    #     # cut decoder_input_ids if past is used
    #     if past_key_values is not None:
    #         input_ids = input_ids[:, -1:]
    #
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "past_key_values": past_key_values
    #     }

    # def reorder_cache(self, past, beam_idx):
    #     """
    #     This method is specifically used in the context of causal language models, which are
    #     language models that generate text sequentially in a left-to-right or right-to-left
    #     manner, such as autoregressive models like GPT-2 or GPT-3.
    #     """
    #     reordered_past = ()
    #     for layer_past in past:
    #         reordered_past += (tuple(past_state.index_select(0, beam_idx) \
    #                                  for past_state in layer_past),)
    #     return reordered_past


class MegatronBertForMaskedLM(MegatronBertPreTrainedModel):
    """
    It consists of multiple Transformer encoder layers, each containing a multi-head self-attention
    mechanism and a feed-forward neural network, as well as other operations such as normalization
    and residual connection.
    In masked language models, randomly selected words from the input text sequence are masked
    (i.e., replaced with special [MASK] tokens) and the model is asked to predict these masked
    words based on the context of other words.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"seq_relationship"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MegatronBertForMaskedLM` make sure `config.is_decoder=False` "
                "for bi-directional self-attention.")

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.cls = MegatronBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        get output embeddings
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        set output embeddings
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        labels (`mindspore.int32` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with
            indices set to `-100` are ignored (masked), the loss is only computed for the
            tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return (
            masked_lm_loss,
            prediction_scores,
            outputs.hidden_states,
            outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None):
        """
        'prepare_inputs_for_generation' is used for generating sequences of text using the model.

        inputs:

            input_ids ('tensor'):
                A tensor of token ids representing the input sequence.
            attention_mask ('tensor'):
                A tensor that indicates which tokens should be attended to and which should not.
                This tensor is used to mask out padding tokens.
            decoder_input_ids ('tensor'):
                A tensor of token ids representing the initial input for the decoder. This tensor
                is used to specify the initial context for generation.
            use_cache ('boolean'):
                indicates whether to use cached states from previous forward passes.
            **model_kwargs:
                Additional arguments that are passed to the MegatronBertForMaskedLM model.

        The method first checks whether the decoder_input_ids tensor is provided.
        If it is not provided, it sets decoder_input_ids to input_ids.
        This is done to ensure that the model starts generating from the beginning
        of the input sequence.

        Next, the method creates a masked_positions tensor by finding the positions of the
        [MASK] tokens in the input_ids tensor.
        This tensor is used to replace the [MASK] tokens with generated tokens during generation.

        returns:

        input_ids: The input_ids tensor.

        attention_mask: The attention_mask tensor.

        It is an important method for preparing input data for sequence generation using the
        MegatronBertForMaskedLM model. It takes care of finding the positions of [MASK] tokens and
        preparing the initial context for generation, which are essential for generating coherent
        sequences of text.
        """
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
        attention_mask = ops.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], axis=-1)
        dummy_token = numpy.full((effective_batch_size, 1), self.config.pad_token_id, dtype=Tensor.long)
        input_ids = ops.cat([input_ids, dummy_token], axis=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MegatronBertForNextSentencePrediction(MegatronBertPreTrainedModel):
    """
    Input Text sentence Pair:
        The input to the model is a sentence pair consisting of two text sentences.
        Among them, one sentence is used as "above" and the other sentence is used as "below".
        These two sentences can be randomly selected from a large-scale unlabeled corpus.

    Special tokens ([CLS] and [SEP]) :
        When processing an input text sentence pair, special tokens ([CLS] and [SEP]) need
        to be added between the two sentences to indicate the beginning and end of the
        input to the model.
        Specifically, the beginning of the input sequence is added with a [CLS] token,
        the middle is added with a [SEP] token to separate the two sentences, and the end
        is also added with a [SEP] token.
    """
    _keys_to_ignore_on_load_unexpected = [r"predictions"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        self.cls = MegatronBertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # **kwargs
    ) -> Union[Tuple, Tuple]:
        r"""
        labels (`mindspore.int32` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss.
            Input should be a sequence pair (see `input_ids` docstring).

            Indices should be in `[0, 1]`:
            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:


        """

        # if "next_sentence_label" in kwargs:
        #     warnings.warn(
        #         "The `next_sentence_label` argument is deprecated and will be removed in a future
        #         version, use `labels` instead.",
        #         FutureWarning,
        #     )
        #     labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return (
            next_sentence_loss,
            seq_relationship_scores,
            outputs.hidden_states,
            outputs.attentions,
        )


class MegatronBertForSequenceClassification(MegatronBertPreTrainedModel):
    """
    The input is a sequence of text, and the output is a vector of size the number of classes,
    representing the probability that the text sequence corresponds to each class.
    The loss function of the model usually uses the cross-entropy loss function,
    which is used to minimize the gap between the predicted result and the true label.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MegatronBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        labels (`mindspore.int32` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (Tensor.long, Tensor.int)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (
            loss,
            logits,
            outputs.hidden_states,
            outputs.attentions,
        )


class MegatronBertForMultipleChoice(MegatronBertPreTrainedModel):
    """
    In particular, MegatronBertForMultipleChoice model contains the following main components:

    MegatronBertModel:
        A Pre-trained Language Model based on a Transformer Architecture for learning
        representations of sentences. In a multiple-choice task, the model can take each question
        and the corresponding multiple options as input and generate the corresponding context
        representation.

    Multiple choice classifier:
        A linear classifier used to map context representations to option labels.
        In a multiple-choice task, this classifier receives as input a contextual representation
        from the MegatronBertModel and outputs a score corresponding to each option, with the model
        selecting the option with the highest score as the best answer.

    Dropout layer:
        A layer to prevent overfitting by randomly setting the output of a subset of neurons to 0.
        In multiple choice tasks, this layer can help improve the generalization ability and
        robustness of the model.

    MegatronBertForMultipleChoice input is a problem of multiple options and multiple choice set,
    the output is a vector size for the number of options, said scores of each option.
    The loss function of the model usually uses the cross-entropy loss function, which is used
    to minimize the gap between the predicted result and the true label.

    It is important to note that MegatronBertForMultipleChoice model is on the basis of the
    training model for fine-tuning, therefore need to be done on the basis of the training model
    of fine - tuning. Specifically, we can use annotated multiple-choice datasets to fine-tune
    this model so that it can better adapt to new multiple-choice tasks.
    """

    def __init__(self, config):
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        labels (`mindspore.int32` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss.
            Indices should be in `[0, ..., num_choices-1]` where `num_choices`
            is the size of the second dimension of the input tensors.
            (See `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (inputs_embeds.view(-1, inputs_embeds.size(-2),
                                            inputs_embeds.size(-1)) if inputs_embeds is not None else None)

        outputs = self.bert(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (
            loss,
            reshaped_logits,
            outputs.hidden_states,
            outputs.attentions,
        )


class MegatronBertForTokenClassification(MegatronBertPreTrainedModel):
    """
    The 'MegatronBertForTokenClassification' class is used for token classification tasks.
    In token classification tasks, the model needs to classify each token in a given sequence of
    text, typically used for tasks such as named entity recognition.

    The MegatronBertForTokenClassification model consists of the following components:

    MegatronBertModel:
        A pre-trained language model based on the Transformer architecture that learns the
        representation of the input text. In token classification tasks, this model encodes the
        input text sequence and generates the corresponding contextual representation.

    Token classifier:
        A linear classifier that maps the contextual representation generated by the
        'MegatronBertModel' to the token label.
        In token classification tasks, this classifier takes the contextual representation as
        input from the 'MegatronBertModel' and outputs scores for each token label.
        The model then selects the token label with the highest score as the prediction.

    Dropout layer:
        A layer that randomly sets a fraction of the neuron outputs to zero, which helps prevent
        overfitting. In token classification tasks, this layer can help improve the model's
        generalization ability and robustness.

    The input to 'MegatronBertForTokenClassification' is a tensor consisting of the text sequence,
    and the output is a tensor of size equal to the number of tokens, representing the score for
    each token. The model's loss function typically uses the cross-entropy loss function, which
    minimizes the difference between the predicted results and the ground truth labels.

    MegatronBertForTokenClassification is fine-tuned on top of a pre-trained model, and thus
    requires fine-tuning using annotated token classification datasets to better adapt to new
    token classification tasks.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        labels (`mindspore.int32` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in
            `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (
            loss,
            logits,
            outputs.hidden_states,
            outputs.attentions,
        )


class MegatronBertForQuestionAnswering(MegatronBertPreTrainedModel):
    """
    The MegatronBertForQuestionAnswering class is used for question-answering tasks.
    In a question-answering task, the model is given a question and a context,
    and the goal is to extract the answer from the context.

    The MegatronBertForQuestionAnswering model consists of the following components:

    MegatronBertModel:
        A pre-trained language model based on the Transformer architecture that learns the
        representation of the input text. In a question-answering task, this model encodes both
        the question and the context and generates the corresponding contextual representation.

    Start and End token classifiers:
        Two linear classifiers that map the contextual representation generated by the M
        egatronBertModel to the start and end positions of the answer in the context.
        In a question-answering task, these classifiers take the contextual representation as
        input from the MegatronBertModel and output scores for each position in the context.
        The model then selects the start and end positions with the highest scores as the answer
        span.

    Dropout layer:
        A layer that randomly sets a fraction of the neuron outputs to zero, which helps prevent
        overfitting. In a question-answering task, this layer can help improve the model's
        generalization ability and robustness.

    input:
        a tensor consisting of the question and the context,
    output:
        a tuple consisting of two tensors, one for the start position and one for the end position
        of the answer span.

    During training, the model is fine-tuned on top of a pre-trained model using a loss function
    such as the cross-entropy loss function. The loss function measures the difference between the
    predicted start and end positions and the ground truth start and end positions of the answer
    span.

    It is important to note that MegatronBertForQuestionAnswering is fine-tuned on top of a
    pre-trained model, and thus requires fine-tuning using annotated question-answering datasets
    to better adapt to new question-answering tasks.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple, Tuple]:
        r"""
        start_positions (`mindspore.int32` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token
            classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (`mindspore.int32` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token
            classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return (
            total_loss,
            start_logits,
            end_logits,
            outputs.hidden_states,
            outputs.attentions,
        )
