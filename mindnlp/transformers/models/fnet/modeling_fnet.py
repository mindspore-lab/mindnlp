# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
"""FNet model."""

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import initializer, Normal
from ....utils import is_scipy_available
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward
from ....utils import (
    logging,
)
from .configuration_fnet import FNetConfig

if is_scipy_available():
    from scipy import linalg

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/fnet-base"
_CONFIG_FOR_DOC = "FNetConfig"


# Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    """Applies 2D matrix multiplication to 3D input arrays."""
    seq_length = x.shape[1]
    matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
    x = x.type(ms.complex64)
    return ops.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)


# # Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two)


# Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def fftn(x):
    """
    Applies n-dimensional Fast Fourier Transform (FFT) to input array.

    Args:
        x: Input n-dimensional array.

    Returns:
        n-dimensional Fourier transform of input n-dimensional array.
    """
    out = x
    for axis in reversed(range(x.ndim)[1:]):  # We don't need to apply FFT to last axis
        out = ops.fft(out, dim=axis)
    return out


class FNetEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )
        # NOTE: This is the project layer and will be needed. The original code allows for different embedding and different model dimensions.
        self.projection = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to(
            (1, -1)
        )

        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=ms.int64)

    def construct(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
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
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FNetBasicFourierTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self._init_fourier_transform(config)

    def _init_fourier_transform(self, config):
        if not config.use_tpu_fourier_optimizations:
            self.fourier_transform = partial(ops.fftn, dim=(1, 2))
        elif config.max_position_embeddings <= 4096:
            if is_scipy_available():
                self.dft_mat_hidden = ms.tensor(
                    linalg.dft(config.hidden_size), dtype=ms.complex64
                )
                self.dft_mat_seq = ms.tensor(
                    linalg.dft(config.tpu_short_seq_length), dtype=ms.complex64
                )
                self.fourier_transform = partial(
                    two_dim_matmul,
                    matrix_dim_one=self.dft_mat_seq,
                    matrix_dim_two=self.dft_mat_hidden,
                )
            else:
                logging.warning(
                    "SciPy is needed for DFT matrix calculation and is not found. Using TPU optimized fast fourier"
                    " transform instead."
                )
                self.fourier_transform = fftn
        else:
            self.fourier_transform = fftn

    def construct(self, hidden_states):
        # NOTE: We do not use torch.vmap as it is not integrated into PyTorch stable versions.
        # Interested users can modify the code to use vmap from the nightly versions, getting the vmap from here:
        # https://pytorch.org/docs/master/generated/torch.vmap.html. Note that fourier transform methods will need
        # change accordingly.
        hidden_states = hidden_states.astype(ms.complex64)
        outputs = self.fourier_transform(hidden_states)
        return (outputs,)


class FNetBasicOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.LayerNorm(input_tensor + hidden_states)
        return hidden_states


class FNetFourierTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self = FNetBasicFourierTransform(config)
        self.output = FNetBasicOutput(config)

    def construct(self, hidden_states):
        self_outputs = self.self(hidden_states)
        fourier_output = self.output(self_outputs[0].astype(ms.float32), hidden_states)
        # print("1111111111111", hidden_states)
        outputs = (fourier_output,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->FNet
class FNetIntermediate(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->FNet
class FNetOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FNetLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1  # The dimension which has the sequence length
        self.fourier = FNetFourierTransform(config)
        self.intermediate = FNetIntermediate(config)
        self.output = FNetOutput(config)

    def construct(self, hidden_states):
        self_fourier_outputs = self.fourier(hidden_states)
        fourier_output = self_fourier_outputs[0]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            fourier_output,
        )

        outputs = (layer_output,)

        return outputs

    def feed_forward_chunk(self, fourier_output):
        intermediate_output = self.intermediate(fourier_output)
        layer_output = self.output(intermediate_output, fourier_output)
        return layer_output


class FNetEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList(
            [FNetLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def construct(self, hidden_states, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__, hidden_states
                )
            else:
                layer_outputs = layer_module(hidden_states)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->FNet
class FNetPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->FNet
class FNetPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FNetLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = FNetPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size)

        self.bias = ms.Parameter(ops.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _tie_weights(self) -> None:
        # For accelerate compatibility and to not break backward compatibility
        self.bias = self.decoder.bias


class FNetOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = FNetLMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->FNet
class FNetOnlyNSPHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->FNet
class FNetPreTrainingHeads(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = FNetLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FNetConfig
    base_model_prefix = "fnet"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.set_data(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
            # NOTE: Original code uses same initialization as weights for biases as well.
            if module.bias is not None:
                module.bias.set_data(
                    initializer("zeros", module.bias.shape, module.bias.dtype)
                )
        elif isinstance(module, nn.Embedding):
            module.weight.set_data(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = initializer(
                    "zeros",
                    module.weight[module.padding_idx].shape,
                    module.weight.dtype,
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.initialize("zeros")
            module.weight.data.fill(1.0)


@dataclass
class FNetForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FNetForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `ms.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`ms.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`ms.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """

    loss: Optional[ms.Tensor] = None
    prediction_logits: ms.Tensor = None
    seq_relationship_logits: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None


FNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Cell](https://pytorch.org/docs/stable/nn.html#torch.nn.Cell) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`FNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        token_type_ids (`ms.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`ms.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        inputs_embeds (`ms.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FNetModel(FNetPreTrainedModel):
    """

    The model can behave as an encoder, following the architecture described in [FNet: Mixing Tokens with Fourier
    Transforms](https://arxiv.org/abs/2105.03824) by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = FNetEmbeddings(config)
        self.encoder = FNetEncoder(config)

        self.pooler = FNetPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
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
        elif input_ids is not None:
            input_shape = input_ids.shape
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if (
            self.config.use_tpu_fourier_optimizations
            and seq_length <= 4096
            and self.config.tpu_short_seq_length != seq_length
        ):
            raise ValueError(
                "The `tpu_short_seq_length` in FNetConfig should be set equal to the sequence length being passed to"
                " the model when using TPU optimizations."
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=ms.int64)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooler_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class FNetForPreTraining(FNetPreTrainedModel):
    _tied_weights_keys = [
        "cls.predictions.decoder.bias",
        "cls.predictions.decoder.weight",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.fnet = FNetModel(config)
        self.cls = FNetPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        next_sentence_label: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FNetForPreTrainingOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FNetForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
        >>> model = FNetForPreTraining.from_pretrained("google/fnet-base")
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = ops.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            next_sentence_loss = ops.cross_entropy(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return FNetForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
        )


class FNetForMaskedLM(FNetPreTrainedModel):
    _tied_weights_keys = [
        "cls.predictions.decoder.bias",
        "cls.predictions.decoder.weight",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.fnet = FNetModel(config)
        self.cls = FNetOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )


class FNetForNextSentencePrediction(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.fnet = FNetModel(config)
        self.cls = FNetOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, NextSentencePredictorOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FNetForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
        >>> model = FNetForNextSentencePrediction.from_pretrained("google/fnet-base")
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        >>> outputs = model(**encoding, labels=ms.Tensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```"""

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            next_sentence_loss = ops.cross_entropy(
                seq_relationship_scores.view(-1, 2), labels.view(-1)
            )

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return (
                ((next_sentence_loss,) + output)
                if next_sentence_loss is not None
                else output
            )

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
        )


class FNetForSequenceClassification(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.fnet = FNetModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
                elif self.num_labels > 1 and (labels.dtype in (ms.int64, ms.int32)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )


class FNetForMultipleChoice(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.fnet = FNetModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
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

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states
        )


class FNetForTokenClassification(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.fnet = FNetModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )


class FNetForQuestionAnswering(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.fnet = FNetModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.cross_entropy(
                start_logits, start_positions, ignore_index=ignored_index
            )
            end_loss = ops.cross_entropy(
                end_logits, end_positions, ignore_index=ignored_index
            )
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "FNetForMaskedLM",
    "FNetForMultipleChoice",
    "FNetForNextSentencePrediction",
    "FNetForPreTraining",
    "FNetForQuestionAnswering",
    "FNetForSequenceClassification",
    "FNetForTokenClassification",
    "FNetModel",
]
