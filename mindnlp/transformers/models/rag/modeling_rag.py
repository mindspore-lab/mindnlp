# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""RAG model implementation."""
# pylint: disable=unexpected-keyword-arg, not-callable, consider-using-generator

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import mindspore
import numpy as np

import mindnlp.core.nn.functional as F
from mindnlp.core import ops
from mindnlp.utils import logging
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RagConfig"


@dataclass
class RetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (`mindspore.Tensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        past_key_values (`List[mindspore.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `mindspore.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (`mindspore.Tensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`mindspore.Tensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    doc_scores: mindspore.Tensor = None
    past_key_values: Optional[List[mindspore.Tensor]] = None
    retrieved_doc_embeds: Optional[mindspore.Tensor] = None
    retrieved_doc_ids: Optional[mindspore.Tensor] = None
    context_input_ids: Optional[mindspore.Tensor] = None
    context_attention_mask: Optional[mindspore.Tensor] = None
    question_encoder_last_hidden_state: Optional[mindspore.Tensor] = None
    question_enc_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    question_enc_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_enc_last_hidden_state: Optional[mindspore.Tensor] = None
    generator_enc_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_enc_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_dec_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_dec_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_cross_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class RetrievAugLMOutput(ModelOutput):
    """
    Args:
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (`mindspore.Tensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        past_key_values (`List[mindspore.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `mindspore.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (`mindspore.Tensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`mindspore.Tensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    logits: mindspore.Tensor = None
    doc_scores: mindspore.Tensor = None
    past_key_values: Optional[List[mindspore.Tensor]] = None
    retrieved_doc_embeds: Optional[mindspore.Tensor] = None
    retrieved_doc_ids: Optional[mindspore.Tensor] = None
    context_input_ids: Optional[mindspore.Tensor] = None
    context_attention_mask: Optional[mindspore.Tensor] = None
    question_encoder_last_hidden_state: Optional[mindspore.Tensor] = None
    question_enc_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    question_enc_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_enc_last_hidden_state: Optional[mindspore.Tensor] = None
    generator_enc_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_enc_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_dec_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_dec_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    generator_cross_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


class RagPreTrainedModel(PreTrainedModel):
    r"""
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """

    config_class = RagConfig
    base_model_prefix = "rag"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained_question_encoder_generator(
            cls,
            question_encoder_pretrained_model_name_or_path: str = None,
            generator_pretrained_model_name_or_path: str = None,
            retriever: RagRetriever = None,
            **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained
        model checkpoints.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            question_encoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the question encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            generator_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the generator. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            retriever ([`RagRetriever`], *optional*):
                The retriever to use.
            kwwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the question_encoder configuration, use the prefix *question_encoder_* for each
                  configuration parameter.
                - To update the generator configuration, use the prefix *generator_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        """

        kwargs_question_encoder = {
            argument[len("question_encoder_"):]: value
            for argument, value in kwargs.items()
            if argument.startswith("question_encoder_")
        }

        kwargs_generator = {
            argument[len("generator_"):]: value
            for argument, value in kwargs.items()
            if argument.startswith("generator_")
        }

        # remove question_encoder, generator kwargs from kwargs
        for key in kwargs_question_encoder.keys():
            del kwargs["question_encoder_" + key]
        for key in kwargs_generator.keys():
            del kwargs["generator_" + key]

        # Load and initialize the question_encoder and generator
        # The distinction between question_encoder and generator at the model level is made
        # by the value of the flag `is_generator` that we need to set correctly.
        question_encoder = kwargs_question_encoder.pop("model", None)
        if question_encoder is None:
            assert question_encoder_pretrained_model_name_or_path is not None, (
                "If `model` is not defined as an argument, a `question_encoder_pretrained_model_name_or_path` has to"
                " be defined"
            )
            from ..auto.modeling_auto import AutoModel

            if "config" not in kwargs_question_encoder:
                from ..auto.configuration_auto import AutoConfig

                question_encoder_config, kwargs_question_encoder = AutoConfig.from_pretrained(
                    question_encoder_pretrained_model_name_or_path,
                    **kwargs_question_encoder,
                    return_unused_kwargs=True,
                )
                kwargs_question_encoder["config"] = question_encoder_config

            question_encoder = AutoModel.from_pretrained(
                question_encoder_pretrained_model_name_or_path, **kwargs_question_encoder
            )

        generator = kwargs_generator.pop("model", None)
        if generator is None:
            assert generator_pretrained_model_name_or_path is not None, (
                "If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has"
                " to be defined"
            )
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            if "config" not in kwargs_generator:
                from ..auto.configuration_auto import AutoConfig

                generator_config, kwargs_generator = AutoConfig.from_pretrained(
                    generator_pretrained_model_name_or_path, **kwargs_generator, return_unused_kwargs=True
                )

                kwargs_generator["config"] = generator_config

            generator = AutoModelForSeq2SeqLM.from_pretrained(
                generator_pretrained_model_name_or_path, **kwargs_generator
            )

        # instantiate config with corresponding kwargs
        config = kwargs.get("config", None)
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        return cls(question_encoder=question_encoder, generator=generator, config=config,
                   retriever=retriever)  # pylint: disable=unexpected-keyword-arg


class RagModel(RagPreTrainedModel):
    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            question_encoder: Optional[PreTrainedModel] = None,
            generator: Optional[PreTrainedModel] = None,
            retriever: Optional[RagRetriever] = None,  # or maybe just use a `set_retriever(...)` method
            **kwargs,
    ):
        assert config is not None or (
                question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config)
        if question_encoder is None:
            from ..auto.modeling_auto import AutoModel

            question_encoder = AutoModel.from_config(
                config.question_encoder
            )

        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            generator = AutoModelForSeq2SeqLM.from_config(
                config.generator
            )

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        self.ctx_encoder = None
        self.context_encoder_training = False

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            decoder_input_ids: Optional[mindspore.Tensor] = None,
            decoder_attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            doc_scores: Optional[mindspore.Tensor] = None,
            context_input_ids: Optional[mindspore.Tensor] = None,
            context_attention_mask: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_retrieved: Optional[bool] = None,
            n_docs: Optional[int] = None,
    ) -> Union[Tuple[mindspore.Tensor], RetrievAugLMOutput]:

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )
                question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.to(mindspore.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    return_tensors="ms",
                )
                if self.context_encoder_training:
                    (
                        context_input_ids,
                        context_attention_mask,
                        retrieved_doc_embeds,
                        retrived_doc_input_ids,
                        retrived_doc_attention_mask,
                        retrieved_doc_ids,
                    ) = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["tokenized_doc_ids"],
                        retriever_outputs["tokenized_doc_attention_mask"],
                        retriever_outputs["doc_ids"],
                    )

                    context_input_ids = context_input_ids.to(input_ids.dtype)
                    context_attention_mask = context_attention_mask.to(input_ids.dtype)

                    retrived_doc_input_ids = retrived_doc_input_ids.to(input_ids.dtype)
                    retrived_doc_attention_mask = retrived_doc_attention_mask.to(input_ids.dtype)
                    retrieved_doc_embeds = self.ctx_encoder(
                        retrived_doc_input_ids, attention_mask=retrived_doc_attention_mask, return_dict=True
                    ).pooler_output
                    retrieved_doc_embeds = retrieved_doc_embeds.view(
                        -1, n_docs, question_encoder_last_hidden_state.shape[1]
                    )  # reshaping

                    # compute doc_scores involving ctx_encoder
                    doc_scores = ops.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), ops.transpose(retrieved_doc_embeds, 1, 2)
                    ).squeeze(1)

                else:
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["doc_ids"],
                    )

                    # set to correct device
                    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state.dtype)
                    context_input_ids = context_input_ids.to(input_ids.dtype)
                    context_attention_mask = context_attention_mask.to(input_ids.dtype)

                    # compute doc_scores
                    doc_scores = ops.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), ops.transpose(retrieved_doc_embeds, 1, 2)
                    ).squeeze(1)
            else:
                assert context_input_ids is not None, (
                    "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can"
                    " set a retriever using the `set_retriever(...)` function."
                )
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

        assert (
                doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (doc_scores.shape[1] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
        )

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = ops.repeat_interleave(decoder_input_ids,n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.astype(mindspore.int32)
            decoder_attention_mask = ops.repeat_interleave(decoder_attention_mask,n_docs, dim=0)

        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
            generator_cross_attentions=gen_outputs.cross_attentions,
        )


class RagSequenceForGeneration(RagPreTrainedModel):
    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            question_encoder: Optional[PreTrainedModel] = None,
            generator: Optional[PreTrainedModel] = None,
            retriever: Optional[RagRetriever] = None,
            **kwargs,
    ):
        assert config is not None or (
                question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        super().__init__(config)

        # instantiate model
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            decoder_input_ids: Optional[mindspore.Tensor] = None,
            decoder_attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            context_input_ids: Optional[mindspore.Tensor] = None,
            context_attention_mask: Optional[mindspore.Tensor] = None,
            doc_scores: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_retrieved: Optional[bool] = None,
            exclude_bos_score: Optional[bool] = None,
            reduce_loss: Optional[bool] = None,
            labels: Optional[mindspore.Tensor] = None,
            n_docs: Optional[int] = None,
            **kwargs,  # needs kwargs for generation
    ) -> RetrievAugLMMarginOutput:
        r"""
        exclude_bos_score (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the score of the BOS token is disregarded when computing
            the loss.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `mindspore.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
             Legacy dictionary, which is required so that model can use *generate()* function.

        Returns:

        """

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        if labels is not None:
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                decoder_input_ids,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                exclude_bos_score=exclude_bos_score,
                n_docs=n_docs,
            )

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=outputs.logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @mindspore._no_grad()
    def generate(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            context_input_ids: Optional[mindspore.Tensor] = None,
            context_attention_mask: Optional[mindspore.Tensor] = None,
            doc_scores: Optional[mindspore.Tensor] = None,
            do_deduplication: Optional[bool] = None,  # defaults to True
            num_return_sequences: Optional[int] = None,  # defaults to 1
            num_beams: Optional[int] = None,  # defaults to 1
            n_docs: Optional[int] = None,
            **model_kwargs,
    ) -> mindspore.Tensor:
        """
        Implements RAG sequence "thorough" decoding. Read the [`~generation.GenerationMixin.generate`]` documentation
        for more information on how to set other generate input parameters.

        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            context_attention_mask (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model is not initialized with a `retriever` or `input_ids` is not given, `context_input_ids` and
                `context_attention_mask` have to be provided to the forward pass. They are returned by
                [`~RagRetriever.__call__`].
            doc_scores (`mindspore.Tensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model is not initialized with a `retriever` or `input_ids` is not given, `doc_scores` has to be
                provided to the forward pass. `doc_scores` are returned by [`~RagRetriever.__call__`].
            do_deduplication (`bool`, *optional*):
                Whether or not to deduplicate the generations from different context documents for a given input. Has
                to be set to `False` if used while training with distributed backend.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation.GenerationMixin.generate`]` function,
                where we set `num_return_sequences` to `num_beams`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs will be passed to [`~generation.GenerationMixin.generate`].

        Return:
            `mindspore.Tensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """


        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        assert (
                input_ids is not None or context_input_ids is not None
        ), " At least one of input_ids or context_input_ids must be given"

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[
                0]  # pylint: disable=not-callable
            context_input_ids = self.retriever(  # pylint: disable=not-callable
                input_ids,
                question_hidden_states.to(mindspore.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="ms",
            )["context_input_ids"]

            # set to correct device
            context_input_ids = context_input_ids.to(input_ids.dtype)

        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams
        model_kwargs["attention_mask"] = None

        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs

        for index in range(batch_size):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs: (index + 1) * n_docs]  # (n_docs, max_len)
            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            if do_deduplication:
                # do_deduplication, max_output_len
                output_sequences = ops.stack(list({str(k.tolist()): k for k in output_sequences}.values()))

            num_candidates = output_sequences.shape[
                0
            ]  # after deduplication, this number can be less than n_docs*n_beam

            # then, run model forwards to get nll scores:
            if input_ids is not None:

                new_input_ids = ops.tile(input_ids[index: index + 1], (num_candidates, 1))
                outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            else:  # input_ids is None, need context_input_ids/mask and doc_scores
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

                individual_input_ids = ops.tile(generator_input_ids, (
                    num_candidates, 1
                ))  # (num_candidates*n_docs, max_len)

                individual_attention_mask = context_attention_mask[index * n_docs: (index + 1) * n_docs]
                individual_attention_mask = ops.tile(individual_attention_mask, (num_candidates, 1))

                individual_doc_scores = doc_scores[index: (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = ops.tile(individual_doc_scores, (num_candidates, 1))  # [num_candidates, n_docs]

                outputs = self(
                    context_input_ids=individual_input_ids,
                    context_attention_mask=individual_attention_mask,
                    doc_scores=individual_doc_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                )

            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    def get_nll(
            self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        # shift tokens left
        new_fill = mindspore.Tensor((np.full((target.shape[0], 1), self.config.generator.pad_token_id)),
                                    dtype=target[0].dtype)
        target = ops.cat(
            # [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
            [target[:, 1:], new_fill], 1
        )

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # bos_token_id is None for T5
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill(pad_mask, 0.0)
                smooth_obj.masked_fill(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
        seq_logprobs = F.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.shape[-1]
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = F.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = ops.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # calculate loss
        target = ops.tile(target.unsqueeze(1).unsqueeze(-1), (1, n_docs, 1, 1))
        assert target.dim() == rag_logprobs.dim()

        ll = ops.gather(rag_logprobs, dim=-1, index=target)
        # ll = rag_logprobs.gather_nd(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(axis=-1, keepdims=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.shape[-1]
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        # output = (
        #     tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        # )
        new_fill = mindspore.Tensor(
            np.full((sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])), pad_token_id),
            # pylint: disable=consider-using-generator
            dtype=tensors[0].dtype)
        output = (
            new_fill
        )
        ind = 0
        for t in tensors:
            output[ind: ind + t.shape[0], : t.shape[1]] = t
            ind += t.shape[0]
        return output


class RagTokenForGeneration(RagPreTrainedModel):
    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            question_encoder: Optional[PreTrainedModel] = None,
            generator: Optional[PreTrainedModel] = None,
            retriever: Optional[RagRetriever] = None,
            **kwargs,
    ):
        assert config is not None or (
                question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        # instantiate model
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            doc_scores=None,
            n_docs=None,
            **kwargs,
    ):
        if past_key_values is not None:
            # if past is defined use only last decoder_input_ids
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "doc_scores": doc_scores,
            "context_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "do_marginalize": True,
            "n_docs": n_docs,
        }

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs"""

        def _reorder_stacked(hidden_states, new_order):
            n_docs = hidden_states.shape[0] // new_order.shape[0]
            hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
            hidden_states = hidden_states.index_select(0, new_order)
            result = hidden_states.view(-1, *hidden_states.shape[2:])
            return result

        reordered_past = ()
        for layer_past in past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            reordered_past += (
                tuple(_reorder_stacked(past_state, beam_idx) for past_state in layer_past),
            )

        return reordered_past

    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # RAG-token marginalization
        seq_logprobs = F.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.shape[-1]
        )
        doc_logprobs = F.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        # return ops.logsumexp(log_prob_sum, dim=1)
        return mindspore.ops.logsumexp(log_prob_sum, axis=1)

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            decoder_input_ids: Optional[mindspore.Tensor] = None,
            decoder_attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            context_input_ids: Optional[mindspore.Tensor] = None,
            context_attention_mask: Optional[mindspore.Tensor] = None,
            doc_scores: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_retrieved: Optional[bool] = None,
            do_marginalize: Optional[bool] = None,
            reduce_loss: Optional[bool] = None,
            labels: Optional[mindspore.Tensor] = None,
            n_docs: Optional[int] = None,
            **kwargs,  # needs kwargs for generation
    ) -> RetrievAugLMMarginOutput:
        r"""
        do_marginalize (`bool`, *optional*):
            If `True`, the logits are marginalized over all documents by making use of
            `ops.F.log_softmax`.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `mindspore.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Legacy dictionary, which is required so that model can use *generate()* function.

        Returns:
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @mindspore._no_grad()
    def generate(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            context_input_ids: Optional[mindspore.Tensor] = None,
            context_attention_mask: Optional[mindspore.Tensor] = None,
            doc_scores: Optional[mindspore.Tensor] = None,
            n_docs: Optional[int] = None,
            generation_config: Optional[GenerationConfig] = None,
            prefix_allowed_tokens_fn: Callable[[int, mindspore.Tensor], List[int]] = None,
            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
            **kwargs,
    ) -> mindspore.Tensor:
        """
        Implements RAG token decoding.

        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            context_attention_mask (`mindspore.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            doc_scores (`mindspore.Tensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            prefix_allowed_tokens_fn (`Callable[[int, mindspore.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID
                `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on
                the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for
                constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and a
                model's config. If a logit processor is passed that is already created with the arguments or a model's
                config an error is thrown.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                model's config. If a stopping criteria is passed that is already created with the arguments or a
                model's config an error is thrown.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            `mindspore.Tensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """
        # Handle `generation_config` and kwargs that might update it
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        def _prepare_special_tokens(
                self,
                generation_config: GenerationConfig,
                kwargs_has_attention_mask: Optional[bool] = None,
        ):
            """
            Prepares the special tokens for generation, overwriting the generation config with their processed versions
            converted to tensor.

            Note that `generation_config` is changed in place and stops being serializable after this method is called.
            That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
            function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
            """

            # Convert special tokens to tensors
            def _tensor_or_none(token):
                if token is None:
                    return token
                if isinstance(token, mindspore.Tensor):
                    return token
                return mindspore.Tensor(token, dtype=mindspore.int64)

            bos_token_tensor = _tensor_or_none(generation_config.bos_token_id)
            eos_token_tensor = _tensor_or_none(generation_config.eos_token_id)
            pad_token_tensor = _tensor_or_none(generation_config.pad_token_id)
            decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id)

            # for BC we also try to get `decoder_start_token_id` or `bos_token_id` (#30892)
            if self.config.is_encoder_decoder:
                decoder_start_token_tensor = (
                    decoder_start_token_tensor if decoder_start_token_tensor is not None else bos_token_tensor
                )

            # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
            if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
                eos_token_tensor = eos_token_tensor.unsqueeze(0)

            # Set pad token if unset (and there are conditions to do so)
            if pad_token_tensor is None and eos_token_tensor is not None:
                if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                    logger.warning(
                        "The attention mask and the pad token id were not set. As a consequence, you may observe "
                        "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                    )
                pad_token_tensor = eos_token_tensor[0]
                logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

            # we can't infer attn mask if pad token is set to be eos token in model's generation config
            if eos_token_tensor is not None and pad_token_tensor in eos_token_tensor:
                if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                    logger.warning_once(
                        "The attention mask is not set and cannot be inferred from input because pad token is same as eos token."
                        "As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` "
                        "to obtain reliable results."
                    )

            # Sanity checks/warnings
            if self.config.is_encoder_decoder and decoder_start_token_tensor is None:
                raise ValueError(
                    "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
                )
            if eos_token_tensor is not None and (
                    ops.is_floating_point(eos_token_tensor) or (eos_token_tensor < 0).any()
            ):
                logger.warning(
                    f"`eos_token_id` should consist of positive integers, but is {eos_token_tensor}. Your generation will not "
                    "stop until the maximum length is reached. Depending on other flags, it may even crash."
                )

            generation_config._bos_token_tensor = bos_token_tensor
            generation_config._eos_token_tensor = eos_token_tensor
            generation_config._pad_token_tensor = pad_token_tensor
            generation_config._decoder_start_token_tensor = decoder_start_token_tensor

        _prepare_special_tokens(self, generation_config, kwargs_has_attention_mask)

        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # retrieve docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[
                0]  # pylint: disable=not-callable
            out = self.retriever(  # pylint: disable=not-callable
                input_ids,
                question_hidden_states.to(mindspore.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="ms",
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            # set to correct device
            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states.dtype)
            context_input_ids = context_input_ids.to(input_ids.dtype)
            context_attention_mask = context_attention_mask.to(input_ids.dtype)

            # compute doc_scores
            doc_scores = ops.bmm(question_hidden_states.unsqueeze(1),
                                 ops.transpose(retrieved_doc_embeds, 1, 2)).squeeze(
                1
            )

        assert (context_input_ids.shape[0] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
        )

        # batch_size
        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)

        input_ids = ops.full(
            (batch_size * generation_config.num_beams, 1),
            generation_config.decoder_start_token_id,
            dtype=mindspore.int64,
        )
        input_ids_seq_length = input_ids.shape[-1]
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.broadcast_to((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=generation_config.num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(
            last_hidden_state, num_beams=generation_config.num_beams
        )

        doc_scores = ops.repeat_interleave(doc_scores,generation_config.num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        pre_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=context_input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if generation_config.num_beams == 1:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                    " greedy search."
                )
            return self._sample(
                input_ids,
                logits_processor=pre_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=False,
                streamer=None,
                logits_warper=None,
                **model_kwargs,
            )
        elif generation_config.num_beams > 1:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            return self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=pre_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=False,
                logits_warper=None,
                **model_kwargs,
            )
        else:
            raise ValueError(
                f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {generation_config.num_beams}"
            )

    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.rag.generator.set_output_embeddings(new_embeddings)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # shift tokens left
        new_fill = mindspore.Tensor((np.full((target.shape[0], 1), self.config.generator.pad_token_id)),
                                    dtype=target[0].dtype)
        target = ops.cat(
            [target[:, 1:],new_fill], 1
            # [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill(pad_mask, 0.0)
                smooth_obj.masked_fill(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        # ll = rag_logprobs.gather_nd(dim=-1, index=target)

        ll = ops.gather(rag_logprobs,dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(axis=-1, keepdims=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.shape[-1]
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


__all__ = [
    "RagModel",
    "RagPreTrainedModel",
    "RagSequenceForGeneration",
    "RagTokenForGeneration",
]
