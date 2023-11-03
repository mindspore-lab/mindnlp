# coding=utf-8
# Copyright 2018 The Google MS Team Authors and The HuggingFace Inc. team.
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
# pylint: disable=C0115

""" Auto Model class."""


from collections import OrderedDict

from mindnlp.utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("albert", "MSAlbertModel"),
        ("bart", "MSBartModel"),
        ("beit", "MSBeitModel"),
        ("bert", "MSBertModel"),
        ("big_bird", "MSBigBirdModel"),
        ("blenderbot", "MSBlenderbotModel"),
        ("blenderbot-small", "MSBlenderbotSmallModel"),
        ("bloom", "MSBloomModel"),
        ("clip", "MSCLIPModel"),
        ("distilbert", "MSDistilBertModel"),
        ("electra", "MSElectraModel"),
        ("gpt-sw3", "MSGPT2Model"),
        ("gpt2", "MSGPT2Model"),
        ("gpt_neo", "MSGPTNeoModel"),
        ("gptj", "MSGPTJModel"),
        ("longt5", "MSLongT5Model"),
        ("marian", "MSMarianModel"),
        ("mbart", "MSMBartModel"),
        ("mt5", "MSMT5Model"),
        ("opt", "MSOPTModel"),
        ("pegasus", "MSPegasusModel"),
        ("regnet", "MSRegNetModel"),
        ("resnet", "MSResNetModel"),
        ("roberta", "MSRobertaModel"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormModel"),
        ("roformer", "MSRoFormerModel"),
        ("t5", "MST5Model"),
        ("vision-text-dual-encoder", "MSVisionTextDualEncoderModel"),
        ("vit", "MSViTModel"),
        ("wav2vec2", "MSWav2Vec2Model"),
        ("whisper", "MSWhisperModel"),
        ("xglm", "MSXGLMModel"),
        ("xlm-roberta", "MSXLMRobertaModel"),
    ]
)

FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "MSAlbertForPreTraining"),
        ("bart", "MSBartForConditionalGeneration"),
        ("bert", "MSBertForPreTraining"),
        ("big_bird", "MSBigBirdForPreTraining"),
        ("electra", "MSElectraForPreTraining"),
        ("longt5", "MSLongT5ForConditionalGeneration"),
        ("mbart", "MSMBartForConditionalGeneration"),
        ("mt5", "MSMT5ForConditionalGeneration"),
        ("roberta", "MSRobertaForMaskedLM"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForMaskedLM"),
        ("roformer", "MSRoFormerForMaskedLM"),
        ("t5", "MST5ForConditionalGeneration"),
        ("wav2vec2", "MSWav2Vec2ForPreTraining"),
        ("whisper", "MSWhisperForConditionalGeneration"),
        ("xlm-roberta", "MSXLMRobertaForMaskedLM"),
    ]
)

FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "MSAlbertForMaskedLM"),
        ("bart", "MSBartForConditionalGeneration"),
        ("bert", "MSBertForMaskedLM"),
        ("big_bird", "MSBigBirdForMaskedLM"),
        ("distilbert", "MSDistilBertForMaskedLM"),
        ("electra", "MSElectraForMaskedLM"),
        ("mbart", "MSMBartForConditionalGeneration"),
        ("roberta", "MSRobertaForMaskedLM"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForMaskedLM"),
        ("roformer", "MSRoFormerForMaskedLM"),
        ("xlm-roberta", "MSXLMRobertaForMaskedLM"),
    ]
)

FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "MSBartForConditionalGeneration"),
        ("blenderbot", "MSBlenderbotForConditionalGeneration"),
        ("blenderbot-small", "MSBlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "MSEncoderDecoderModel"),
        ("longt5", "MSLongT5ForConditionalGeneration"),
        ("marian", "MSMarianMTModel"),
        ("mbart", "MSMBartForConditionalGeneration"),
        ("mt5", "MSMT5ForConditionalGeneration"),
        ("pegasus", "MSPegasusForConditionalGeneration"),
        ("t5", "MST5ForConditionalGeneration"),
    ]
)

FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image-classsification
        ("beit", "MSBeitForImageClassification"),
        ("regnet", "MSRegNetForImageClassification"),
        ("resnet", "MSResNetForImageClassification"),
        ("vit", "MSViTForImageClassification"),
    ]
)

FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("vision-encoder-decoder", "MSVisionEncoderDecoderModel"),
    ]
)

FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bart", "MSBartForCausalLM"),
        ("bert", "MSBertForCausalLM"),
        ("big_bird", "MSBigBirdForCausalLM"),
        ("bloom", "MSBloomForCausalLM"),
        ("electra", "MSElectraForCausalLM"),
        ("gpt-sw3", "MSGPT2LMHeadModel"),
        ("gpt2", "MSGPT2LMHeadModel"),
        ("gpt_neo", "MSGPTNeoForCausalLM"),
        ("gptj", "MSGPTJForCausalLM"),
        ("opt", "MSOPTForCausalLM"),
        ("roberta", "MSRobertaForCausalLM"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForCausalLM"),
        ("xglm", "MSXGLMForCausalLM"),
        ("xlm-roberta", "MSXLMRobertaForCausalLM"),
    ]
)

FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "MSAlbertForSequenceClassification"),
        ("bart", "MSBartForSequenceClassification"),
        ("bert", "MSBertForSequenceClassification"),
        ("big_bird", "MSBigBirdForSequenceClassification"),
        ("distilbert", "MSDistilBertForSequenceClassification"),
        ("electra", "MSElectraForSequenceClassification"),
        ("mbart", "MSMBartForSequenceClassification"),
        ("roberta", "MSRobertaForSequenceClassification"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForSequenceClassification"),
        ("roformer", "MSRoFormerForSequenceClassification"),
        ("xlm-roberta", "MSXLMRobertaForSequenceClassification"),
    ]
)

FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "MSAlbertForQuestionAnswering"),
        ("bart", "MSBartForQuestionAnswering"),
        ("bert", "MSBertForQuestionAnswering"),
        ("big_bird", "MSBigBirdForQuestionAnswering"),
        ("distilbert", "MSDistilBertForQuestionAnswering"),
        ("electra", "MSElectraForQuestionAnswering"),
        ("mbart", "MSMBartForQuestionAnswering"),
        ("roberta", "MSRobertaForQuestionAnswering"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForQuestionAnswering"),
        ("roformer", "MSRoFormerForQuestionAnswering"),
        ("xlm-roberta", "MSXLMRobertaForQuestionAnswering"),
    ]
)

FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "MSAlbertForTokenClassification"),
        ("bert", "MSBertForTokenClassification"),
        ("big_bird", "MSBigBirdForTokenClassification"),
        ("distilbert", "MSDistilBertForTokenClassification"),
        ("electra", "MSElectraForTokenClassification"),
        ("roberta", "MSRobertaForTokenClassification"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForTokenClassification"),
        ("roformer", "MSRoFormerForTokenClassification"),
        ("xlm-roberta", "MSXLMRobertaForTokenClassification"),
    ]
)

FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "MSAlbertForMultipleChoice"),
        ("bert", "MSBertForMultipleChoice"),
        ("big_bird", "MSBigBirdForMultipleChoice"),
        ("distilbert", "MSDistilBertForMultipleChoice"),
        ("electra", "MSElectraForMultipleChoice"),
        ("roberta", "MSRobertaForMultipleChoice"),
        ("roberta-prelayernorm", "MSRobertaPreLayerNormForMultipleChoice"),
        ("roformer", "MSRoFormerForMultipleChoice"),
        ("xlm-roberta", "MSXLMRobertaForMultipleChoice"),
    ]
)

FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "MSBertForNextSentencePrediction"),
    ]
)

FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech-encoder-decoder", "MSSpeechEncoderDecoderModel"),
        ("whisper", "MSWhisperForConditionalGeneration"),
    ]
)

FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("whisper", "MSWhisperForAudioClassification"),
    ]
)

FLAX_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_MAPPING_NAMES)
FLAX_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES)
FLAX_MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES)
FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
FLAX_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)
FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
)
FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)


class MSAutoModel(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_MAPPING


MSAutoModel = auto_class_update(MSAutoModel)


class MSAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_PRETRAINING_MAPPING


MSAutoModelForPreTraining = auto_class_update(MSAutoModelForPreTraining, head_doc="pretraining")


class MSAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING


MSAutoModelForCausalLM = auto_class_update(MSAutoModelForCausalLM, head_doc="causal language modeling")


class MSAutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_MASKED_LM_MAPPING


MSAutoModelForMaskedLM = auto_class_update(MSAutoModelForMaskedLM, head_doc="masked language modeling")


class MSAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


MSAutoModelForSeq2SeqLM = auto_class_update(
    MSAutoModelForSeq2SeqLM, head_doc="sequence-to-sequence language modeling", checkpoint_for_example="t5-base"
)


class MSAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


MSAutoModelForSequenceClassification = auto_class_update(
    MSAutoModelForSequenceClassification, head_doc="sequence classification"
)


class MSAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING


MSAutoModelForQuestionAnswering = auto_class_update(MSAutoModelForQuestionAnswering, head_doc="question answering")


class MSAutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


MSAutoModelForTokenClassification = auto_class_update(
    MSAutoModelForTokenClassification, head_doc="token classification"
)


class MSAutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING


MSAutoModelForMultipleChoice = auto_class_update(MSAutoModelForMultipleChoice, head_doc="multiple choice")


class MSAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


MSAutoModelForNextSentencePrediction = auto_class_update(
    MSAutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


class MSAutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


MSAutoModelForImageClassification = auto_class_update(
    MSAutoModelForImageClassification, head_doc="image classification"
)


class MSAutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING


MSAutoModelForVision2Seq = auto_class_update(MSAutoModelForVision2Seq, head_doc="vision-to-text modeling")


class MSAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


MSAutoModelForSpeechSeq2Seq = auto_class_update(
    MSAutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)
