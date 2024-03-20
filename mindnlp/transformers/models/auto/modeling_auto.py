# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
# ============================================================================
# pylint: disable=C0115

""" Auto Model class."""

import warnings
from collections import OrderedDict

from mindnlp.utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("albert", "AlbertModel"),
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("audio-spectrogram-transformer", "ASTModel"),
        ("autoformer", "AutoformerModel"),
        ("bark", "BarkModel"),
        ("bart", "BartModel"),
        ("bert", "BertModel"),
        ("big_bird", "BigBirdModel"),
        ("biogpt", "BioGptModel"),
        ("bloom", "BloomModel"),
        ("codegen", "CodeGenModel"),
        ("cpmant", "CpmAntModel"),
        ("cpmbee", "CpmBeeModel"),
        ("chatglm", "ChatGLMModel"),
        ("clip", "CLIPModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("deberta", "DebertaModel"),
        ('encodec','EncodecModel'),
        ("esm", "EsmModel"),
        ("ernie", "ErnieModel"),
        ("ernie_m", "ErnieMModel"),
        ("falcon", "FalconModel"),
        ("gemma", "GemmaModel"),
        ("gpt_bigcode", "GPTBigCodeModel"),
        ("gpt", "GPTModel"),
        ("gpt2", "GPT2Model"),
        ("gpt_pangu", "GPTPanguModel"),
        ("longformer", "LongformerModel"),
        ("mamba", "MambaModel"),
        ('mbart','MBartModel'),
        ('minicpm', 'MiniCPMModel'),
        ("mistral", "MistralModel"),
        ("mixtral", "MixtralModel"),
        ("phi", "PhiModel"),
        ("qwen2", "Qwen2Model"),
        ("reformer", "ReformerModel"),
        ("roberta", "RobertaModel"),
        ("rwkv", "RwkvModel"),
        ("starcoder2", "Starcoder2Model"),
        ("t5", "T5Model"),
        ("whisper", "WhisperModel"),
        ("xlm-roberta", "XLMRobertaModel"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "AlbertForPreTraining"),
        ("bert", "BertForPreTraining"),
        ("big_bird", "BigBirdForPreTraining"),
        ("bloom", "BloomForCausalLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("gpt_pangu", "GPTPanguForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ('minicpm', 'MiniCPMForCausalLM'),
        ("rwkv", "RwkvForCausalLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "AlbertForMaskedLM"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("bloom", "BloomForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("cpmbee", "CpmBeeForCausalLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("gpt_pangu", "GPTPanguForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ('minicpm', 'MiniCPMForCausalLM'),
        ("reformer", "ReformerModelWithLMHead"),
        ("roberta", "RobertaForMaskedLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("whisper", "WhisperForConditionalGeneration"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bert", "BertLMHeadModel"),
        ("big_bird", "BigBirdForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("cpmbee", "CpmBeeForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_pangu", "GPTPanguForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ('minicpm', 'MiniCPMForCausalLM'),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("roberta", "RobertaLMHeadModel"),
        ("rwkv", "RwkvForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
    ]
)


MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image Classification mapping
        ("beit", "BeitForImageClassification"),
        ("bit", "BitForImageClassification"),
        ("clip", "CLIPForImageClassification"),
        ("convnext", "ConvNextForImageClassification"),
        ("convnextv2", "ConvNextV2ForImageClassification"),
        ("cvt", "CvtForImageClassification"),
        ("data2vec-vision", "Data2VecVisionForImageClassification"),
        (
            "deit",
            ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher"),
        ),
        ("dinat", "DinatForImageClassification"),
        ("dinov2", "Dinov2ForImageClassification"),
        (
            "efficientformer",
            (
                "EfficientFormerForImageClassification",
                "EfficientFormerForImageClassificationWithTeacher",
            ),
        ),
        ("efficientnet", "EfficientNetForImageClassification"),
        ("focalnet", "FocalNetForImageClassification"),
        ("imagegpt", "ImageGPTForImageClassification"),
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),
        ),
        ("mobilenet_v1", "MobileNetV1ForImageClassification"),
        ("mobilenet_v2", "MobileNetV2ForImageClassification"),
        ("mobilevit", "MobileViTForImageClassification"),
        ("mobilevitv2", "MobileViTV2ForImageClassification"),
        ("nat", "NatForImageClassification"),
        (
            "perceiver",
            (
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
            ),
        ),
        ("poolformer", "PoolFormerForImageClassification"),
        ("pvt", "PvtForImageClassification"),
        ("regnet", "RegNetForImageClassification"),
        ("resnet", "ResNetForImageClassification"),
        ("segformer", "SegformerForImageClassification"),
        ("siglip", "SiglipForImageClassification"),
        ("swiftformer", "SwiftFormerForImageClassification"),
        ("swin", "SwinForImageClassification"),
        ("swinv2", "Swinv2ForImageClassification"),
        ("van", "VanForImageClassification"),
        ("vit", "ViTForImageClassification"),
        ("vit_hybrid", "ViTHybridForImageClassification"),
        ("vit_msn", "ViTMSNForImageClassification"),
    ]
)

MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Instance Segmentation mapping
        # MaskFormerForInstanceSegmentation can be removed from this mapping in v5
        ("maskformer", "MaskFormerForInstanceSegmentation"),
    ]
)

MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Universal Segmentation mapping
        ("detr", "DetrForSegmentation"),
        ("mask2former", "Mask2FormerForUniversalSegmentation"),
        ("maskformer", "MaskFormerForInstanceSegmentation"),
        ("oneformer", "OneFormerForUniversalSegmentation"),
    ]
)

MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("timesformer", "TimesformerForVideoClassification"),
        ("videomae", "VideoMAEForVideoClassification"),
        ("vivit", "VivitForVideoClassification"),
    ]
)

MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("git", "GitForCausalLM"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "AlbertForMaskedLM"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("reformer", "ReformerForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Object Detection mapping
        ("conditional_detr", "ConditionalDetrForObjectDetection"),
        ("deformable_detr", "DeformableDetrForObjectDetection"),
        ("deta", "DetaForObjectDetection"),
        ("detr", "DetrForObjectDetection"),
        ("table-transformer", "TableTransformerForObjectDetection"),
        ("yolos", "YolosForObjectDetection"),
    ]
)

MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Object Detection mapping
        ("owlv2", "Owlv2ForObjectDetection"),
        ("owlvit", "OwlViTForObjectDetection"),
    ]
)

MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for depth estimation mapping
        ("dpt", "DPTForDepthEstimation"),
        ("glpn", "GLPNForDepthEstimation"),
    ]
)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "BartForConditionalGeneration"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("blenderbot", "BlenderbotForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ('chatglm', "ChatGLMForConditionalGeneration"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("gptsan-japanese", "GPTSanJapaneseForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("marian", "MarianMTModel"),
        ("mbart", "MBartForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("pegasus", "PegasusForConditionalGeneration"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("prophetnet", "ProphetNetForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForTextToText"),
        ("switch_transformers", "SwitchTransformersForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("umt5", "UMT5ForConditionalGeneration"),
        ("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
    ]
)

MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForSpeechToText"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderModel"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("speecht5", "SpeechT5ForSpeechToText"),
        ("whisper", "WhisperForConditionalGeneration"),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "AlbertForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("bert", "BertForSequenceClassification"),
        ("big_bird", "BigBirdForSequenceClassification"),
        ("biogpt", "BioGptForSequenceClassification"),
        ("bloom", "BloomForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("esm", "EsmForSequenceClassification"),
        ("falcon", "FalconForSequenceClassification"),
        ('minicpm', 'MiniCPMForSequenceClassification'),
        ("mistral", "MistralForSequenceClassification"),
        ("mixtral", "MixtralForSequenceClassification"),
        ("phi", "PhiForSequenceClassification"),
        ("qwen2", "Qwen2ForSequenceClassification"),
        ("reformer", "ReformerForSequenceClassification"),
        ("starcoder2", "Starcoder2ForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "AlbertForQuestionAnswering"),
        ("bart", "BartForQuestionAnswering"),
        ("bert", "BertForQuestionAnswering"),
        ("big_bird", "BigBirdForQuestionAnswering"),
        ("bigbird_pegasus", "BigBirdPegasusForQuestionAnswering"),
        ("bloom", "BloomForQuestionAnswering"),
        ("camembert", "CamembertForQuestionAnswering"),
        ("canine", "CanineForQuestionAnswering"),
        ("convbert", "ConvBertForQuestionAnswering"),
        ("data2vec-text", "Data2VecTextForQuestionAnswering"),
        ("deberta", "DebertaForQuestionAnswering"),
        ("deberta-v2", "DebertaV2ForQuestionAnswering"),
        ("distilbert", "DistilBertForQuestionAnswering"),
        ("electra", "ElectraForQuestionAnswering"),
        ("ernie", "ErnieForQuestionAnswering"),
        ("ernie_m", "ErnieMForQuestionAnswering"),
        ("falcon", "FalconForQuestionAnswering"),
        ("flaubert", "FlaubertForQuestionAnsweringSimple"),
        ("fnet", "FNetForQuestionAnswering"),
        ("funnel", "FunnelForQuestionAnswering"),
        ("gpt2", "GPT2ForQuestionAnswering"),
        ("gpt_neo", "GPTNeoForQuestionAnswering"),
        ("gpt_neox", "GPTNeoXForQuestionAnswering"),
        ("gptj", "GPTJForQuestionAnswering"),
        ("ibert", "IBertForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
        ("led", "LEDForQuestionAnswering"),
        ("lilt", "LiltForQuestionAnswering"),
        ("longformer", "LongformerForQuestionAnswering"),
        ("luke", "LukeForQuestionAnswering"),
        ("lxmert", "LxmertForQuestionAnswering"),
        ("markuplm", "MarkupLMForQuestionAnswering"),
        ("mbart", "MBartForQuestionAnswering"),
        ("mega", "MegaForQuestionAnswering"),
        ("megatron-bert", "MegatronBertForQuestionAnswering"),
        ("mobilebert", "MobileBertForQuestionAnswering"),
        ("mpnet", "MPNetForQuestionAnswering"),
        ("mpt", "MptForQuestionAnswering"),
        ("mra", "MraForQuestionAnswering"),
        ("mt5", "MT5ForQuestionAnswering"),
        ("mvp", "MvpForQuestionAnswering"),
        ("nezha", "NezhaForQuestionAnswering"),
        ("nystromformer", "NystromformerForQuestionAnswering"),
        ("opt", "OPTForQuestionAnswering"),
        ("qdqbert", "QDQBertForQuestionAnswering"),
        ("reformer", "ReformerForQuestionAnswering"),
        ("rembert", "RemBertForQuestionAnswering"),
        ("roberta", "RobertaForQuestionAnswering"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForQuestionAnswering"),
        ("roc_bert", "RoCBertForQuestionAnswering"),
        ("roformer", "RoFormerForQuestionAnswering"),
        ("splinter", "SplinterForQuestionAnswering"),
        ("squeezebert", "SqueezeBertForQuestionAnswering"),
        ("t5", "T5ForQuestionAnswering"),
        ("umt5", "UMT5ForQuestionAnswering"),
        ("xlm", "XLMForQuestionAnsweringSimple"),
        ("xlm-roberta", "XLMRobertaForQuestionAnswering"),
        ("xlm-roberta-xl", "XLMRobertaXLForQuestionAnswering"),
        ("xlnet", "XLNetForQuestionAnsweringSimple"),
        ("xmod", "XmodForQuestionAnswering"),
        ("yoso", "YosoForQuestionAnswering"),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TapasForQuestionAnswering"),
    ]
)

MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("vilt", "ViltForQuestionAnswering"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassification"),
        ("bert", "BertForTokenClassification"),
        ("big_bird", "BigBirdForTokenClassification"),
        ("biogpt", "BioGptForTokenClassification"),
        ("bloom", "BloomForTokenClassification"),
        ("bros", "BrosForTokenClassification"),
        ("camembert", "CamembertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("data2vec-text", "Data2VecTextForTokenClassification"),
        ("deberta", "DebertaForTokenClassification"),
        ("deberta-v2", "DebertaV2ForTokenClassification"),
        ("distilbert", "DistilBertForTokenClassification"),
        ("electra", "ElectraForTokenClassification"),
        ("ernie", "ErnieForTokenClassification"),
        ("ernie_m", "ErnieMForTokenClassification"),
        ("esm", "EsmForTokenClassification"),
        ("falcon", "FalconForTokenClassification"),
        ("flaubert", "FlaubertForTokenClassification"),
        ("fnet", "FNetForTokenClassification"),
        ("funnel", "FunnelForTokenClassification"),
        ("gpt-sw3", "GPT2ForTokenClassification"),
        ("gpt2", "GPT2ForTokenClassification"),
        ("gpt_bigcode", "GPTBigCodeForTokenClassification"),
        ("gpt_neo", "GPTNeoForTokenClassification"),
        ("gpt_neox", "GPTNeoXForTokenClassification"),
        ("ibert", "IBertForTokenClassification"),
        ("layoutlm", "LayoutLMForTokenClassification"),
        ("layoutlmv2", "LayoutLMv2ForTokenClassification"),
        ("layoutlmv3", "LayoutLMv3ForTokenClassification"),
        ("lilt", "LiltForTokenClassification"),
        ("longformer", "LongformerForTokenClassification"),
        ("luke", "LukeForTokenClassification"),
        ("markuplm", "MarkupLMForTokenClassification"),
        ("mega", "MegaForTokenClassification"),
        ("megatron-bert", "MegatronBertForTokenClassification"),
        ("mobilebert", "MobileBertForTokenClassification"),
        ("mpnet", "MPNetForTokenClassification"),
        ("mpt", "MptForTokenClassification"),
        ("mra", "MraForTokenClassification"),
        ("nezha", "NezhaForTokenClassification"),
        ("nystromformer", "NystromformerForTokenClassification"),
        ("phi", "PhiForTokenClassification"),
        ("qdqbert", "QDQBertForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForTokenClassification"),
        ("roc_bert", "RoCBertForTokenClassification"),
        ("roformer", "RoFormerForTokenClassification"),
        ("squeezebert", "SqueezeBertForTokenClassification"),
        ("xlm", "XLMForTokenClassification"),
        ("xlm-roberta", "XLMRobertaForTokenClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
        ("xlnet", "XLNetForTokenClassification"),
        ("xmod", "XmodForTokenClassification"),
        ("yoso", "YosoForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "AlbertForMultipleChoice"),
        ("bert", "BertForMultipleChoice"),
        ("big_bird", "BigBirdForMultipleChoice"),
        ("camembert", "CamembertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("data2vec-text", "Data2VecTextForMultipleChoice"),
        ("deberta-v2", "DebertaV2ForMultipleChoice"),
        ("distilbert", "DistilBertForMultipleChoice"),
        ("electra", "ElectraForMultipleChoice"),
        ("ernie", "ErnieForMultipleChoice"),
        ("ernie_m", "ErnieMForMultipleChoice"),
        ("flaubert", "FlaubertForMultipleChoice"),
        ("fnet", "FNetForMultipleChoice"),
        ("funnel", "FunnelForMultipleChoice"),
        ("ibert", "IBertForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("luke", "LukeForMultipleChoice"),
        ("mega", "MegaForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("mpnet", "MPNetForMultipleChoice"),
        ("mra", "MraForMultipleChoice"),
        ("nezha", "NezhaForMultipleChoice"),
        ("nystromformer", "NystromformerForMultipleChoice"),
        ("qdqbert", "QDQBertForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMultipleChoice"),
        ("roc_bert", "RoCBertForMultipleChoice"),
        ("roformer", "RoFormerForMultipleChoice"),
        ("squeezebert", "SqueezeBertForMultipleChoice"),
        ("xlm", "XLMForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("xlm-roberta-xl", "XLMRobertaXLForMultipleChoice"),
        ("xlnet", "XLNetForMultipleChoice"),
        ("xmod", "XmodForMultipleChoice"),
        ("yoso", "YosoForMultipleChoice"),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertForNextSentencePrediction"),
        ("ernie", "ErnieForNextSentencePrediction"),
        ("fnet", "FNetForNextSentencePrediction"),
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),
        ("mobilebert", "MobileBertForNextSentencePrediction"),
        ("nezha", "NezhaForNextSentencePrediction"),
        ("qdqbert", "QDQBertForNextSentencePrediction"),
    ]
)

MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("audio-spectrogram-transformer", "ASTForAudioClassification"),
        ("data2vec-audio", "Data2VecAudioForSequenceClassification"),
        ("gemma", "GemmaForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("sew", "SEWForSequenceClassification"),
        ("sew-d", "SEWDForSequenceClassification"),
        ("unispeech", "UniSpeechForSequenceClassification"),
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForSequenceClassification"),
        ("wavlm", "WavLMForSequenceClassification"),
        ("whisper", "WhisperForAudioClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # Model for Connectionist temporal classification (CTC) mapping
        ("data2vec-audio", "Data2VecAudioForCTC"),
        ("hubert", "HubertForCTC"),
        ("mctct", "MCTCTForCTC"),
        ("sew", "SEWForCTC"),
        ("sew-d", "SEWDForCTC"),
        ("unispeech", "UniSpeechForCTC"),
        ("unispeech-sat", "UniSpeechSatForCTC"),
        ("wav2vec2", "Wav2Vec2ForCTC"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC"),
        ("wavlm", "WavLMForCTC"),
    ]
)

MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("data2vec-audio", "Data2VecAudioForAudioFrameClassification"),
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForAudioFrameClassification"),
        ("wavlm", "WavLMForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("data2vec-audio", "Data2VecAudioForXVector"),
        ("unispeech-sat", "UniSpeechSatForXVector"),
        ("wav2vec2", "Wav2Vec2ForXVector"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForXVector"),
        ("wavlm", "WavLMForXVector"),
    ]
)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Spectrogram mapping
        ("speecht5", "SpeechT5ForTextToSpeech"),
    ]
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Waveform mapping
        ("bark", "BarkModel"),
        ("musicgen", "MusicgenForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForTextToSpeech"),
        ("vits", "VitsModel"),
    ]
)

MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Image Classification mapping
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("blip", "BlipModel"),
        ("chinese_clip", "ChineseCLIPModel"),
        ("clip", "CLIPModel"),
        ("clipseg", "CLIPSegModel"),
    ]
)

MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        # Backbone mapping
        ("bit", "BitBackbone"),
        ("convnext", "ConvNextBackbone"),
        ("convnextv2", "ConvNextV2Backbone"),
        ("dinat", "DinatBackbone"),
        ("dinov2", "Dinov2Backbone"),
        ("focalnet", "FocalNetBackbone"),
        ("maskformer-swin", "MaskFormerSwinBackbone"),
        ("nat", "NatBackbone"),
        ("resnet", "ResNetBackbone"),
        ("swin", "SwinBackbone"),
        ("timm_backbone", "TimmBackbone"),
        ("vitdet", "VitDetBackbone"),
    ]
)

MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
    ]
)

MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "AlbertModel"),
        ("bert", "BertModel"),
        ("big_bird", "BigBirdModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("distilbert", "DistilBertModel"),
        ("electra", "ElectraModel"),
        ("flaubert", "FlaubertModel"),
        ("ibert", "IBertModel"),
        ("longformer", "LongformerModel"),
        ("mobilebert", "MobileBertModel"),
        ("mt5", "MT5EncoderModel"),
        ("nystromformer", "NystromformerModel"),
        ("reformer", "ReformerModel"),
        ("rembert", "RemBertModel"),
        ("roberta", "RobertaModel"),
        ("roberta-prelayernorm", "RobertaPreLayerNormModel"),
        ("roc_bert", "RoCBertModel"),
        ("roformer", "RoFormerModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("t5", "T5EncoderModel"),
        ("umt5", "UMT5EncoderModel"),
        ("xlm", "XLMModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
    ]
)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        ("swin2sr", "Swin2SRForImageSuperResolution"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)

MODEL_FOR_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES)
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES
)
MODEL_FOR_DEPTH_ESTIMATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_XVECTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES)

MODEL_FOR_BACKBONE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES)

MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)

MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)


class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING


class AutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING


class AutoModelForImageToImage(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING

# Private on purpose, the public class will add the deprecation warnings.
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING


class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING



class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING


class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING


class AutoModelForObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING


class AutoModelForDepthEstimation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING


class AutoModelForVideoClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING


class AutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


class AutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING


class AutoModelForAudioXVector(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING


class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING


class AutoModelForTextToWaveform(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING


class AutoBackbone(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_BACKBONE_MAPPING


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
