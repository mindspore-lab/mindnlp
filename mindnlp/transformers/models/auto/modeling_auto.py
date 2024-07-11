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
        ("beit", "BeitModel"),
        ("bert", "BertModel"),
        ("bert-generation", "BertGenerationEncoder"),
        ("bge-m3", "BgeM3Model"),
        ("big_bird", "BigBirdModel"),
        ("bigbird_pegasus", "BigBirdPegasusModel"),
        ("biogpt", "BioGptModel"),
        ("bit", "BitModel"),
        ("blenderbot", "BlenderbotModel"),
        ("blenderbot-small", "BlenderbotSmallModel"),
        ("blip", "BlipModel"),
        ("blip-2", "Blip2Model"),
        ("instructblip", "InstructBlipVisionModel"),
        ("bloom", "BloomModel"),
        ("bridgetower", "BridgeTowerModel"),
        ("bros", "BrosModel"),
        ("camembert", "CamembertModel"),
        ("canine", "CanineModel"),
        ("codegen", "CodeGenModel"),
        ("cohere", "CohereModel"),
        ("cogvlm", "CogVLMModel"),
        ("cpmant", "CpmAntModel"),
        ("ctrl", "CTRLModel"),
        ("cpmbee", "CpmBeeModel"),
        ("chatglm", "ChatGLMModel"),
        ("clap", "ClapModel"),
        ("clip", "CLIPModel"),
        ("clipseg", ("CLIPSegModel", "CLIPSegVisionModel")),
        ("clipseg_vision_model", "CLIPSegVisionModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("convbert", "ConvBertModel"),
        ("convnext", "ConvNextModel"),
        ("cvt", "CvtModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("data2vec-audio", "Data2VecAudioModel"),
        ("deit", "DeiTModel"),
        ("deberta", "DebertaModel"),
        ("decision_transformer", "DecisionTransformerModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("detr", "DetrModel"),
        ("dinov2", "Dinov2Model"),
        ("efficientformer", "EfficientFormerModel"),
        ("encodec", "EncodecModel"),
        ("esm", "EsmModel"),
        ("ernie", "ErnieModel"),
        ("ernie_m", "ErnieMModel"),
        ("falcon", "FalconModel"),
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("flava", "FlavaModel"),
        ("fnet","FNetModel"),
        ("focalnet", "FocalNetModel"),
        ("funnel", ("FunnelModel", "FunnelBaseModel")),
        ("gemma", "GemmaModel"),
        ("git", "GitModel"),
        ("gpt_bigcode", "GPTBigCodeModel"),
        ("gpt", "GPTModel"),
        ("gptJ", "GPTjModel"),
        ("gpt2", "GPT2Model"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseModel"),
        ("gpt_pangu", "GPTPanguModel"),
        ("groupvit", "GroupViTModel"),
        ("ibert", "IBertModel"),
        ("layoutlmv2", "LayoutLMv2Model"),
        ("longformer", "LongformerModel"),
        ("jetmoe", "JetMoEModel"),
        ("led", "LEDModel"),
        ("llama", "LlamaModel"),
        ("m2m_100", "M2M100Model"),
        ("lxmert", "LxmertModel"),
        ("mamba", "MambaModel"),
        ("marian", "MarianModel"),
        ("mask2former", "Mask2FormerModel"),
        ("mbart", "MBartModel"),
        ("mctct","MCTCTModel"),
        ("mgp-str", "MgpstrForSceneTextRecognition"),
        ("minicpm", "MiniCPMModel"),
        ("mistral", "MistralModel"),
        ("mixtral", "MixtralModel"),
        ("mobilenet_v1", "MobileNetV1Model"),
        ("mobilenet_v2", "MobileNetV2Model"),
        ("mvp", "MvpModel"),
        ("olmo", "OlmoModel"),
        ("owlv2", "Owlv2Model"),
        ("owlv2_vision_model", "Owlv2VisionModel"),
        ("owlv2_text_model", "Owlv2TextModel"),
        ("oneformer", "OneFormerModel"),
        ("openelm", "OpenELMModel"),
        ("opt", "OPTModel"),
        ("pegasus", "PegasusModel"),
        ("persimmon", "PersimmonModel"),
        ("phi", "PhiModel"),
        ("phi3", "Phi3Model"),
        ("plbart", "PLBartModel"),
        ("qdqbert", "QDQBertModel"),
        ("qwen2", "Qwen2Model"),
        ("qwen2_moe", "Qwen2MoeModel"),
        ("reformer", "ReformerModel"),
        ("rembert", "RemBertModel"),
        ("resnet", "ResNetModel"),
        ("roberta", "RobertaModel"),
        ("roc_bert", "RoCBertModel"),
        ("rwkv", "RwkvModel"),
        ("sam", "SamModel"),
        ("segformer", "SegformerModel"),
        ("speech_to_text", "Speech2TextModel"),
        ("speecht5", "SpeechT5Model"),
        ("stablelm", "StableLmModel"),
        ("splinter", "SplinterModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("starcoder2", "Starcoder2Model"),
        ("swiftformer", "SwiftFormerModel"),
        ("swin", "SwinModel"),
        ("switch_transformers", "SwitchTransformersModel"),
        ("t5", "T5Model"),
        ("tapas", "TapasModel"),
        ("time_series_transformer", "TimeSeriesTransformerModel"),
        ("timesformer", "TimesformerModel"),
        ("unispeech-sat", "UniSpeechSatModel"),
        ("univnet", "UnivNetModel"),
        ("videomae", "VideoMAEModel"),
        ("vilt", "ViltModel"),
        ("vision_text_dual_encoder", "VisionTextDualEncoderModel"),
        ("visual_bert", "VisualBertModel"),
        ("vit", "ViTModel"),
        ("vit_hybrid", "ViTHybridModel"),
        ("vit_msn", "ViTMSNModel"),
        ("vitdet", "VitDetModel"),
        ("whisper", "WhisperModel"),
        ("wav2vec2", "Wav2Vec2Model"),
        ("wavlm", "WavLMModel"),
        ("wav2vec2-bert", "Wav2Vec2BertModel"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerModel"),
        ("xclip", "XCLIPModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
        ("xlnet", "XLNetModel"),
        ("umt5", "UMT5Model"),
        ("xmod", "XmodModel"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "AlbertForPreTraining"),
        ("bert", "BertForPreTraining"),
        ("big_bird", "BigBirdForPreTraining"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("flava", "FlavaForPreTraining"),
        ("fnet", "FNetForPreTraining"),
        ("funnel", "FunnelForPreTraining"),
        ("gpt_pangu", "GPTPanguForCausalLM"),
        ("ibert", "IBertForMaskedLM"),
        ("lxmert", "LxmertForPreTraining"),
        ("mamba", "MambaForCausalLM"),
        ("minicpm", "MiniCPMForCausalLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("rwkv", "RwkvForCausalLM"),
        ("splinter", "SplinterForPreTraining"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("switch_transformers", "SwitchTransformersForConditionalGeneration"),
        ("roc_bert", "RoCBertForPreTraining"),
        ("unispeech-sat", "UniSpeechSatForPreTraining"),
        ("videomae", "VideoMAEForPreTraining"),
        ("visual_bert", "VisualBertForPreTraining"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForPreTraining"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("tapas", "TapasPreTrainedModel"),
        ("xmod", "XmodForMaskedLM"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "AlbertForMaskedLM"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("cpmbee", "CpmBeeForCausalLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("fnet", "FNetForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        ("git", "GitForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gpt_pangu", "GPTPanguForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("led", "LEDForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("mamba", "MambaForCausalLM"),
        ("marian", "MarianMTModel"),
        ("minicpm", "MiniCPMForCausalLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("reformer", "ReformerModelWithLMHead"),
        ("ibert", "IBertForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("roc_bert", "RoCBertForMaskedLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("switch_transformers", "SwitchTransformersForConditionalGeneration"),
        ("whisper", "WhisperForConditionalGeneration"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("tapas", "TapasForMaskedLM"),
        ("xmod", "XmodForMaskedLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("big_bird", "BigBirdForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cohere", "CohereForCausalLM"),
        ("cogvlm", "CogVLMForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("cpmbee", "CpmBeeForCausalLM"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_pangu", "GPTPanguForCausalLM"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("git", "GitForCausalLM"),
        ("jetmoe", "JetMoEForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("minicpm", "MiniCPMForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("musicgen", "MusicgenForCausalLM"),
        ("musicgen_melody", "MusicgenMelodyForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("olmo", "OlmoForCausalLM"),
        ("openelm", "OpenELMForCausalLM"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("qwen2_moe", "Qwen2MoeForCausalLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("stablelm", "StableLmForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("trocr", "TrOCRForCausalLM"),
        ("vilt", "ViltForMaskedLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("tapas", "TapasForMaskedLM"),
        ("xmod", "XmodForCausalLM"),
    ]
)


MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "DeiTForMaskedImageModeling"),
        # ("focalnet", "FocalNetForMaskedImageModeling"),
        # ("swin", "SwinForMaskedImageModeling"),
        # ("swinv2", "Swinv2ForMaskedImageModeling"),
        # ("vit", "ViTForMaskedImageModeling"),
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
        ("gptj", "GPTJForSequenceClassification"),
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
        ("vilt", "ViltForImagesAndTextClassification"),
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
        ("camembert", "CamembertForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("fnet", "FNetForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        {"ibert", "IBertForMaskedLM"},
        ("mvp", "MvpForConditionalGeneration"),
        ("rembert", "RemBertForMaskedLM"),
        ("reformer", "ReformerForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("roc_bert", "RoCBertForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("xmod", "XmodForMaskedLM"),
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
        ("chatglm", "ChatGLMForConditionalGeneration"),
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
        ("bigbird_pegasus", "BigBirdPegasusForSequenceClassification"),
        ("biogpt", "BioGptForSequenceClassification"),
        ("bloom", "BloomForSequenceClassification"),
        ("camembert", "CamembertForSequenceClassification"),
        ("canine", "CanineForSequenceClassification"),
        ("ctrl", "CTRLForSequenceClassification"),
        ("data2vec-text", "Data2VecTextForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("deberta-v2", "DebertaV2ForSequenceClassification"),
        ("distilbert", "DistilBertForSequenceClassification"),
        ("esm", "EsmForSequenceClassification"),
        ("falcon", "FalconForSequenceClassification"),
        ("fnet", "FNetForSequenceClassification"),
        ("funnel", "FunnelForSequenceClassification"),
        ("ibert", "IBertForSequenceClassification"),
        ("layoutlmv2", "LayoutLMv2ForSequenceClassification"),
        ("led", "LEDForSequenceClassification"),
        ("jetmoe", "JetMoEForSequenceClassification"),
        ("minicpm", "MiniCPMForSequenceClassification"),
        ("mistral", "MistralForSequenceClassification"),
        ("mixtral", "MixtralForSequenceClassification"),
        ("mvp", "MvpForSequenceClassification"),
        ("opt", "OPTForSequenceClassification"),
        ("persimmon", "PersimmonForSequenceClassification"),
        ("phi", "PhiForSequenceClassification"),
        ("phi3", "Phi3ForSequenceClassification"),
        ("qwen2", "Qwen2ForSequenceClassification"),
        ("qwen2_moe", "Qwen2MoeForSequenceClassification"),
        ("reformer", "ReformerForSequenceClassification"),
        ("rembert", "RemBertForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("squeezebert", "SqueezeBertForSequenceClassification"),
        ("roc_bert", "RoCBertForSequenceClassification"),
        ("squeezebert", "SqueezeBertForSequenceClassification"),
        ("stablelm", "StableLmForSequenceClassification"),
        ("starcoder2", "Starcoder2ForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForSequenceClassification"),
        ("xlnet", "XLNetForSequenceClassification"),
        ("tapas", "TapasForSequenceClassification"),
        ("xmod", "XmodForSequenceClassification"),
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
        ("blip", "BlipForQuestionAnswering"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("vilt", "ViltForQuestionAnswering"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
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
        ("persimmon", "PersimmonForTokenClassification"),
        ("phi", "PhiForTokenClassification"),
        ("phi3", "Phi3ForTokenClassification"),
        ("qdqbert", "QDQBertForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForTokenClassification"),
        ("roc_bert", "RoCBertForTokenClassification"),
        ("roformer", "RoFormerForTokenClassification"),
        ("squeezebert", "SqueezeBertForTokenClassification"),
        ("stablelm", "StableLmForTokenClassification"),
        ("vilt", "ViltForTokenClassification"),
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
        ("wav2vec2-bert", "Wav2Vec2BertForSequenceClassification"),
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
        ("wav2vec2-bert", "Wav2Vec2BertForCTC"),
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
        ("wav2vec2-bert", "Wav2Vec2BertForAudioFrameClassification"),
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
        ("wav2vec2-bert", "Wav2Vec2BertForXVector"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForXVector"),
        ("wavlm", "WavLMForXVector"),
    ]
)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Spectrogram mapping
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("speecht5", "SpeechT5ForTextToSpeech"),
    ]
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Waveform mapping
        ("bark", "BarkModel"),
        ("fastspeech2_conformer", "FastSpeech2ConformerWithHifiGan"),
        ("musicgen", "MusicgenForConditionalGeneration"),
        ("musicgen_melody", "MusicgenMelodyForConditionalGeneration"),
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
        ("instructblip", "InstructBlipVisionModel"),
    ]
)

MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        # Backbone mapping
        ("beit", "BeitBackbone"),
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
        ("donut-swin", "DonutSwinModel"),
        ("electra", "ElectraModel"),
        ("flaubert", "FlaubertModel"),
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

MODEL_FOR_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image mapping
        ("beit", "BeitModel"),
        ("bit", "BitModel"),
        ("conditional_detr", "ConditionalDetrModel"),
        ("convnext", "ConvNextModel"),
        ("convnextv2", "ConvNextV2Model"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("deformable_detr", "DeformableDetrModel"),
        ("deit", "DeiTModel"),
        ("deta", "DetaModel"),
        ("detr", "DetrModel"),
        ("dinat", "DinatModel"),
        ("dinov2", "Dinov2Model"),
        ("dpt", "DPTModel"),
        ("efficientformer", "EfficientFormerModel"),
        ("efficientnet", "EfficientNetModel"),
        ("focalnet", "FocalNetModel"),
        ("glpn", "GLPNModel"),
        ("imagegpt", "ImageGPTModel"),
        ("levit", "LevitModel"),
        ("mobilenet_v1", "MobileNetV1Model"),
        ("mobilenet_v2", "MobileNetV2Model"),
        ("mobilevit", "MobileViTModel"),
        ("mobilevitv2", "MobileViTV2Model"),
        ("nat", "NatModel"),
        ("poolformer", "PoolFormerModel"),
        ("pvt", "PvtModel"),
        ("regnet", "RegNetModel"),
        ("resnet", "ResNetModel"),
        ("segformer", "SegformerModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("swiftformer", "SwiftFormerModel"),
        ("swin", "SwinModel"),
        ("swin2sr", "Swin2SRModel"),
        ("swinv2", "Swinv2Model"),
        ("table-transformer", "TableTransformerModel"),
        ("timesformer", "TimesformerModel"),
        ("timm_backbone", "TimmBackbone"),
        ("van", "VanModel"),
        ("videomae", "VideoMAEModel"),
        ("vit", "ViTModel"),
        ("vit_hybrid", "ViTHybridModel"),
        ("vit_mae", "ViTMAEModel"),
        ("vit_msn", "ViTMSNModel"),
        ("vitdet", "VitDetModel"),
        ("vivit", "VivitModel"),
        ("yolos", "YolosModel"),
    ]
)

MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Do not add new models here, this class will be deprecated in the future.
        # Model for Image Segmentation mapping
        ("detr", "DetrForSegmentation"),
    ]
)

MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Semantic Segmentation mapping
        ("beit", "BeitForSemanticSegmentation"),
        ("data2vec-vision", "Data2VecVisionForSemanticSegmentation"),
        ("dpt", "DPTForSemanticSegmentation"),
        ("mobilenet_v2", "MobileNetV2ForSemanticSegmentation"),
        ("mobilevit", "MobileViTForSemanticSegmentation"),
        ("mobilevitv2", "MobileViTV2ForSemanticSegmentation"),
        ("segformer", "SegformerForSemanticSegmentation"),
        ("upernet", "UPerNetForSemanticSegmentation"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES
)
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES
)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)

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
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
)
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES
)

MODEL_FOR_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES
)
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES
)
MODEL_FOR_DEPTH_ESTIMATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES
)
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
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES
)
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
)
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_XVECTOR_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES
)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES
)

MODEL_FOR_BACKBONE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES
)

MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MASK_GENERATION_MAPPING_NAMES
)

MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES
)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES
)

MODEL_FOR_IMAGE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_MAPPING_NAMES
)

MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
)


class AutoModelForMaskGeneration(_BaseAutoModelClass):
    """
    Represents a class for generating masks automatically based on a given model.
    This class inherits functionality from the _BaseAutoModelClass, providing methods and attributes for mask generation.
    """

    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING


class AutoModelForTextEncoding(_BaseAutoModelClass):
    """
    The AutoModelForTextEncoding class represents a model for encoding text data.
    It is a subclass of the _BaseAutoModelClass and inherits its behavior and attributes.
    This class provides functionality for automatically encoding text data and can be used for various natural language processing tasks.
    """

    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING


class AutoModelForImageToImage(_BaseAutoModelClass):
    """
    Represents an automatic model for image-to-image tasks.

    This class inherits from the _BaseAutoModelClass and provides functionality for automatically selecting and
    using models for image-to-image tasks.
    It encapsulates the logic for model selection, configuration, and inference for image-to-image transformation tasks.
    Users can leverage this class to streamline the process of selecting and using the most suitable model
    for their specific image-to-image transformation needs.

    Attributes:
        _BaseAutoModelClass: The base class providing foundational functionality for automatic model selection and usage.

    Note:
        This class is designed to streamline the process of model selection and utilization for image-to-image transformation tasks.
        It encapsulates the underlying complexities of model selection and configuration, enabling users to focus on
        the specifics of their image transformation requirements.

    """

    _model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING


class AutoModel(_BaseAutoModelClass):
    """
    Represents an automated model for performing various tasks related to vehicle models.

    This class inherits from _BaseAutoModelClass and provides functionalities for managing and analyzing vehicle models in an automated manner.
    It includes methods for data processing, model training, evaluation, and prediction.
    The AutoModel class serves as a foundation for building automated systems that work with vehicle models efficiently.
    """

    _model_mapping = MODEL_MAPPING


class AutoModelForPreTraining(_BaseAutoModelClass):
    """
    Represents a Python class for an auto model used for pre-training natural language processing (NLP) tasks.
    This class inherits functionality from the _BaseAutoModelClass, providing a foundation for pre-training NLP models.
    It encapsulates methods and attributes specific to pre-training tasks, allowing for efficient development and training of NLP models.
    """

    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


# Private on purpose, the public class will add the deprecation warnings.


class _AutoModelWithLMHead(_BaseAutoModelClass):
    """
    This class represents an automatic model with a language modeling head, implementing the functionality of generating text based on given input.

    It inherits from the '_BaseAutoModelClass' class, which provides the base functionality for automatic models.

    Attributes:
        model_name_or_path (str): The name or path of the pre-trained model to be used.
        config (PretrainedConfig): The configuration object for the model.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization and encoding of text inputs.
        model (PreTrainedModel): The pre-trained model used for generating text.

    Methods:
        generate: Generates text based on the given input using the language modeling head of the model.
        forward: Performs a forward pass through the model to generate text.
        prepare_inputs_for_generation: Prepares the input data for generation by the model.
        save_pretrained: Saves the model and its configuration to the specified directory.
        from_pretrained: Creates a new instance of the class from a pre-trained model.
        __call__: Invokes the model to generate text based on the given input.

    Note:
        This class serves as a convenient interface to easily generate text using a pre-trained language model with a language modeling head.
        It provides methods for generating text as well as saving and loading the model.

    Example usage:
        ```python
        >>> model = _AutoModelWithLMHead.from_pretrained('bert-base-uncased')
        >>> inputs = "Once upon a time"
        >>> generated_text = model.generate(inputs)
        ```
    """

    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


class AutoModelForCausalLM(_BaseAutoModelClass):
    """
    Represents a Python class for an automatic model tailored for Causal Language Modeling tasks.
    This class inherits from the _BaseAutoModelClass and provides functionality for training, fine-tuning, and utilizing models for causal language modeling tasks.
    It includes methods for loading pre-trained models, generating text sequences, and evaluating model performance.
    """

    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


class AutoModelForMaskedLM(_BaseAutoModelClass):
    """
    Represents a class for automatically generating masked language model outputs based on a pre-trained model.

    This class serves as a specialized extension of the _BaseAutoModelClass,
    inheriting its core functionality and adding specific methods and attributes tailored for masked language model tasks.
    It provides a convenient interface for utilizing pre-trained language models to predict masked tokens
    within a given input sequence.
    """

    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    """
    Represents a class for automatic generation of models for sequence-to-sequence language modeling tasks.
    This class inherits functionality from the _BaseAutoModelClass, providing a base for creating and customizing
    sequence-to-sequence language models for various natural language processing applications.
    """

    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    """
    The 'AutoModelForSequenceClassification' class represents an automatic model for sequence classification tasks in Python.
    This class inherits functionality from the '_BaseAutoModelClass' class and provides a high-level interface for
    creating and utilizing pre-trained models for sequence classification tasks.
    """

    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    """
    This class represents an automatic model for question answering in Python. It is a subclass of the _BaseAutoModelClass,
    which provides a base implementation for automatic models.

    The AutoModelForQuestionAnswering class is designed to handle the task of question answering,
    where given a question and a context, it predicts the answer within the given context.
    It leverages pre-trained models and fine-tuning techniques to achieve high accuracy and performance.

    Attributes:
        model_name_or_path (str): The name or path of the pre-trained model to be used for question answering.
        config (AutoConfig): The configuration object that holds the model's configuration settings.
        tokenizer (PreTrainedTokenizer): The tokenizer used to preprocess input data for the model.
        model (PreTrainedModel): The pre-trained model for question answering.

    Methods:
        from_pretrained: Class method that loads a pre-trained model and returns an instance of the AutoModelForQuestionAnswering class.
        forward: Performs forward pass through the model given input IDs and other optional arguments, and returns the predicted answer.
        save_pretrained: Saves the model and its configuration to the specified directory for future use.
        from_config: Class method that creates an instance of the AutoModelForQuestionAnswering class from a provided configuration object.
        resize_token_embeddings: Resizes the token embeddings of the model to match the new number of tokens.

    Example:
        ```python
        >>> # Instantiate the AutoModelForQuestionAnswering class
        >>> model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
        ...
        >>> # Perform question answering
        >>> question = "What is the capital of France?"
        >>> context = "Paris is the capital of France."
        >>> input_ids = tokenizer.encode(question, context)
        >>> answer = model.forward(input_ids)
        ...
        >>> # Save the model
        >>> model.save_pretrained('models/qa_model')
        ...
        >>> # Load the saved model
        >>> loaded_model = AutoModelForQuestionAnswering.from_pretrained('models/qa_model')
        ```

    Note:
        The AutoModelForQuestionAnswering class is built on top of the transformers library,
        which provides a wide range of pre-trained models for various NLP tasks.
        It is recommended to refer to the transformers documentation for more details on using this class and customizing its behavior.
    """

    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    """
    AutoModelForTableQuestionAnswering is a Python class that represents a model for table-based question answering tasks.
    This class inherits from the _BaseAutoModelClass, providing functionality for processing and generating answers for questions related to tables.

    This class encapsulates the necessary methods and attributes for initializing, loading,
    and utilizing a pre-trained model for table question answering.
    It provides an interface for encoding table data and questions, and generating answers
    based on the learned patterns and representations.

    The AutoModelForTableQuestionAnswering class is designed to be flexible and customizable,
    allowing users to fine-tune and adapt the model to specific table question answering tasks.
    It serves as a high-level abstraction for working with table-based question answering models,
    enabling seamless integration into various applications and workflows.

    Users can leverage the capabilities of this class to efficiently handle table question answering tasks,
    benefiting from the underlying mechanisms for processing and interpreting tabular data in the context
    of natural language questions.
    The class facilitates the integration of table question answering functionality into larger projects,
    providing a powerful and efficient solution for handling such tasks within a Python environment.
    """

    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    """
    Represents a specialized model class for visual question answering (VQA) tasks.

    This class serves as an extension of the _BaseAutoModelClass and provides functionality tailored specifically
    for visual question answering applications.
    It encapsulates the necessary components and methods for processing both visual and textual inputs to
    generate accurate answers to questions related to images. Users can leverage the capabilities of this class to
    build, train, and deploy VQA models with ease.

    Attributes:
        Inherits from _BaseAutoModelClass: A base class that defines essential attributes and methods for auto-generated model classes.
        Additional attributes specific to visual question answering tasks may be present within this class.

    Methods:
        Specific methods for processing visual data, textual data, and combining them to produce answers to given questions.
        Utility functions for preprocessing input data, handling model inference, and post-processing the output for interpretation.
        Customizable parameters and settings to fine-tune the model's behavior for different VQA scenarios.

    Usage:
        Instantiate an object of AutoModelForVisualQuestionAnswering to access its VQA-specific functionalities
        and utilize them in developing VQA solutions. Users can extend and customize the class to adapt
        to different datasets and requirements, enhancing the model's performance on varying VQA tasks.

    Note:
        It is recommended to refer to the documentation of _BaseAutoModelClass for general information on inherited attributes and methods.

    For detailed information on the implementation and usage of AutoModelForVisualQuestionAnswering, please refer to the official documentation or codebase.
    """

    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING


class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    """
    This class represents an auto model for document question answering tasks.
    It inherits from the _BaseAutoModelClass, providing functionalities for processing text input
    and generating answers to questions based on the provided document context.
    """

    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


class AutoModelForTokenClassification(_BaseAutoModelClass):
    """
    AutoModelForTokenClassification is a class that represents an automatic model for token classification in Python.
    It inherits from _BaseAutoModelClass and provides functionality for token classification tasks.
    This class is designed to be used with pre-trained models and offers methods for token classification tasks,
    such as named entity recognition and part-of-speech tagging.
    """

    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    """
    Represents a class for automatically generating a model for multiple choice tasks.

    This class inherits from the _BaseAutoModelClass and provides functionality for creating a model specifically
    designed for handling multiple choice questions.
    It encapsulates the logic and operations required for training and inference on multiple choice datasets.

    The AutoModelForMultipleChoice class offers a set of methods and attributes for
    fine-tuning, evaluating, and utilizing the model for multiple choice tasks.
    It leverages the underlying architecture and components inherited from the _BaseAutoModelClass
    while adding specific functionality tailored to the requirements of multiple choice scenarios.

    Users can instantiate objects of this class to create, customize, and deploy models for multiple choice tasks,
    enabling seamless integration of machine learning capabilities into applications and workflows
    dealing with multiple choice question answering.
    """

    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    """
    A class representing an autoencoder model for next sentence prediction.

    This class inherits from _BaseAutoModelClass and provides a pre-trained model for next sentence prediction tasks.
    It can be used to generate predictions for whether a given pair of sentences are likely to be consecutive in a text sequence.

    Attributes:
        config (AutoConfig): The configuration class used to instantiate the model.
        base_model_prefix (str): The prefix for the base model.

    """

    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    """
    This class represents an automatic model for zero-shot image classification in Python.

    The 'AutoModelForZeroShotImageClassification' class is a subclass of the '_BaseAutoModelClass' class,
    which provides a base implementation for automatic models.
    It is designed specifically for zero-shot image classification tasks,
    where images are classified into predefined classes based on their visual content.

    The class encapsulates the necessary functionality to automatically train, evaluate, and use a model for zero-shot image classification.
    It includes methods for data preprocessing, model training, hyperparameter tuning, model evaluation, and inference.
    Additionally, it provides convenient interfaces to load and save trained models, as well as to fine-tune pre-trained models for specific tasks.

    One of the key features of this class is its ability to handle zero-shot learning,
    where the model can classify images into classes that were not seen during training.
    This is achieved through the use of semantic embeddings or textual descriptions associated with each class.
    By leveraging the semantic information, the model can make predictions for unseen classes based on their similarity to the seen classes.

    To use this class, you can instantiate an object of the 'AutoModelForZeroShotImageClassification' class
    and provide the necessary parameters, such as the training data, class labels, and hyperparameters.
    Once the model is trained, you can use it to classify new images by calling the appropriate methods.

    Note that this class assumes the input images are in a suitable format and the class labels or
    semantic embeddings are provided for zero-shot learning.
    It is recommended to preprocess the data and ensure the proper format before using this class.

    For more details on how to use this class, please refer to the documentation and examples provided with the package.

    Attributes:
        None.

    Methods:
        __init__(self, *args, **kwargs): Initializes the 'AutoModelForZeroShotImageClassification' object with the given parameters.
        preprocess_data(self, data): Preprocesses the input data, such as resizing images, normalizing pixel values, etc.
        train(self, train_data, train_labels, **kwargs): Trains the model using the provided training data and labels.
        tune_hyperparameters(self, train_data, train_labels, **kwargs): Performs hyperparameter tuning to optimize the model's performance.
        evaluate(self, test_data, test_labels): Evaluates the trained model on the provided test data and labels.
        classify(self, images): Classifies the given images into their respective classes.
        save_model(self, filepath): Saves the trained model to the specified filepath.
        load_model(self, filepath): Loads a pre-trained model from the specified filepath.
        fine_tune(self, new_data, new_labels): Fine-tunes the pre-trained model on new data and labels for transfer learning.

    """

    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    """
    This class represents an automatic model for universal segmentation in Python.
    It is a subclass of the _BaseAutoModelClass, which provides a base implementation for automatic models.

    Universal segmentation is the task of dividing an input sequence into meaningful segments or units.
    The AutoModelForUniversalSegmentation class encapsulates the functionality required to automatically
    train and evaluate models for this task.

    Attributes:
        model_name_or_path (str): The pre-trained model name or path.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input sequences.
        model (AutoModel): The underlying pre-trained model for universal segmentation.
        device (str): The device (e.g., 'cpu', 'cuda') on which the model is loaded.
        config (AutoConfig): The configuration for the pre-trained model.

    Methods:
        __init__: Initializes a new instance of AutoModelForUniversalSegmentation.
        train: Trains the model using the provided training dataset and evaluates it on the evaluation dataset. Additional
            keyword arguments can be passed to customize the training process.
        predict: Predicts the segments for the given input sequence using the trained model.
        save_model: Saves the trained model to the specified output directory.
        load_model: Loads a pre-trained model from the specified path.

    Inherited Attributes:
        - base_attribute_1 (type): Description of the attribute inherited from _BaseAutoModelClass.
        - base_attribute_2 (type): Description of another attribute inherited from _BaseAutoModelClass.

    Inherited Methods:
        - base_method_1: Description of the method inherited from _BaseAutoModelClass.
        - base_method_2: Description of another method inherited from _BaseAutoModelClass.

    Note:
        This class assumes that the input sequences are already tokenized and encoded using the tokenizer.
        The predict method returns a list of Segment objects, where each Segment represents a segment of the input sequence.

    Example:
        ```python
        >>> model = AutoModelForUniversalSegmentation(model_name_or_path='bert-base-uncased')
        >>> model.train(train_dataset, eval_dataset)
        >>> segments = model.predict('This is an example sentence.')
        >>> model.save_model('output/model')
        >>> model.load_model('output/model')
        ```

    For more details on the usage and available models, refer to the documentation and examples provided with this class.
    """

    _model_mapping = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING


class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    """ "
    Represents a class for automatic model generation for instance segmentation tasks.

    This class provides functionality for automatically generating models tailored for instance segmentation,
    which is the task of identifying and delineating individual objects within an image. The class
    inherits from _BaseAutoModelClass, providing a base for creating specialized instance segmentation models.

    Attributes:
        _BaseAutoModelClass:
            The base class for automatic model generation, providing foundational functionality for creating custom models.

    Methods:
        (Include any specific methods and their functionality here)

    Usage:
        (Include any usage examples or guidelines here)
    """

    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING


class AutoModelForObjectDetection(_BaseAutoModelClass):
    """
    Represents a class for automatic model selection and configuration for object detection tasks.

    This class inherits from _BaseAutoModelClass and provides methods for automatically selecting and configuring
    a model for object detection tasks based on input data and performance metrics.

    The AutoModelForObjectDetection class encapsulates functionality for model selection, hyperparameter optimization,
    and model evaluation, making it a convenient and efficient tool for automating the process
    of model selection and configuration for object detection applications.
    """

    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    """
    The AutoModelForZeroShotObjectDetection class represents an automatic model for zero-shot object detection.
    It inherits from the _BaseAutoModelClass and provides functionality for detecting objects in images without
    the need for training on specific object classes.
    """

    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING


class AutoModelForDepthEstimation(_BaseAutoModelClass):
    """
    Represents a specialized class for automatically generating models for depth estimation tasks.
    This class inherits functionality from the _BaseAutoModelClass to provide a base structure for creating depth estimation models.
    """

    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING


class AutoModelForVideoClassification(_BaseAutoModelClass):
    """
    Represents a class for automatic model selection for video classification tasks.

    This class serves as a specialized implementation for selecting the optimal model for video classification
    based on specified criteria.
    It inherits functionality from the _BaseAutoModelClass, providing a foundation for automatic model selection
    with a focus on video classification tasks.
    """

    _model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING


class AutoModelForVision2Seq(_BaseAutoModelClass):
    """
    AutoModelForVision2Seq is a Python class that represents an automatic model for vision-to-sequence tasks.
    This class inherits from the _BaseAutoModelClass, providing additional functionalities specific to vision-to-sequence tasks.

    Attributes:
        model_name_or_path (str): The pre-trained model name or path.
        config (AutoConfig): The configuration class for the model.
        feature_extractor (FeatureExtractor): The feature extractor for the model.
        encoder (Encoder): The encoder module for the model.
        decoder (Decoder): The decoder module for the model.
        tokenizer (Tokenizer): The tokenizer used for tokenization tasks.
        vision_embedding (nn.Module): The module responsible for embedding the visual features.
        sequence_embedding (nn.Module): The module responsible for embedding the sequence input.
        classifier (nn.Module): The classifier module for the model.

    Methods:
        forward: Performs a forward pass through the model, taking visual features and sequence input as input.
        encode_visual_features: Encodes the visual features using the vision_embedding module.
        encode_sequence_input: Encodes the sequence input using the sequence_embedding module.
        generate: Generates a sequence output based on the encoded visual features and sequence input.
        save_pretrained: Saves the model and its configuration to the specified path.
        from_pretrained: Loads a pre-trained model and its configuration from the specified path.
        resize_token_embeddings: Resizes the token embeddings of the tokenizer.

    Note:
        AutoModelForVision2Seq is designed to be used for vision-to-sequence tasks, where the model takes in visual features
        and sequence input, and generates a sequence output. It provides an interface for loading pre-trained models,
        performing inference, and fine-tuning on custom datasets. The class inherits from _BaseAutoModelClass to leverage
        the shared functionalities across different automatic models.

    Example:
        ```python
        >>> model = AutoModelForVision2Seq.from_pretrained('model_name')
        >>> outputs = model.forward(visual_features, sequence_input)
        ```
    """

    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


class AutoModelForAudioClassification(_BaseAutoModelClass):
    """
    This class represents an automatic model for audio classification tasks. It inherits from the _BaseAutoModelClass,
    providing functionalities for processing audio data and making predictions for classification.
    The class provides methods and attributes for training, evaluating, and using the model for audio classification tasks.
    """

    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


class AutoModelForCTC(_BaseAutoModelClass):
    """
    This class represents an automatic model for Connectionist Temporal Classification (CTC) tasks in Python.

    The 'AutoModelForCTC' class inherits from the '_BaseAutoModelClass' class and provides a high-level interface for
    training, evaluating, and using CTC models.
    CTC is a type of sequence transduction problem where the input and output sequences have different lengths.
    It is commonly used in speech recognition and handwriting recognition tasks.

    The 'AutoModelForCTC' class encapsulates all the necessary components for building, training, and using CTC models.
    It provides methods for loading data, preprocessing, model architecture selection,
    hyperparameter tuning, training, evaluation, and inference.
    It also supports various options for customization and fine-tuning.

    To use this class, instantiate an object of the 'AutoModelForCTC' class and specify the desired configuration.
    Then, call the appropriate methods to perform the desired operations.
    The class takes care of handling the complexities of CTC model training and usage,
    allowing users to focus on their specific tasks.

    Note that this class assumes a basic understanding of CTC and neural networks.
    It is recommended to have prior knowledge of deep learning concepts before using this class.
    Detailed information about CTC and neural networks can be found in relevant literature and online resources.

    For more details on the available methods and functionalities of the 'AutoModelForCTC' class, refer to the documentation and code comments.

    """

    _model_mapping = MODEL_FOR_CTC_MAPPING


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    """
    This class represents an automatic model for speech sequence-to-sequence (Seq2Seq) tasks in Python.

    The 'AutoModelForSpeechSeq2Seq' class is a subclass of the '_BaseAutoModelClass' and provides a pre-trained model
    for speech-to-text conversion tasks. It is designed to simplify the process of building and training
    speech Seq2Seq models by providing a high-level interface for developers.

    The class inherits all the properties and methods from the '_BaseAutoModelClass', which includes functionalities
    for model configuration, training, and inference.
    It also contains additional methods specific to speech Seq2Seq tasks,
    such as audio preprocessing, text tokenization, and attention mechanisms.

    To use this class, instantiate an object of the 'AutoModelForSpeechSeq2Seq' class
    and provide the necessary parameters for model initialization.
    Once the model is initialized, you can use the provided methods to train the model on
    your speech dataset or perform inference on new speech inputs.

    Note that this class assumes the availability of a pre-trained model for speech Seq2Seq tasks.
    If you don't have a pre-trained model, you can refer to the documentation for the '_BaseAutoModelClass' on how
    to train a model from scratch.

    Example:
        ```python
        >>> model = AutoModelForSpeechSeq2Seq(model_name='speech_model', num_layers=3)
        >>> model.train(dataset)
        >>> transcriptions = model.transcribe(audio_inputs)
        ```

    Please refer to the documentation of the '_BaseAutoModelClass' for more details on general model functionalities and best practices for training and fine-tuning models.
    """

    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    """
    Represents an auto model for audio frame classification tasks.

    This class serves as a template for creating neural network models specifically designed for audio frame classification.
    It inherits functionality from the _BaseAutoModelClass, providing a foundation for implementing automatic model selection and configuration.

    Attributes:
        Inherited attributes from _BaseAutoModelClass

    Methods:
        Inherited methods from _BaseAutoModelClass
        Additional methods for audio frame classification tasks

    This class is intended to be extended and customized for specific audio classification projects,
    allowing for efficient development and experimentation in the audio signal processing domain.
    """

    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING


class AutoModelForAudioXVector(_BaseAutoModelClass):
    """
    The 'AutoModelForAudioXVector' class is a specialized class for automatic audio feature extraction using x-vectors.
    It is designed to provide a convenient interface for extracting audio features and performing various machine
    learning tasks using the x-vector representation.

    This class inherits from the '_BaseAutoModelClass', which provides the basic functionality for automatic feature extraction.
    By inheriting from this base class, the 'AutoModelForAudioXVector' class gains access to common methods and attributes
    required for audio feature extraction and machine learning.

    The 'AutoModelForAudioXVector' class encapsulates the logic and algorithms necessary for extracting x-vector features
    from audio data.
    It provides methods for loading audio files, preprocessing the audio data, and extracting x-vectors using a pre-trained model.

    One of the key features of the 'AutoModelForAudioXVector' class is its ability to perform various
    machine learning tasks using the extracted x-vectors.
    It includes methods for tasks such as speaker identification, speaker verification, and speech recognition.
    These methods leverage the power of the x-vector representation to achieve accurate results.

    Overall, the 'AutoModelForAudioXVector' class is a powerful tool for automatic audio feature extraction using x-vectors.
    It simplifies the process of extracting and working with x-vector features, enabling users to focus on
    their specific machine learning tasks without having to worry about the underlying implementation details.
    """

    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING


class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    """
    Represents a Python class for generating spectrograms from text using an auto model for text-to-spectrogram conversion.
    This class inherits from the _BaseAutoModelClass, providing additional functionality and customization options
    for text-to-spectrogram processing.

    The AutoModelForTextToSpectrogram class encapsulates the necessary methods and attributes for processing text inputs
    and generating corresponding spectrograms.
    It leverages the functionalities inherited from the _BaseAutoModelClass and extends them with specific capabilities
    tailored for the text-to-spectrogram transformation.

    This class serves as a powerful tool for converting textual data into visual representations in the form of spectrograms,
    enabling advanced analysis and visualization of linguistic patterns and acoustic features.
    By utilizing the AutoModelForTextToSpectrogram, users can efficiently process text inputs and obtain corresponding spectrogram outputs,
    facilitating a wide range of applications in fields such as natural language processing, speech recognition, and audio processing.

    Note:
        Please refer to the _BaseAutoModelClass documentation for inherited methods and attributes.
    """

    _model_mapping = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING


class AutoModelForTextToWaveform(_BaseAutoModelClass):
    """
    AutoModelForTextToWaveform is a Python class that represents an automatic model for converting text to waveform data.
    This class inherits from the _BaseAutoModelClass, which provides a base implementation for automatic models.

    The AutoModelForTextToWaveform class is specifically designed for processing text and generating corresponding waveform data.
    It leverages various natural language processing techniques and audio generation algorithms to achieve this functionality.

    Attributes:
        model_name_or_path (str): The name or path of the pre-trained model to be used for text-to-waveform conversion.
        tokenizer (Tokenizer): An instance of the Tokenizer class used for tokenizing text input.
        audio_generator (AudioGenerator): An instance of the AudioGenerator class used for generating waveform data from tokenized text.

    Methods:
        __init__: Initializes a new instance of the AutoModelForTextToWaveform class with the specified pre-trained model.
        preprocess_text: Preprocesses the input text by tokenizing and applying any necessary transformations.
        generate_waveform: Generates waveform data for the given input text using the pre-trained model and audio generation techniques.
        save_model: Saves the current model and associated resources to the specified directory.
        load_model: Loads a pre-trained model and associated resources from the specified directory.

    Example:
        ```python
        >>> # Initialize an AutoModelForTextToWaveform instance with a pre-trained model
        >>> model = AutoModelForTextToWaveform('model_name')
        ...
        >>> # Preprocess text and generate waveform data
        >>> preprocessed_text = model.preprocess_text('Hello, how are you?')
        >>> waveform_data = model.generate_waveform(preprocessed_text)
        ...
        >>> # Save and load the model
        >>> model.save_model('saved_model')
        >>> model.load_model('saved_model')
        ```
    """

    _model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING


class AutoBackbone(_BaseAutoModelClass):
    """
    Represents an AutoBackbone Python class that inherits from _BaseAutoModelClass.

    The AutoBackbone class is a specialized class that provides functionality for generating automatic backbones in Python.
    It is designed to be used as a base class for creating custom backbone models.
    The class inherits from the _BaseAutoModelClass, which provides common functionality for all auto models.

    Attributes:
        None

    Methods:
        None

    Usage:
        To use the AutoBackbone class, simply create a new instance and customize it as needed.
        As a base class, it does not provide any specific attributes or methods.
        Its purpose is to serve as a starting point for creating custom backbone models.

    Inheritance:
        The AutoBackbone class inherits from the _BaseAutoModelClass, which is a base class for all auto models.
        This allows the AutoBackbone class to leverage common functionality and adhere to a consistent interface
        across different auto models.

    Note:
        It is recommended to review the documentation of the _BaseAutoModelClass for a better understanding of
        the common functionality and attributes available in the AutoBackbone class.

    """

    _model_mapping = MODEL_FOR_BACKBONE_MAPPING


class AutoModelWithLMHead(_AutoModelWithLMHead):
    """
    This class represents a deprecated version of `AutoModelWithLMHead` and will be removed in a future version.
    It is recommended to use `AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM`
    for masked language models, and `AutoModelForSeq2SeqLM` for encoder-decoder models instead.

    Inherits from: `_AutoModelWithLMHead`

    Methods:
        from_config:

           - This method is used to create an instance of the class from a configuration object.
           - Parameters:

               - config: The configuration object used to initialize the class instance.
           - Returns:

               - An instance of the class.

        from_pretrained:

           - This method is used to create an instance of the class from a pretrained model.
           - Parameters:

               - pretrained_model_name_or_path: The name or path of the pretrained model.
               - *model_args: Additional model-specific arguments.
               - **kwargs: Additional keyword arguments.

           - Returns:

               - An instance of the class.

    Note:
        This class is deprecated and should not be used in new implementations.
        Please refer to the appropriate classes mentioned above based on your specific use case.
    """

    @classmethod
    def from_config(cls, config):
        """
        This method creates an instance of the 'AutoModelWithLMHead' class based on the provided 'config' parameter.

        Args:
            cls (class): The class method is called from.
            config (object): The configuration object used to create the instance. It contains the necessary information to initialize the model.

        Returns:
            None.

        Raises:
            FutureWarning: If the 'AutoModelWithLMHead' class is used, a warning is issued to inform the user
            that it is deprecated and will be removed in a future version.
            Users are advised to use 'AutoModelForCausalLM' for causal language models, 'AutoModelForMaskedLM'
            for masked language models, and 'AutoModelForSeq2SeqLM' for encoder-decoder models instead.
        """
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Loads a pretrained model from a given model name or path.

        Args:
            cls (class): The class itself.
            pretrained_model_name_or_path (str): The name or path of the pretrained model.
                This can be a local path or a URL to a pretrained model repository.

        Returns:
            None

        Raises:
            FutureWarning: If using the deprecated class `AutoModelWithLMHead`.
                Please use `AutoModelForCausalLM` for causal language models,
                `AutoModelForMaskedLM` for masked language models, and
                `AutoModelForSeq2SeqLM` for encoder-decoder models.
        """
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
