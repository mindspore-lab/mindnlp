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
"""Auto Config class."""

import importlib
import re
import warnings
from collections import OrderedDict
from typing import List, Union

from mindnlp.configs import CONFIG_NAME
from mindnlp.transformers.configuration_utils import PretrainedConfig
from mindnlp.utils import logging

logger = logging.get_logger(__name__)

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("albert", "AlbertConfig"),
        ("align", "AlignConfig"),
        ("altclip", "AltCLIPConfig"),
        ("audio-spectrogram-transformer", "ASTConfig"),
        ("autoformer", "AutoformerConfig"),
        ("bark", "BarkConfig"),
        ("bart", "BartConfig"),
        ("beit", "BeitConfig"),
        ("bert", "BertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("bge-m3", "BgeM3Config"),
        ("big_bird", "BigBirdConfig"),
        ("bigbird_pegasus", "BigBirdPegasusConfig"),
        ("biogpt", "BioGptConfig"),
        ("bit", "BitConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("blip", "BlipConfig"),
        ("blip-2", "Blip2Config"),
        ("bloom", "BloomConfig"),
        ("bridgetower", "BridgeTowerConfig"),
        ("bros", "BrosConfig"),
        ("camembert", "CamembertConfig"),
        ("canine", "CanineConfig"),
        ("chatglm", "ChatGLMConfig"),
        ("clap", "ClapConfig"),
        ("clip", "CLIPConfig"),
        ("clipseg", "CLIPSegConfig"),
        ("clipseg_vision_model", "CLIPSegVisionConfig"),
        ("clip_vision_model", "CLIPVisionConfig"),
        ("codegen", "CodeGenConfig"),
        ("cohere", "CohereConfig"),
        ("cogvlm", "CogVLMConfig"),
        ("convbert", "ConvBertConfig"),
        ("convnext", "ConvNextConfig"),
        ("cpmant", "CpmAntConfig"),
        ("ctrl", "CTRLConfig"),
        ("cpmbee", "CpmBeeConfig"),
        ("cvt", "CvtConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
        ("data2vec-audio","Data2VecAudioConfig"),
        ("deit", "DeiTConfig"),
        ("deberta", "DebertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("detr", "DetrConfig"),
        ("distilbert", "DistilBertConfig"),
        ("donut-swin", "DonutSwinConfig"),
        ("efficientformer", "EfficientFormerConfig"),
        ("encodec", "EncodecConfig"),
        ("esm", "EsmConfig"),
        ("falcon", "FalconConfig"),
        ("flava", "FlavaConfig"),
        ("fnet", "FNetConfig"),
        ("focalnet", "FocalNetConfig"),
        ("funnel", "FunnelConfig"),
        ("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
        ("gemma", "GemmaConfig"),
        ("git", "GitConfig"),
        ("gpt2", "GPT2Config"),
        ("gpt_bigcode", "GPTBigCodeConfig"),
        ("gptj", "GPTJConfig"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("gpt_pangu", "GPTPanguConfig"),
        ("groupvit", "GroupViTConfig"),
        ("hubert", "HubertConfig"),
        ("ibert", "IBertConfig"),
        ("instructblip", "InstructBlipConfig"),
        ("jetmoe", "JetMoEConfig"),
        ("led", "LEDConfig"),
        ("llama", "LlamaConfig"),
        ("llava", "LlavaConfig"),
        ("llava_next", "LlavaNextConfig"),
        ("m2m_100", "M2M100Config"),
        ("lxmert", "LxmertConfig"),
        ("mamba", "MambaConfig"),
        ("marian", "MarianConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("mbart", "MBartConfig"),
        ("mctct", "MCTCTConfig"),
        ("minicpm", "MiniCPMConfig"),
        ("mistral", "MistralConfig"),
        ("mixtral", "MixtralConfig"),
        ("mobilevit", "MobileViTConfig"),
        ("mobilenet_v1", "MobileNetV1Config"),
        ("mobilenet_v2", "MobileNetV2Config"),
        ("musicgen", "MusicgenConfig"),
        ("musicgen_melody", "MusicgenMelodyConfig"),
        ("mt5", "MT5Config"),
        ("mvp", "MvpConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("olmo", "OlmoConfig"),
        ("oneformer", "OneFormerConfig"),
        ("openelm", "OpenELMConfig"),
        ("opt", "OPTConfig"),
        ("owlv2", "Owlv2Config"),
        ("owlvit", "OwlViTConfig"),
        ("pegasus", "PegasusConfig"),
        ("persimmon", "PersimmonConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("plbart", "PLBartConfig"),
        ("qdqbert", "QDQBertConfig"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("reformer", "ReformerConfig"),
        ("rembert", "RemBertConfig"),
        ("resnet", "ResNetConfig"),
        ("roberta", "RobertaConfig"),
        ("roc_bert", "RoCBertConfig"),
        ("rwkv", "RwkvConfig"),
        ("sam", "SamConfig"),
        ("segformer", "SegformerConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("speech_to_text", "Speech2TextConfig"),
        ("speecht5", "SpeechT5Config"),
        ("stablelm", "StableLmConfig"),
        ("splinter", "SplinterConfig"),
        ("squeezebert", "SqueezeBertConfig"),
        ("starcoder2", "Starcoder2Config"),
        ("swin", "SwinConfig"),
        ("swiftformer", "SwiftFormerConfig"),
        ("switch_transformers", "SwitchTransformersConfig"),
        ("swin2sr", "Swin2SRConfig"),
        ("t5", "T5Config"),
        ("tapas", "TapasConfig"),
        ("time_series_transformer", "TimeSeriesTransformerConfig"),
        ("timesformer", "TimesformerConfig"),
        ("trocr", "TrOCRConfig"),
        ("upernet", "UPerNetConfig"),
        ("umt5", "UMT5Config"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("univnet", "UnivNetConfig"),
        ("videomae", "VideoMAEConfig"),
        ("vit", "ViTConfig"),
        ("vilt", "ViltConfig"),
        ("vit_hybrid", "ViTHybridConfig"),
        ("vit_msn", "ViTMSNConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
        ("vipllava", "VipLlavaConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("vitdet", "VitDetConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("wavlm", "WavLMConfig"),
        ("wav2vec2-bert", "Wav2Vec2BertConfig"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerConfig"),
        ("whisper", "WhisperConfig"),
        ("xclip", "XCLIPConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("layoutlmv2", "LayoutLMv2Config"),
        ("xlnet", "XLNetConfig"),
        ("xmod", "XmodConfig"),
    ]
)

CONFIG_ARCHIVE_MAP_MAPPING_NAMES = OrderedDict(
    [
        # Add archive maps here)
        ("albert", "ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("align", "ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("altclip", "ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        (
            "audio-spectrogram-transformer",
            "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        ),
        ("autoformer", "AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bark", "BARK_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bart", "BART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("beit", "BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bert", "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("big_bird", "BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bigbird_pegasus", "BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("biogpt", "BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bit", "BIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blenderbot", "BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blenderbot-small", "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blip", "BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blip-2", "BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bloom", "BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bridgetower", "BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bros", "BROS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("camembert", "CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("canine", "CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("chinese_clip", "CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("chatglm", "CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST"),
        ("clap", "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST"),
        ("clip", "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("clipseg", "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("codegen", "CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("cohere", "COHERE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("conditional_detr", "CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convbert", "CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convnext", "CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convnextv2", "CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("cpmant", "CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("cpmbee", "CPMBEE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ctrl", "CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("cvt", "CVT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-audio", "DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-text", "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-vision", "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deberta", "DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deberta-v2", "DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("decision_transformer", "DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deformable_detr", "DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deit", "DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deta", "DETA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("detr", "DETR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dinat", "DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dinov2", "DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("distilbert", "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("donut-swin", "DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dpr", "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dpt", "DPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("efficientformer", "EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("efficientnet", "EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("electra", "ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("encodec", "ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ernie", "ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ernie_m", "ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("esm", "ESM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("falcon", "FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("flaubert", "FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("flava", "FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fnet", "FNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("focalnet", "FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fsmt", "FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("funnel", "FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fuyu", "FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gemma", "GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("git", "GIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("glpn", "GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt2", "GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_bigcode", "GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_neo", "GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_neox", "GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_neox_japanese", "GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_pangu", "GPTPANGU_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gptj", "GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gptsan-japanese", "GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("graphormer", "GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("groupvit", "GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("hubert", "HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("idefics", "IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("imagegpt", "IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("informer", "INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("instructblip", "INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("jukebox", "JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("kosmos-2", "KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlm", "LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlmv2", "LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlmv3", "LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("led", "LED_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("levit", "LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("lilt", "LILT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("llama", "LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("longformer", "LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("longt5", "LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("luke", "LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("lxmert", "LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("m2m_100", "M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mamba", "MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("marian", "MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("markuplm", "MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mask2former", "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("maskformer", "MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mbart", "MBART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mctct", "MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mega", "MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("megatron-bert", "MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mgp-str", "MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mistral", "MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mixtral", "MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilenet_v1", "MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilenet_v2", "MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilevit", "MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilevitv2", "MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mpnet", "MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mpt", "MPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mra", "MRA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("musicgen", "MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mvp", "MVP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nat", "NAT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nezha", "NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nllb-moe", "NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nystromformer", "NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("oneformer", "ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("open-llama", "OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("openai-gpt", "OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("opt", "OPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("owlv2", "OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("owlvit", "OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pegasus", "PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pegasus_x", "PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("perceiver", "PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("persimmon", "PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("phi", "PHI_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pix2struct", "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("plbart", "PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("poolformer", "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pop2piano", "POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("prophetnet", "PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pvt", "PVT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("qdqbert", "QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("qwen2", "QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("realm", "REALM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("regnet", "REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("rembert", "REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("resnet", "RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roberta", "ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roberta-prelayernorm", "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roc_bert", "ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roformer", "ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("rwkv", "RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sam", "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("seamless_m4t", "SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("segformer", "SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sew", "SEW_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sew-d", "SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speech_to_text", "SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speech_to_text_2", "SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speecht5", "SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("splinter", "SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("squeezebert", "SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("starcoder2", "STARCODER2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swiftformer", "SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swin", "SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swin2sr", "SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swinv2", "SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("switch_transformers", "SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("t5", "T5_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("table-transformer", "TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("tapas", "TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        (
            "time_series_transformer",
            "TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        ),
        ("timesformer", "TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("transfo-xl", "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("tvlt", "TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("unispeech", "UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("unispeech-sat", "UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("van", "VAN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("videomae", "VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vilt", "VILT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("visual_bert", "VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit", "VIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit_hybrid", "VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit_mae", "VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit_msn", "VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vitdet", "VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vitmatte", "VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vits", "VITS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vivit", "VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("wav2vec2", "WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("wav2vec2-conformer", "WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("whisper", "WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xclip", "XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xglm", "XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm", "XLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm-prophetnet", "XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm-roberta", "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm-roberta-xl", "XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlnet", "XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xmod", "XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("yolos", "YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("yoso", "YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        # Add full (and cased) model names here
        ("albert", "ALBERT"),
        ("align", "ALIGN"),
        ("altclip", "AltCLIP"),
        ("audio-spectrogram-transformer", "Audio Spectrogram Transformer"),
        ("autoformer", "Autoformer"),
        ("bark", "Bark"),
        ("bart", "BART"),
        ("barthez", "BARThez"),
        ("bartpho", "BARTpho"),
        ("beit", "BEiT"),
        ("bert", "BERT"),
        ("bert-generation", "Bert Generation"),
        ("bert-japanese", "BertJapanese"),
        ("bertweet", "BERTweet"),
        ("bge-m3", "BgeM3"),
        ("big_bird", "BigBird"),
        ("bigbird_pegasus", "BigBird-Pegasus"),
        ("biogpt", "BioGpt"),
        ("bit", "BiT"),
        ("blenderbot", "Blenderbot"),
        ("blenderbot-small", "BlenderbotSmall"),
        ("blip", "BLIP"),
        ("blip-2", "BLIP-2"),
        ("bloom", "BLOOM"),
        ("bort", "BORT"),
        ("bridgetower", "BridgeTower"),
        ("bros", "BROS"),
        ("byt5", "ByT5"),
        ("camembert", "CamemBERT"),
        ("canine", "CANINE"),
        ("chinese_clip", "Chinese-CLIP"),
        ("chatglm", "ChatGLM"),
        ("clap", "CLAP"),
        ("clip", "CLIP"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("clipseg", "CLIPSeg"),
        ("clipseg_vision_model", "CLIPSegVisionModel"),
        ("code_llama", "CodeLlama"),
        ("codegen", "CodeGen"),
        ("cohere", "Cohere"),
        ("conditional_detr", "Conditional DETR"),
        ("cogvlm", "CogVLM"),
        ("convbert", "ConvBERT"),
        ("convnext", "ConvNeXT"),
        ("convnextv2", "ConvNeXTV2"),
        ("cpm", "CPM"),
        ("cpmant", "CPM-Ant"),
        ("cpmbee", "CPM-Bee"),
        ("ctrl", "CTRL"),
        ("cvt", "CvT"),
        ("data2vec-audio", "Data2VecAudio"),
        ("data2vec-text", "Data2VecText"),
        ("data2vec-vision", "Data2VecVision"),
        ("deberta", "DeBERTa"),
        ("deberta-v2", "DeBERTa-v2"),
        ("decision_transformer", "Decision Transformer"),
        ("deformable_detr", "Deformable DETR"),
        ("deit", "DeiT"),
        ("deplot", "DePlot"),
        ("deta", "DETA"),
        ("detr", "DETR"),
        ("dialogpt", "DialoGPT"),
        ("dinat", "DiNAT"),
        ("dinov2", "DINOv2"),
        ("distilbert", "DistilBERT"),
        ("donut", "Donut"),
        ("donut-swin", "DonutSwin"),
        ("dit", "DiT"),
        ("donut-swin", "DonutSwin"),
        ("dpr", "DPR"),
        ("dpt", "DPT"),
        ("efficientformer", "EfficientFormer"),
        ("efficientnet", "EfficientNet"),
        ("electra", "ELECTRA"),
        ("encodec", "EnCodec"),
        ("encoder-decoder", "Encoder decoder"),
        ("ernie", "ERNIE"),
        ("ernie_m", "ErnieM"),
        ("esm", "ESM"),
        ("falcon", "Falcon"),
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("flan-t5", "FLAN-T5"),
        ("flan-ul2", "FLAN-UL2"),
        ("flaubert", "FlauBERT"),
        ("flava", "FLAVA"),
        ("fnet", "FNet"),
        ("focalnet", "FocalNet"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("funnel", "Funnel Transformer"),
        ("fuyu", "Fuyu"),
        ("gemma", "Gemma"),
        ("git", "GIT"),
        ("glpn", "GLPN"),
        ("gpt-sw3", "GPT-Sw3"),
        ("gpt2", "OpenAI GPT-2"),
        ("gpt_bigcode", "GPTBigCode"),
        ("gpt_neo", "GPT Neo"),
        ("gpt_neox", "GPT NeoX"),
        ("gpt_neox_japanese", "GPT NeoX Japanese"),
        ("gpt_pangu", "GPTPangu"),
        ("gptj", "GPT-J"),
        ("gptsan-japanese", "GPTSAN-japanese"),
        ("graphormer", "Graphormer"),
        ("groupvit", "GroupViT"),
        ("herbert", "HerBERT"),
        ("hubert", "Hubert"),
        ("ibert", "I-BERT"),
        ("idefics", "IDEFICS"),
        ("imagegpt", "ImageGPT"),
        ("informer", "Informer"),
        ("instructblip", "InstructBLIP"),
        ("jukebox", "Jukebox"),
        ("jetmoe", "JetMoE"),
        ("kosmos-2", "KOSMOS-2"),
        ("layoutlm", "LayoutLM"),
        ("layoutlmv2", "LayoutLMv2"),
        ("layoutlmv3", "LayoutLMv3"),
        ("layoutxlm", "LayoutXLM"),
        ("led", "LED"),
        ("levit", "LeViT"),
        ("lilt", "LiLT"),
        ("llama", "LLaMA"),
        ("llama2", "Llama2"),
        ("llava", "LLaVa"),
        ("llava_next", "LLaVA-NeXT"),
        ("longformer", "Longformer"),
        ("longt5", "LongT5"),
        ("luke", "LUKE"),
        ("lxmert", "LXMERT"),
        ("m2m_100", "M2M100"),
        ("mamba", "Mamba"),
        ("marian", "Marian"),
        ("markuplm", "MarkupLM"),
        ("mask2former", "Mask2Former"),
        ("maskformer", "MaskFormer"),
        ("maskformer-swin", "MaskFormerSwin"),
        ("matcha", "MatCha"),
        ("mbart", "mBART"),
        ("mbart50", "mBART-50"),
        ("mctct", "M-CTC-T"),
        ("mega", "MEGA"),
        ("megatron-bert", "Megatron-BERT"),
        ("megatron_gpt2", "Megatron-GPT2"),
        ("mgp-str", "MGP-STR"),
        ("minicpm", "MiniCPM"),
        ("mistral", "Mistral"),
        ("mixtral", "Mixtral"),
        ("mluke", "mLUKE"),
        ("mms", "MMS"),
        ("mobilebert", "MobileBERT"),
        ("mobilenet_v1", "MobileNetV1"),
        ("mobilenet_v2", "MobileNetV2"),
        ("mobilevit", "MobileViT"),
        ("mobilevitv2", "MobileViTV2"),
        ("mpnet", "MPNet"),
        ("mpt", "MPT"),
        ("mra", "MRA"),
        ("mt5", "MT5"),
        ("musicgen", "MusicGen"),
        ("musicgen_melody", "MusicGen Melody"),
        ("mvp", "MVP"),
        ("nat", "NAT"),
        ("nezha", "Nezha"),
        ("nllb", "NLLB"),
        ("nllb-moe", "NLLB-MOE"),
        ("nougat", "Nougat"),
        ("nystromformer", "Nyströmformer"),
        ("olmo", "OLMo"),
        ("openelm", "OpenELM"),
        ("oneformer", "OneFormer"),
        ("open-llama", "OpenLlama"),
        ("openai-gpt", "OpenAI GPT"),
        ("opt", "OPT"),
        ("owlv2", "OWLv2"),
        ("owlvit", "OWL-ViT"),
        ("pegasus", "Pegasus"),
        ("pegasus_x", "PEGASUS-X"),
        ("perceiver", "Perceiver"),
        ("persimmon", "Persimmon"),
        ("phi", "Phi"),
        ("phi3", "Phi3"),
        ("phobert", "PhoBERT"),
        ("pix2struct", "Pix2Struct"),
        ("plbart", "PLBart"),
        ("poolformer", "PoolFormer"),
        ("pop2piano", "Pop2Piano"),
        ("prophetnet", "ProphetNet"),
        ("pvt", "PVT"),
        ("qdqbert", "QDQBert"),
        ("qwen2", "Qwen2"),
        ("qwen2_moe", "Qwen2MoE"),
        ("rag", "RAG"),
        ("realm", "REALM"),
        ("reformer", "Reformer"),
        ("regnet", "RegNet"),
        ("rembert", "RemBERT"),
        ("resnet", "ResNet"),
        ("roberta", "RoBERTa"),
        ("roberta-prelayernorm", "RoBERTa-PreLayerNorm"),
        ("roc_bert", "RoCBert"),
        ("roformer", "RoFormer"),
        ("rwkv", "RWKV"),
        ("sam", "SAM"),
        ("seamless_m4t", "SeamlessM4T"),
        ("segformer", "SegFormer"),
        ("sew", "SEW"),
        ("sew-d", "SEW-D"),
        ("speech-encoder-decoder", "Speech Encoder decoder"),
        ("speech_to_text", "Speech2Text"),
        ("speech_to_text_2", "Speech2Text2"),
        ("speecht5", "SpeechT5"),
        ("splinter", "Splinter"),
        ("squeezebert", "SqueezeBERT"),
        ("stablelm", "StableLm"),
        ("starcoder2", "Starcoder2"),
        ("swiftformer", "SwiftFormer"),
        ("swin", "Swin Transformer"),
        ("swin2sr", "Swin2SR"),
        ("swinv2", "Swin Transformer V2"),
        ("switch_transformers", "SwitchTransformers"),
        ("t5", "T5"),
        ("t5v1.1", "T5v1.1"),
        ("table-transformer", "Table Transformer"),
        ("tapas", "TAPAS"),
        ("tapex", "TAPEX"),
        ("time_series_transformer", "Time Series Transformer"),
        ("timesformer", "TimeSformer"),
        ("timm_backbone", "TimmBackbone"),
        ("trajectory_transformer", "Trajectory Transformer"),
        ("transfo-xl", "Transformer-XL"),
        ("trocr", "TrOCR"),
        ("tvlt", "TVLT"),
        ("ul2", "UL2"),
        ("umt5", "UMT5"),
        ("unispeech", "UniSpeech"),
        ("unispeech-sat", "UniSpeechSat"),
        ("univnet", "UnivNet"),
        ("upernet", "UPerNet"),
        ("van", "VAN"),
        ("videomae", "VideoMAE"),
        ("vilt", "ViLT"),
        ("vipllava", "VipLlava"),
        ("vision-encoder-decoder", "Vision Encoder decoder"),
        ("vision-text-dual-encoder", "VisionTextDualEncoder"),
        ("visual_bert", "VisualBERT"),
        ("vit", "ViT"),
        ("vit_hybrid", "ViT Hybrid"),
        ("vit_mae", "ViTMAE"),
        ("vit_msn", "ViTMSN"),
        ("vitdet", "VitDet"),
        ("vitmatte", "ViTMatte"),
        ("vits", "VITS"),
        ("vivit", "ViViT"),
        ("wav2vec2", "Wav2Vec2"),
        ("wav2vec2-bert", "Wav2Vec2-BERT"),
        ("wav2vec2-conformer", "Wav2Vec2-Conformer"),
        ("wav2vec2_phoneme", "Wav2Vec2Phoneme"),
        ("wavlm", "WavLM"),
        ("whisper", "Whisper"),
        ("xclip", "X-CLIP"),
        ("xglm", "XGLM"),
        ("xlm", "XLM"),
        ("xlm-prophetnet", "XLM-ProphetNet"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("xlm-roberta-xl", "XLM-RoBERTa-XL"),
        ("xlm-v", "XLM-V"),
        ("xlnet", "XLNet"),
        ("xls_r", "XLS-R"),
        ("xlsr_wav2vec2", "XLSR-Wav2Vec2"),
        ("xmod", "X-MOD"),
        ("yolos", "YOLOS"),
        ("yoso", "YOSO"),
    ]
)

DEPRECATED_MODELS = [
    # "bort",
    # "mctct",
    # "mmbt",
    # "open_llama",
    # "retribert",
    # "tapex",
    # "trajectory_transformer",
    # "van",
]

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("kosmos-2", "kosmos2"),
        ("maskformer-swin", "maskformer"),
        ("clip_vision_model", "clip"),
        ("clipseg_vision_model", "clipseg"),
        ("xclip", "x_clip"),
    ]
)


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # if key not found check in extra content
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        """
        Initializes a new instance of the _LazyConfigMapping class.

        Args:
            self (object): The instance of the _LazyConfigMapping class.
            mapping (dict): A dictionary containing the initial mapping for the configuration.
               The keys represent the configuration keys, and the values represent the corresponding values.
               This parameter is required and must be of type dict.

        Returns:
            None.

        Raises:
            None.
        """
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        """
                __getitem__

                Retrieve the value associated with the given key from the _LazyConfigMapping object.

                Args:
                    self (_LazyConfigMapping): The instance of the _LazyConfigMapping class.
                    key (str): The key used to retrieve the corresponding value. It should be a string.

        Returns:
            None:
                - If the key is present in the _extra_content, the associated value is returned.
                - If the key is not in _extra_content but is in _mapping, the value associated with the key is returned after
                performing necessary module imports and attribute retrieval.

        Raises:
            KeyError: If the key is not found in either _extra_content or _mapping, a KeyError is raised.
            AttributeError:
                If the attribute associated with the value corresponding to the key is not found in the dynamically imported module,
                an AttributeError is raised.
            ModuleNotFoundError: If the required module is not found during dynamic import, a ModuleNotFoundError is raised.
        """
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "mindnlp.transformers.models"
            )
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("mindnlp.transformers")
        return getattr(transformers_module, value)

    def keys(self):
        """
        This method returns a list of all the keys in the _LazyConfigMapping object.

        Args:
            self (_LazyConfigMapping): The instance of the _LazyConfigMapping class.

        Returns:
            list: A list containing all the keys in the _mapping and _extra_content attributes of the _LazyConfigMapping object.

        Raises:
            None
        """
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        """
        This method returns a list containing the values of the mapping and any extra content in the _LazyConfigMapping class.

        Args:
            self (_LazyConfigMapping): The instance of the _LazyConfigMapping class.
                This parameter is used to access the instance's mapping and extra content.

        Returns:
            list: A list containing the values of the mapping and any extra content in the _LazyConfigMapping class.

        Raises:
            None
        """
        return [self[k] for k in self._mapping.keys()] + list(
            self._extra_content.values()
        )

    def items(self):
        """
        items(self)
            This method returns a list of key-value pairs from the _LazyConfigMapping instance.

        Args:
            self (_LazyConfigMapping): The instance of the _LazyConfigMapping class.

        Returns:
            list: A list of key-value pairs from the _LazyConfigMapping instance.

        Raises:
            None.
        """
        return [(k, self[k]) for k in self._mapping.keys()] + list(
            self._extra_content.items()
        )

    def __iter__(self):
        """
        Iterates over the keys of the '_LazyConfigMapping' instance.

        Args:
            self (object): The instance of the '_LazyConfigMapping' class.

        Returns:
            None.

        Raises:
            None.
        """
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        """
        Check if the given item is contained within the '_LazyConfigMapping' object.

        Args:
            self (_LazyConfigMapping): The instance of the '_LazyConfigMapping' class.
            item: The item to be checked for containment within the object. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(
                f"'{key}' is already used by a Transformers config, pick another name."
            )
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


class _LazyLoadAllMappings(OrderedDict):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        """
        Initializes an instance of the '_LazyLoadAllMappings' class.

        Args:
            self: An instance of the '_LazyLoadAllMappings' class.
            mapping: A dictionary containing the mapping data to be used by the instance.

        Returns:
            None.

        Raises:
            None.
        """
        self._mapping = mapping
        self._initialized = False
        self._data = {}

    def _initialize(self):
        """
        Initializes the lazy loading of all mappings for the _LazyLoadAllMappings class.

        Args:
            self: _LazyLoadAllMappings - The instance of the _LazyLoadAllMappings class.

        Returns:
            None.

        Raises:
            FutureWarning: If ALL_PRETRAINED_CONFIG_ARCHIVE_MAP is deprecated and will be removed in v5 of Transformers.
                It does not contain all available model checkpoints. Refer to hf.co/models for that.
        """
        if self._initialized:
            return
        warnings.warn(
            "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP is deprecated and will be removed in v5 of Transformers. "
            "It does not contain all available model checkpoints, far from it. Checkout hf.co/models for that.",
            FutureWarning,
        )

        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            mapping = getattr(module, map_name)
            self._data.update(mapping)

        self._initialized = True

    def __getitem__(self, key):
        """
        __getitem__

        This method retrieves the value associated with the specified key from the _LazyLoadAllMappings instance.

        Args:
            self (_LazyLoadAllMappings): The instance of the _LazyLoadAllMappings class.
            key: The key for which the associated value needs to be retrieved. Type: Any hashable object.

        Returns:
            `_data[key]`: This method returns the value associated with the specified key.
                If the key is not found, a KeyError is raised.

        Raises:
            KeyError: If the specified key is not found in the _LazyLoadAllMappings instance.
        """
        self._initialize()
        return self._data[key]

    def keys(self):
        """
        Returns a list of all the keys in the mappings.

        Args:
            self (obj): An instance of the _LazyLoadAllMappings class.

        Returns:
            None.

        Raises:
            None.
        """
        self._initialize()
        return self._data.keys()

    def values(self):
        """
        Method 'values' in the class '_LazyLoadAllMappings' returns the values of the internal data dictionary after initialization.

        Args:
            self (object): The instance of the class.
                This parameter represents the current instance of the class '_LazyLoadAllMappings'.
                It is required to access the internal data and perform necessary operations.

        Returns:
            None:
                This method does not return any specific value but retrieves and returns the values of the internal data dictionary.

        Raises:
            No specific exceptions:
                No specific exceptions are raised by this method under normal circumstances,
                However, if there are any issues during the initialization process, unexpected behavior may occur.
        """
        self._initialize()
        return self._data.values()

    def items(self):
        """
        Method 'items' in the class '_LazyLoadAllMappings' retrieves the keys of the data after initialization.

        Args:
            self (object): The instance of the class.
                This parameter is required to access the instance attributes and methods.
                It must be an instance of the class '_LazyLoadAllMappings'.

        Returns:
            None:
                This method returns the keys of the data stored in the '_data' attribute after initialization.

        Raises:
            No exceptions are explicitly raised within this method.
        """
        self._initialize()
        return self._data.keys()

    def __iter__(self):
        """
        __iter__

        This method returns an iterator object for the _LazyLoadAllMappings instance.

        Args:
            self (_LazyLoadAllMappings): The instance of the _LazyLoadAllMappings class.

        Returns:
            iter: An iterator object for the _data attribute of the _LazyLoadAllMappings instance.

        Raises:
            None
        """
        self._initialize()
        return iter(self._data)

    def __contains__(self, item):
        """
        This method '__contains__' in the class '_LazyLoadAllMappings' determines if the specified item is contained within the data structure.

        Args:
            self (_LazyLoadAllMappings): The instance of the _LazyLoadAllMappings class.
                This parameter represents the current object instance.
            item: The item to check for existence within the data structure.
                Type: Any
                Purpose: To specify the item to be checked.
                Restrictions: None

        Returns:
            None: This method returns None if the specified item is found within the data structure, otherwise it returns False.

        Raises:
            None.
        """
        self._initialize()
        return item in self._data


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = _LazyLoadAllMappings(
    CONFIG_ARCHIVE_MAP_MAPPING_NAMES
)


def _get_class_name(model_class: Union[str, List[str]]):
    """
    This function returns a formatted string representing the class name or names provided as the 'model_class' parameter.

    Args:
        model_class (Union[str, List[str]]): The class name or names to be formatted. It can be a string or a list of strings.
            If 'model_class' is a list, the function will join the class names using ' or '. Empty strings or None values
            in the list will be ignored.

    Returns:
        str: A formatted string representing the class name or names provided. The class name(s) will be enclosed in
            backticks for clarity.

    Raises:
        None: This function does not raise any exceptions.

    Note:
        The function can be called with either a single class name or a list of class names. If the 'model_class' parameter
        is a list, the resulting string will represent the joined class names using ' or ' as the separator. Empty strings
        or None values will be ignored.
    """
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    """
    Args:
        indent (str): The string used for indentation in the output.
        config_to_class (dict): A dictionary mapping model configurations to their corresponding classes.
            Defaults to None.
        use_model_types (bool): A flag indicating whether to use model types or configurations.
            Defaults to True.

    Returns:
        str: A formatted string containing the model options.

    Raises:
        ValueError: If `use_model_types` is set to False and `config_to_class` is not provided.
    """
    if config_to_class is None and not use_model_types:
        raise ValueError(
            "Using `use_model_types=False` requires a `config_to_class` dictionary."
        )
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {
                model_type: f"[`{config}`]"
                for model_type, config in CONFIG_MAPPING_NAMES.items()
            }
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type]
            for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    """
    This function decorates another function's docstring to include a section for listing options related to model types.

    Args:
        config_to_class (optional): A dictionary representing the configuration to class mapping. Defaults to None.
        use_model_types (optional): A boolean indicating whether model types should be used. Defaults to True.

    Returns:
        None.

    Raises:
        ValueError: If the input function's docstring does not contain an empty 'List options' section as a placeholder.
    """

    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(
                indent, config_to_class=config_to_class, use_model_types=use_model_types
            )
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        """
        Initialize AutoConfig.

        Args:
            self: The instance of the AutoConfig class.
                It is automatically passed when the method is called.

                - Purpose: Represents the instance of the AutoConfig class.
                - Restrictions: None.

        Returns:
            None.

        Raises:
            EnvironmentError:
                If the AutoConfig is instantiated directly using the `__init__` method,
                it raises an EnvironmentError with the message
                'AutoConfig is designed to be instantiated using the `
                AutoConfig.from_pretrained(pretrained_model_name_or_path)` method.'.
        """
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        """
                This class method 'for_model' in the 'AutoConfig' class is used to instantiate a configuration class based on the provided model type.

        Args:
            cls (class): The class itself, automatically passed as the first parameter.
            model_type (str): A string representing the type of the model for which the configuration class needs to be instantiated.
                It must be a key within the CONFIG_MAPPING dictionary.

        Returns:
            None: This method does not return any value directly.
                It instantiates and returns an instance of the appropriate configuration class based on the model type.

        Raises:
            ValueError:
                Raised when the provided 'model_type' is not recognized or is not found as a key in the CONFIG_MAPPING dictionary.
                The exception message indicates the unrecognized model identifier and lists all valid model identifiers
                available in the CONFIG_MAPPING dictionary.
        """
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                hf-mirror.com. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing a configuration file saved using the
                [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                e.g., `./my_model_directory/`.
                - A path or url to a saved configuration JSON *file*, e.g.,
                `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on hf-mirror.com, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.
                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Example:
            ```python
            >>> from transformers import AutoConfig
            ...
            >>> # Download configuration from hf-mirror.com and cache.
            >>> config = AutoConfig.from_pretrained("bert-base-uncased")
            ...
            >>> # Download configuration from hf-mirror.com (user-uploaded) and cache.
            >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")
            ...
            >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
            >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")
            ...
            >>> # Load a specific configuration file.
            >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")
            ...
            >>> # Change some config attributes when loading a pretrained config.
            >>> config = AutoConfig.from_pretrained("bert-base-uncased", output_attentions=True, foo=False)
            >>> config.output_attentions
            True
            >>> config, unused_kwargs = AutoConfig.from_pretrained(
            ...     "bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
            ... )
            >>> config.output_attentions
            True
            >>> unused_kwargs
            {'foo': False}
            ```
        """
        kwargs["name_or_path"] = pretrained_model_name_or_path

        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        if "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **unused_kwargs)
        # Fallback: use pattern matching on the string.
        # We go from longer names to shorter names to catch roberta before bert (for instance)
        for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
            if pattern in str(pretrained_model_name_or_path).lower():
                return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)
