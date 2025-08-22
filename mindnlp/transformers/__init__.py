import sys
from mindnlp.core.configs import ON_ORANGE_PI
from mindnlp.utils.import_utils import *
from mindnlp.utils.import_utils import _LazyModule

# Base objects, independent of any specific backend
_import_structure = {
    # "agents": [
    #     "Agent",
    #     "CodeAgent",
    #     "HfApiEngine",
    #     "ManagedAgent",
    #     "PipelineTool",
    #     "ReactAgent",
    #     "ReactCodeAgent",
    #     "ReactJsonAgent",
    #     "Tool",
    #     "Toolbox",
    #     "ToolCollection",
    #     "TransformersEngine",
    #     "launch_gradio_demo",
    #     "load_tool",
    #     "stream_to_gradio",
    #     "tool",
    # ],
    "audio_utils": [],
    "commands": [],
    "configuration_utils": ["PretrainedConfig"],
    "convert_slow_tokenizers_checkpoints_to_fast": [],
    "data": [
        "DataProcessor",
        "InputExample",
        "InputFeatures",
        "SingleSentenceClassificationProcessor",
        "SquadExample",
        "SquadFeatures",
        "SquadV1Processor",
        "SquadV2Processor",
        "glue_compute_metrics",
        "glue_convert_examples_to_features",
        "glue_output_modes",
        "glue_processors",
        "glue_tasks_num_labels",
        "squad_convert_examples_to_features",
        "xnli_compute_metrics",
        "xnli_output_modes",
        "xnli_processors",
        "xnli_tasks_num_labels",
    ],
    "data.data_collator": [
        "DataCollator",
        "DataCollatorForLanguageModeling",
        "DataCollatorForMultipleChoice",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithFlattening",
        "DataCollatorWithPadding",
        "DefaultDataCollator",
        "default_data_collator",
    ],
    "data.metrics": [],
    "data.processors": [],
    "debug_utils": [],
    "dependency_versions_check": [],
    "dependency_versions_table": [],
    "dynamic_module_utils": [],
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "file_utils": [],
    "generation": [
        "AsyncTextIteratorStreamer",
        "CompileConfig",
        "GenerationConfig",
        "TextIteratorStreamer",
        "TextStreamer",
        "WatermarkingConfig",
    ],
    "hf_argparser": ["HfArgumentParser"],
    "hyperparameter_search": [],
    "image_transforms": [],
    "loss": [],
    "modelcard": ["ModelCard"],
    # Models
    "models": [],
    "models.albert": ["AlbertConfig"],
    "models.align": [
        "AlignConfig",
        "AlignProcessor",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    "models.altclip": [
        "AltCLIPConfig",
        "AltCLIPProcessor",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    "models.aria": [
        "AriaConfig",
        "AriaProcessor",
        "AriaTextConfig",
    ],
    "models.audio_spectrogram_transformer": [
        "ASTConfig",
        "ASTFeatureExtractor",
    ],
    "models.auto": [
        "CONFIG_MAPPING",
        "FEATURE_EXTRACTOR_MAPPING",
        "IMAGE_PROCESSOR_MAPPING",
        "MODEL_NAMES_MAPPING",
        "PROCESSOR_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoFeatureExtractor",
        "AutoImageProcessor",
        "AutoProcessor",
        "AutoTokenizer",
    ],
    "models.autoformer": ["AutoformerConfig"],
    "models.aya_vision": ["AyaVisionConfig", "AyaVisionProcessor"],
    "models.bamba": ["BambaConfig"],
    "models.bark": [
        "BarkCoarseConfig",
        "BarkConfig",
        "BarkFineConfig",
        "BarkProcessor",
        "BarkSemanticConfig",
    ],
    "models.bart": ["BartConfig", "BartTokenizer"],
    "models.barthez": [],
    "models.bartpho": [],
    "models.beit": ["BeitConfig"],
    "models.bert": [
        "BasicTokenizer",
        "BertConfig",
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
    "models.bert_generation": ["BertGenerationConfig"],
    "models.bert_japanese": [
        "BertJapaneseTokenizer",
        "CharacterTokenizer",
        "MecabTokenizer",
    ],
    "models.bertweet": ["BertweetTokenizer"],
    "models.big_bird": ["BigBirdConfig"],
    "models.bigbird_pegasus": ["BigBirdPegasusConfig"],
    "models.biogpt": [
        "BioGptConfig",
        "BioGptTokenizer",
    ],
    "models.bit": ["BitConfig"],
    "models.blenderbot": [
        "BlenderbotConfig",
        "BlenderbotTokenizer",
    ],
    "models.blenderbot_small": [
        "BlenderbotSmallConfig",
        "BlenderbotSmallTokenizer",
    ],
    "models.blip": [
        "BlipConfig",
        "BlipProcessor",
        "BlipTextConfig",
        "BlipVisionConfig",
    ],
    "models.blip_2": [
        "Blip2Config",
        "Blip2Processor",
        "Blip2QFormerConfig",
        "Blip2VisionConfig",
    ],
    "models.bloom": ["BloomConfig"],
    "models.bridgetower": [
        "BridgeTowerConfig",
        "BridgeTowerProcessor",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "models.bros": [
        "BrosConfig",
        "BrosProcessor",
    ],
    "models.byt5": ["ByT5Tokenizer"],
    "models.camembert": ["CamembertConfig"],
    "models.canine": [
        "CanineConfig",
        "CanineTokenizer",
    ],
    "models.chameleon": [
        "ChameleonConfig",
        "ChameleonProcessor",
        "ChameleonVQVAEConfig",
    ],
    "models.chinese_clip": [
        "ChineseCLIPConfig",
        "ChineseCLIPProcessor",
        "ChineseCLIPTextConfig",
        "ChineseCLIPVisionConfig",
    ],
    "models.clap": [
        "ClapAudioConfig",
        "ClapConfig",
        "ClapProcessor",
        "ClapTextConfig",
    ],
    "models.clip": [
        "CLIPConfig",
        "CLIPProcessor",
        "CLIPTextConfig",
        "CLIPTokenizer",
        "CLIPVisionConfig",
    ],
    "models.clipseg": [
        "CLIPSegConfig",
        "CLIPSegProcessor",
        "CLIPSegTextConfig",
        "CLIPSegVisionConfig",
    ],
    "models.clvp": [
        "ClvpConfig",
        "ClvpDecoderConfig",
        "ClvpEncoderConfig",
        "ClvpFeatureExtractor",
        "ClvpProcessor",
        "ClvpTokenizer",
    ],
    "models.code_llama": [],
    "models.codegen": [
        "CodeGenConfig",
        "CodeGenTokenizer",
    ],
    "models.cohere": ["CohereConfig"],
    "models.cohere2": ["Cohere2Config"],
    "models.colpali": [
        "ColPaliConfig",
        "ColPaliProcessor",
    ],
    "models.conditional_detr": ["ConditionalDetrConfig"],
    "models.convbert": [
        "ConvBertConfig",
        "ConvBertTokenizer",
    ],
    "models.convnext": ["ConvNextConfig"],
    "models.convnextv2": ["ConvNextV2Config"],
    "models.cpm": [],
    "models.cpmant": [
        "CpmAntConfig",
        "CpmAntTokenizer",
    ],
    "models.ctrl": [
        "CTRLConfig",
        "CTRLTokenizer",
    ],
    "models.cvt": ["CvtConfig"],
    "models.dab_detr": ["DabDetrConfig"],
    "models.dac": ["DacConfig", "DacFeatureExtractor"],
    "models.data2vec": [
        "Data2VecAudioConfig",
        "Data2VecTextConfig",
        "Data2VecVisionConfig",
    ],
    "models.dbrx": ["DbrxConfig"],
    "models.deberta": [
        "DebertaConfig",
        "DebertaTokenizer",
    ],
    "models.deberta_v2": ["DebertaV2Config"],
    "models.decision_transformer": ["DecisionTransformerConfig"],
    "models.deepseek_v3": ["DeepseekV3Config"],
    "models.deformable_detr": ["DeformableDetrConfig"],
    "models.deit": ["DeiTConfig"],
    "models.deprecated": [],
    "models.deprecated.bort": [],
    "models.deprecated.deta": ["DetaConfig"],
    "models.deprecated.efficientformer": ["EfficientFormerConfig"],
    "models.deprecated.ernie_m": ["ErnieMConfig"],
    "models.deprecated.gptsan_japanese": [
        "GPTSanJapaneseConfig",
        "GPTSanJapaneseTokenizer",
    ],
    "models.deprecated.graphormer": ["GraphormerConfig"],
    "models.deprecated.jukebox": [
        "JukeboxConfig",
        "JukeboxPriorConfig",
        "JukeboxTokenizer",
        "JukeboxVQVAEConfig",
    ],
    "models.deprecated.mctct": [
        "MCTCTConfig",
        "MCTCTFeatureExtractor",
        "MCTCTProcessor",
    ],
    "models.deprecated.mega": ["MegaConfig"],
    "models.deprecated.mmbt": ["MMBTConfig"],
    "models.deprecated.nat": ["NatConfig"],
    "models.deprecated.nezha": ["NezhaConfig"],
    "models.deprecated.open_llama": ["OpenLlamaConfig"],
    "models.deprecated.qdqbert": ["QDQBertConfig"],
    "models.deprecated.realm": [
        "RealmConfig",
        "RealmTokenizer",
    ],
    "models.deprecated.retribert": [
        "RetriBertConfig",
        "RetriBertTokenizer",
    ],
    "models.deprecated.speech_to_text_2": [
        "Speech2Text2Config",
        "Speech2Text2Processor",
        "Speech2Text2Tokenizer",
    ],
    "models.deprecated.tapex": ["TapexTokenizer"],
    "models.deprecated.trajectory_transformer": ["TrajectoryTransformerConfig"],
    "models.deprecated.transfo_xl": [
        "TransfoXLConfig",
        "TransfoXLCorpus",
        "TransfoXLTokenizer",
    ],
    "models.deprecated.tvlt": [
        "TvltConfig",
        "TvltFeatureExtractor",
        "TvltProcessor",
    ],
    "models.deprecated.van": ["VanConfig"],
    "models.deprecated.vit_hybrid": ["ViTHybridConfig"],
    "models.deprecated.xlm_prophetnet": ["XLMProphetNetConfig"],
    "models.depth_anything": ["DepthAnythingConfig"],
    "models.depth_pro": ["DepthProConfig"],
    "models.detr": ["DetrConfig"],
    "models.dialogpt": [],
    "models.diffllama": ["DiffLlamaConfig"],
    "models.dinat": ["DinatConfig"],
    "models.dinov2": ["Dinov2Config"],
    "models.dinov2_with_registers": ["Dinov2WithRegistersConfig"],
    "models.distilbert": [
        "DistilBertConfig",
        "DistilBertTokenizer",
    ],
    "models.dit": [],
    "models.donut": [
        "DonutProcessor",
        "DonutSwinConfig",
    ],
    "models.dpr": [
        "DPRConfig",
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
    "models.dpt": ["DPTConfig"],
    "models.efficientnet": ["EfficientNetConfig"],
    "models.electra": [
        "ElectraConfig",
        "ElectraTokenizer",
    ],
    "models.emu3": [
        "Emu3Config",
        "Emu3Processor",
        "Emu3TextConfig",
        "Emu3VQVAEConfig",
    ],
    "models.encodec": [
        "EncodecConfig",
        "EncodecFeatureExtractor",
    ],
    "models.encoder_decoder": ["EncoderDecoderConfig"],
    "models.ernie": ["ErnieConfig"],
    "models.esm": ["EsmConfig", "EsmTokenizer"],
    "models.falcon": ["FalconConfig"],
    "models.falcon_mamba": ["FalconMambaConfig"],
    "models.fastspeech2_conformer": [
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerTokenizer",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    "models.flaubert": ["FlaubertConfig", "FlaubertTokenizer"],
    "models.flava": [
        "FlavaConfig",
        "FlavaImageCodebookConfig",
        "FlavaImageConfig",
        "FlavaMultimodalConfig",
        "FlavaTextConfig",
    ],
    "models.fnet": ["FNetConfig"],
    "models.focalnet": ["FocalNetConfig"],
    "models.fsmt": [
        "FSMTConfig",
        "FSMTTokenizer",
    ],
    "models.funnel": [
        "FunnelConfig",
        "FunnelTokenizer",
    ],
    "models.fuyu": ["FuyuConfig"],
    "models.gemma": ["GemmaConfig"],
    "models.gemma2": ["Gemma2Config"],
    "models.gemma3": ["Gemma3Config", "Gemma3Processor", "Gemma3TextConfig"],
    "models.git": [
        "GitConfig",
        "GitProcessor",
        "GitVisionConfig",
    ],
    "models.glm": ["GlmConfig"],
    "models.glpn": ["GLPNConfig"],
    "models.got_ocr2": [
        "GotOcr2Config",
        "GotOcr2Processor",
        "GotOcr2VisionConfig",
    ],
    "models.gpt2": [
        "GPT2Config",
        "GPT2Tokenizer",
    ],
    "models.gpt_bigcode": ["GPTBigCodeConfig"],
    "models.gpt_neo": ["GPTNeoConfig"],
    "models.gpt_neox": ["GPTNeoXConfig"],
    "models.gpt_neox_japanese": ["GPTNeoXJapaneseConfig"],
    "models.gpt_sw3": [],
    "models.gptj": ["GPTJConfig"],
    "models.granite": ["GraniteConfig"],
    "models.granitemoe": ["GraniteMoeConfig"],
    "models.granitemoeshared": ["GraniteMoeSharedConfig"],
    "models.grounding_dino": [
        "GroundingDinoConfig",
        "GroundingDinoProcessor",
    ],
    "models.groupvit": [
        "GroupViTConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
    "models.helium": ["HeliumConfig"],
    "models.herbert": ["HerbertTokenizer"],
    "models.hiera": ["HieraConfig"],
    "models.hubert": ["HubertConfig"],
    "models.ibert": ["IBertConfig"],
    "models.idefics": ["IdeficsConfig"],
    "models.idefics2": ["Idefics2Config"],
    "models.idefics3": ["Idefics3Config"],
    "models.ijepa": ["IJepaConfig"],
    "models.imagegpt": ["ImageGPTConfig"],
    "models.informer": ["InformerConfig"],
    "models.instructblip": [
        "InstructBlipConfig",
        "InstructBlipProcessor",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    "models.instructblipvideo": [
        "InstructBlipVideoConfig",
        "InstructBlipVideoProcessor",
        "InstructBlipVideoQFormerConfig",
        "InstructBlipVideoVisionConfig",
    ],
    "models.jamba": ["JambaConfig"],
    "models.jetmoe": ["JetMoeConfig"],
    "models.kosmos2": [
        "Kosmos2Config",
        "Kosmos2Processor",
    ],
    "models.layoutlm": [
        "LayoutLMConfig",
        "LayoutLMTokenizer",
    ],
    "models.layoutlmv2": [
        "LayoutLMv2Config",
        "LayoutLMv2FeatureExtractor",
        "LayoutLMv2ImageProcessor",
        "LayoutLMv2Processor",
        "LayoutLMv2Tokenizer",
    ],
    "models.layoutlmv3": [
        "LayoutLMv3Config",
        "LayoutLMv3FeatureExtractor",
        "LayoutLMv3ImageProcessor",
        "LayoutLMv3Processor",
        "LayoutLMv3Tokenizer",
    ],
    "models.layoutxlm": ["LayoutXLMProcessor"],
    "models.led": ["LEDConfig", "LEDTokenizer"],
    "models.levit": ["LevitConfig"],
    "models.lilt": ["LiltConfig"],
    "models.llama": ["LlamaConfig"],
    "models.llama4": [
        "Llama4Config",
        "Llama4Processor",
        "Llama4TextConfig",
        "Llama4VisionConfig",
    ],
    "models.llava": [
        "LlavaConfig",
        "LlavaProcessor",
    ],
    "models.llava_next": [
        "LlavaNextConfig",
        "LlavaNextProcessor",
    ],
    "models.llava_next_video": [
        "LlavaNextVideoConfig",
        "LlavaNextVideoProcessor",
    ],
    "models.llava_onevision": ["LlavaOnevisionConfig", "LlavaOnevisionProcessor"],
    "models.longformer": [
        "LongformerConfig",
        "LongformerTokenizer",
    ],
    "models.longt5": ["LongT5Config"],
    "models.luke": [
        "LukeConfig",
        "LukeTokenizer",
    ],
    "models.lxmert": [
        "LxmertConfig",
        "LxmertTokenizer",
    ],
    "models.m2m_100": ["M2M100Config"],
    "models.mamba": ["MambaConfig"],
    "models.mamba2": ["Mamba2Config"],
    "models.marian": ["MarianConfig"],
    "models.markuplm": [
        "MarkupLMConfig",
        "MarkupLMFeatureExtractor",
        "MarkupLMProcessor",
        "MarkupLMTokenizer",
    ],
    "models.mask2former": ["Mask2FormerConfig"],
    "models.maskformer": [
        "MaskFormerConfig",
        "MaskFormerSwinConfig",
    ],
    "models.mbart": ["MBartConfig"],
    "models.mbart50": [],
    "models.megatron_bert": ["MegatronBertConfig"],
    "models.megatron_gpt2": [],
    "models.mgp_str": [
        "MgpstrConfig",
        "MgpstrProcessor",
        "MgpstrTokenizer",
    ],
    "models.mimi": ["MimiConfig"],
    "models.mistral": ["MistralConfig"],
    "models.mistral3": ["Mistral3Config"],
    "models.mixtral": ["MixtralConfig"],
    "models.mllama": [
        "MllamaConfig",
        "MllamaProcessor",
    ],
    "models.mluke": [],
    "models.mobilebert": [
        "MobileBertConfig",
        "MobileBertTokenizer",
    ],
    "models.mobilenet_v1": ["MobileNetV1Config"],
    "models.mobilenet_v2": ["MobileNetV2Config"],
    "models.mobilevit": ["MobileViTConfig"],
    "models.mobilevitv2": ["MobileViTV2Config"],
    "models.modernbert": ["ModernBertConfig"],
    "models.moonshine": ["MoonshineConfig"],
    "models.moshi": [
        "MoshiConfig",
        "MoshiDepthConfig",
    ],
    "models.mpnet": [
        "MPNetConfig",
        "MPNetTokenizer",
    ],
    "models.mpt": ["MptConfig"],
    "models.mra": ["MraConfig"],
    "models.mt5": ["MT5Config"],
    "models.musicgen": [
        "MusicgenConfig",
        "MusicgenDecoderConfig",
    ],
    "models.musicgen_melody": [
        "MusicgenMelodyConfig",
        "MusicgenMelodyDecoderConfig",
    ],
    "models.mvp": ["MvpConfig", "MvpTokenizer"],
    "models.myt5": ["MyT5Tokenizer"],
    "models.nemotron": ["NemotronConfig"],
    "models.nllb": [],
    "models.nllb_moe": ["NllbMoeConfig"],
    "models.nougat": ["NougatProcessor"],
    "models.nystromformer": ["NystromformerConfig"],
    "models.olmo": ["OlmoConfig"],
    "models.olmo2": ["Olmo2Config"],
    "models.olmoe": ["OlmoeConfig"],
    "models.omdet_turbo": [
        "OmDetTurboConfig",
        "OmDetTurboProcessor",
    ],
    "models.oneformer": [
        "OneFormerConfig",
        "OneFormerProcessor",
    ],
    "models.openai": [
        "OpenAIGPTConfig",
        "OpenAIGPTTokenizer",
    ],
    "models.opt": ["OPTConfig"],
    "models.owlv2": [
        "Owlv2Config",
        "Owlv2Processor",
        "Owlv2TextConfig",
        "Owlv2VisionConfig",
    ],
    "models.owlvit": [
        "OwlViTConfig",
        "OwlViTProcessor",
        "OwlViTTextConfig",
        "OwlViTVisionConfig",
    ],
    "models.paligemma": ["PaliGemmaConfig"],
    "models.patchtsmixer": ["PatchTSMixerConfig"],
    "models.patchtst": ["PatchTSTConfig"],
    "models.pegasus": [
        "PegasusConfig",
        "PegasusTokenizer",
    ],
    "models.pegasus_x": ["PegasusXConfig"],
    "models.perceiver": [
        "PerceiverConfig",
        "PerceiverTokenizer",
    ],
    "models.persimmon": ["PersimmonConfig"],
    "models.phi": ["PhiConfig"],
    "models.phi3": ["Phi3Config"],
    "models.phi4_multimodal": [
        "Phi4MultimodalAudioConfig",
        "Phi4MultimodalConfig",
        "Phi4MultimodalFeatureExtractor",
        "Phi4MultimodalProcessor",
        "Phi4MultimodalVisionConfig",
    ],
    "models.phimoe": ["PhimoeConfig"],
    "models.phobert": ["PhobertTokenizer"],
    "models.pix2struct": [
        "Pix2StructConfig",
        "Pix2StructProcessor",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    "models.pixtral": ["PixtralProcessor", "PixtralVisionConfig"],
    "models.plbart": ["PLBartConfig"],
    "models.poolformer": ["PoolFormerConfig"],
    "models.pop2piano": ["Pop2PianoConfig"],
    "models.prompt_depth_anything": ["PromptDepthAnythingConfig"],
    "models.prophetnet": [
        "ProphetNetConfig",
        "ProphetNetTokenizer",
    ],
    "models.pvt": ["PvtConfig"],
    "models.pvt_v2": ["PvtV2Config"],
    "models.qwen2": [
        "Qwen2Config",
        "Qwen2Tokenizer",
    ],
    "models.qwen2_5_vl": [
        "Qwen2_5_VLConfig",
        "Qwen2_5_VLProcessor",
    ],
    "models.qwen2_audio": [
        "Qwen2AudioConfig",
        "Qwen2AudioEncoderConfig",
        "Qwen2AudioProcessor",
    ],
    "models.qwen2_moe": ["Qwen2MoeConfig"],
    "models.qwen2_vl": [
        "Qwen2VLConfig",
        "Qwen2VLProcessor",
    ],
    "models.qwen3": ["Qwen3Config"],
    "models.qwen3_moe": ["Qwen3MoeConfig"],
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    "models.recurrent_gemma": ["RecurrentGemmaConfig"],
    "models.reformer": ["ReformerConfig"],
    "models.regnet": ["RegNetConfig"],
    "models.rembert": ["RemBertConfig"],
    "models.resnet": ["ResNetConfig"],
    "models.roberta": [
        "RobertaConfig",
        "RobertaTokenizer",
    ],
    "models.roberta_prelayernorm": ["RobertaPreLayerNormConfig"],
    "models.roc_bert": [
        "RoCBertConfig",
        "RoCBertTokenizer",
    ],
    "models.roformer": [
        "RoFormerConfig",
        "RoFormerTokenizer",
    ],
    "models.rt_detr": ["RTDetrConfig", "RTDetrResNetConfig"],
    "models.rt_detr_v2": ["RTDetrV2Config"],
    "models.rwkv": ["RwkvConfig"],
    "models.sam": [
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamProcessor",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    "models.seamless_m4t": [
        "SeamlessM4TConfig",
        "SeamlessM4TFeatureExtractor",
        "SeamlessM4TProcessor",
    ],
    "models.seamless_m4t_v2": ["SeamlessM4Tv2Config"],
    "models.segformer": ["SegformerConfig"],
    "models.seggpt": ["SegGptConfig"],
    "models.sew": ["SEWConfig"],
    "models.sew_d": ["SEWDConfig"],
    "models.shieldgemma2": [
        "ShieldGemma2Config",
        "ShieldGemma2Processor",
    ],
    "models.siglip": [
        "SiglipConfig",
        "SiglipProcessor",
        "SiglipTextConfig",
        "SiglipVisionConfig",
    ],
    "models.siglip2": [
        "Siglip2Config",
        "Siglip2Processor",
        "Siglip2TextConfig",
        "Siglip2VisionConfig",
    ],
    "models.smolvlm": ["SmolVLMConfig"],
    "models.speech_encoder_decoder": ["SpeechEncoderDecoderConfig"],
    "models.speech_to_text": [
        "Speech2TextConfig",
        "Speech2TextFeatureExtractor",
        "Speech2TextProcessor",
    ],
    "models.speecht5": [
        "SpeechT5Config",
        "SpeechT5FeatureExtractor",
        "SpeechT5HifiGanConfig",
        "SpeechT5Processor",
    ],
    "models.splinter": [
        "SplinterConfig",
        "SplinterTokenizer",
    ],
    "models.squeezebert": [
        "SqueezeBertConfig",
        "SqueezeBertTokenizer",
    ],
    "models.stablelm": ["StableLmConfig"],
    "models.starcoder2": ["Starcoder2Config"],
    "models.superglue": ["SuperGlueConfig"],
    "models.superpoint": ["SuperPointConfig"],
    "models.swiftformer": ["SwiftFormerConfig"],
    "models.swin": ["SwinConfig"],
    "models.swin2sr": ["Swin2SRConfig"],
    "models.swinv2": ["Swinv2Config"],
    "models.switch_transformers": ["SwitchTransformersConfig"],
    "models.t5": ["T5Config"],
    "models.table_transformer": ["TableTransformerConfig"],
    "models.tapas": [
        "TapasConfig",
        "TapasTokenizer",
    ],
    "models.textnet": ["TextNetConfig"],
    "models.time_series_transformer": ["TimeSeriesTransformerConfig"],
    "models.timesformer": ["TimesformerConfig"],
    "models.timm_backbone": ["TimmBackboneConfig"],
    "models.timm_wrapper": ["TimmWrapperConfig"],
    "models.trocr": [
        "TrOCRConfig",
        "TrOCRProcessor",
    ],
    "models.tvp": [
        "TvpConfig",
        "TvpProcessor",
    ],
    "models.udop": [
        "UdopConfig",
        "UdopProcessor",
    ],
    "models.umt5": ["UMT5Config"],
    "models.unispeech": ["UniSpeechConfig"],
    "models.unispeech_sat": ["UniSpeechSatConfig"],
    "models.univnet": [
        "UnivNetConfig",
        "UnivNetFeatureExtractor",
    ],
    "models.upernet": ["UperNetConfig"],
    "models.video_llava": ["VideoLlavaConfig"],
    "models.videomae": ["VideoMAEConfig"],
    "models.vilt": [
        "ViltConfig",
        "ViltFeatureExtractor",
        "ViltImageProcessor",
        "ViltProcessor",
    ],
    "models.vipllava": ["VipLlavaConfig"],
    "models.vision_encoder_decoder": ["VisionEncoderDecoderConfig"],
    "models.vision_text_dual_encoder": [
        "VisionTextDualEncoderConfig",
        "VisionTextDualEncoderProcessor",
    ],
    "models.visual_bert": ["VisualBertConfig"],
    "models.vit": ["ViTConfig"],
    "models.vit_mae": ["ViTMAEConfig"],
    "models.vit_msn": ["ViTMSNConfig"],
    "models.vitdet": ["VitDetConfig"],
    "models.vitmatte": ["VitMatteConfig"],
    "models.vitpose": ["VitPoseConfig"],
    "models.vitpose_backbone": ["VitPoseBackboneConfig"],
    "models.vits": [
        "VitsConfig",
        "VitsTokenizer",
    ],
    "models.vivit": ["VivitConfig"],
    "models.wav2vec2": [
        "Wav2Vec2Config",
        "Wav2Vec2CTCTokenizer",
        "Wav2Vec2FeatureExtractor",
        "Wav2Vec2Processor",
        "Wav2Vec2Tokenizer",
    ],
    "models.wav2vec2_bert": [
        "Wav2Vec2BertConfig",
        "Wav2Vec2BertProcessor",
    ],
    "models.wav2vec2_conformer": ["Wav2Vec2ConformerConfig"],
    "models.wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"],
    "models.wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"],
    "models.wavlm": ["WavLMConfig"],
    "models.whisper": [
        "WhisperConfig",
        "WhisperFeatureExtractor",
        "WhisperProcessor",
        "WhisperTokenizer",
    ],
    "models.x_clip": [
        "XCLIPConfig",
        "XCLIPProcessor",
        "XCLIPTextConfig",
        "XCLIPVisionConfig",
    ],
    "models.xglm": ["XGLMConfig"],
    "models.xlm": ["XLMConfig", "XLMTokenizer"],
    "models.xlm_roberta": ["XLMRobertaConfig"],
    "models.xlm_roberta_xl": ["XLMRobertaXLConfig"],
    "models.xlnet": ["XLNetConfig"],
    "models.xmod": ["XmodConfig"],
    "models.yolos": ["YolosConfig"],
    "models.yoso": ["YosoConfig"],
    "models.zamba": ["ZambaConfig"],
    "models.zamba2": ["Zamba2Config"],
    "models.zoedepth": ["ZoeDepthConfig"],
    "onnx": [],
    "pipelines": [
        "AudioClassificationPipeline",
        "AutomaticSpeechRecognitionPipeline",
        "CsvPipelineDataFormat",
        "DepthEstimationPipeline",
        "DocumentQuestionAnsweringPipeline",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "ImageClassificationPipeline",
        "ImageFeatureExtractionPipeline",
        "ImageSegmentationPipeline",
        "ImageTextToTextPipeline",
        "ImageToImagePipeline",
        "ImageToTextPipeline",
        "JsonPipelineDataFormat",
        "MaskGenerationPipeline",
        "NerPipeline",
        "ObjectDetectionPipeline",
        "PipedPipelineDataFormat",
        "Pipeline",
        "PipelineDataFormat",
        "QuestionAnsweringPipeline",
        "SummarizationPipeline",
        "TableQuestionAnsweringPipeline",
        "Text2TextGenerationPipeline",
        "TextClassificationPipeline",
        "TextGenerationPipeline",
        "TextToAudioPipeline",
        "TokenClassificationPipeline",
        "TranslationPipeline",
        "VideoClassificationPipeline",
        "VisualQuestionAnsweringPipeline",
        "ZeroShotAudioClassificationPipeline",
        "ZeroShotClassificationPipeline",
        "ZeroShotImageClassificationPipeline",
        "ZeroShotObjectDetectionPipeline",
        "pipeline",
    ],
    "processing_utils": ["ProcessorMixin"],
    "quantizers": [],
    "testing_utils": [],
    "tokenization_utils": ["PreTrainedTokenizer"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "utils": [
        "CONFIG_NAME",
        "MODEL_CARD_NAME",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "SPIECE_UNDERLINE",
        "TRANSFORMERS_CACHE",
        "WEIGHTS_NAME",
        "TensorType",
        "add_end_docstrings",
        "add_start_docstrings",
        "is_apex_available",
        "is_av_available",
        "is_bitsandbytes_available",
        "is_datasets_available",
        "is_faiss_available",
        "is_flax_available",
        "is_keras_nlp_available",
        "is_phonemizer_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_pyctcdecode_available",
        "is_sacremoses_available",
        "is_safetensors_available",
        "is_scipy_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_speech_available",
        "is_tensorflow_text_available",
        "is_timm_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_hpu_available",
        "is_torch_mlu_available",
        "is_torch_musa_available",
        "is_torch_neuroncore_available",
        "is_torch_npu_available",
        "is_torchvision_available",
        "is_torch_xla_available",
        "is_torch_xpu_available",
        "is_vision_available",
        "logging",
    ],
    "utils.quantization_config": [
        "AqlmConfig",
        "AwqConfig",
        "BitNetConfig",
        "BitsAndBytesConfig",
        "CompressedTensorsConfig",
        "EetqConfig",
        "FbgemmFp8Config",
        "FineGrainedFP8Config",
        "GPTQConfig",
        "HiggsConfig",
        "HqqConfig",
        "QuantoConfig",
        "QuarkConfig",
        "SpQRConfig",
        "TorchAoConfig",
        "VptqConfig",
    ],
}

# sentencepiece-backed objects
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import dummy_sentencepiece_objects

    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.albert"].append("AlbertTokenizer")
    _import_structure["models.barthez"].append("BarthezTokenizer")
    _import_structure["models.bartpho"].append("BartphoTokenizer")
    _import_structure["models.bert_generation"].append("BertGenerationTokenizer")
    _import_structure["models.big_bird"].append("BigBirdTokenizer")
    _import_structure["models.camembert"].append("CamembertTokenizer")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizer")
    _import_structure["models.cpm"].append("CpmTokenizer")
    _import_structure["models.deberta_v2"].append("DebertaV2Tokenizer")
    _import_structure["models.deprecated.ernie_m"].append("ErnieMTokenizer")
    _import_structure["models.deprecated.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.fnet"].append("FNetTokenizer")
    _import_structure["models.gemma"].append("GemmaTokenizer")
    _import_structure["models.gpt_sw3"].append("GPTSw3Tokenizer")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizer")
    _import_structure["models.llama"].append("LlamaTokenizer")
    _import_structure["models.m2m_100"].append("M2M100Tokenizer")
    _import_structure["models.marian"].append("MarianTokenizer")
    _import_structure["models.mbart"].append("MBartTokenizer")
    _import_structure["models.mbart50"].append("MBart50Tokenizer")
    _import_structure["models.mluke"].append("MLukeTokenizer")
    _import_structure["models.mt5"].append("MT5Tokenizer")
    _import_structure["models.nllb"].append("NllbTokenizer")
    _import_structure["models.pegasus"].append("PegasusTokenizer")
    _import_structure["models.plbart"].append("PLBartTokenizer")
    _import_structure["models.reformer"].append("ReformerTokenizer")
    _import_structure["models.rembert"].append("RemBertTokenizer")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizer")
    _import_structure["models.siglip"].append("SiglipTokenizer")
    _import_structure["models.speech_to_text"].append("Speech2TextTokenizer")
    _import_structure["models.speecht5"].append("SpeechT5Tokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.udop"].append("UdopTokenizer")
    _import_structure["models.xglm"].append("XGLMTokenizer")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizer")
    _import_structure["models.xlnet"].append("XLNetTokenizer")

# tokenizers-backed objects
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import dummy_tokenizers_objects

    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # Fast tokenizers structure
    _import_structure["models.albert"].append("AlbertTokenizerFast")
    _import_structure["models.bart"].append("BartTokenizerFast")
    _import_structure["models.barthez"].append("BarthezTokenizerFast")
    _import_structure["models.bert"].append("BertTokenizerFast")
    _import_structure["models.big_bird"].append("BigBirdTokenizerFast")
    _import_structure["models.blenderbot"].append("BlenderbotTokenizerFast")
    _import_structure["models.blenderbot_small"].append("BlenderbotSmallTokenizerFast")
    _import_structure["models.bloom"].append("BloomTokenizerFast")
    _import_structure["models.camembert"].append("CamembertTokenizerFast")
    _import_structure["models.clip"].append("CLIPTokenizerFast")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizerFast")
    _import_structure["models.codegen"].append("CodeGenTokenizerFast")
    _import_structure["models.cohere"].append("CohereTokenizerFast")
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    _import_structure["models.cpm"].append("CpmTokenizerFast")
    _import_structure["models.deberta"].append("DebertaTokenizerFast")
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")
    _import_structure["models.deprecated.realm"].append("RealmTokenizerFast")
    _import_structure["models.deprecated.retribert"].append("RetriBertTokenizerFast")
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoderTokenizerFast",
            "DPRQuestionEncoderTokenizerFast",
            "DPRReaderTokenizerFast",
        ]
    )
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    _import_structure["models.fnet"].append("FNetTokenizerFast")
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    _import_structure["models.gemma"].append("GemmaTokenizerFast")
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    _import_structure["models.gpt_neox"].append("GPTNeoXTokenizerFast")
    _import_structure["models.gpt_neox_japanese"].append("GPTNeoXJapaneseTokenizer")
    _import_structure["models.herbert"].append("HerbertTokenizerFast")
    _import_structure["models.layoutlm"].append("LayoutLMTokenizerFast")
    _import_structure["models.layoutlmv2"].append("LayoutLMv2TokenizerFast")
    _import_structure["models.layoutlmv3"].append("LayoutLMv3TokenizerFast")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizerFast")
    _import_structure["models.led"].append("LEDTokenizerFast")
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    _import_structure["models.longformer"].append("LongformerTokenizerFast")
    _import_structure["models.lxmert"].append("LxmertTokenizerFast")
    _import_structure["models.markuplm"].append("MarkupLMTokenizerFast")
    _import_structure["models.mbart"].append("MBartTokenizerFast")
    _import_structure["models.mbart50"].append("MBart50TokenizerFast")
    _import_structure["models.mobilebert"].append("MobileBertTokenizerFast")
    _import_structure["models.mpnet"].append("MPNetTokenizerFast")
    _import_structure["models.mt5"].append("MT5TokenizerFast")
    _import_structure["models.mvp"].append("MvpTokenizerFast")
    _import_structure["models.nllb"].append("NllbTokenizerFast")
    _import_structure["models.nougat"].append("NougatTokenizerFast")
    _import_structure["models.openai"].append("OpenAIGPTTokenizerFast")
    _import_structure["models.pegasus"].append("PegasusTokenizerFast")
    _import_structure["models.qwen2"].append("Qwen2TokenizerFast")
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    _import_structure["models.rembert"].append("RemBertTokenizerFast")
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    _import_structure["models.roformer"].append("RoFormerTokenizerFast")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizerFast")
    _import_structure["models.splinter"].append("SplinterTokenizerFast")
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    _import_structure["models.t5"].append("T5TokenizerFast")
    _import_structure["models.udop"].append("UdopTokenizerFast")
    _import_structure["models.whisper"].append("WhisperTokenizerFast")
    _import_structure["models.xglm"].append("XGLMTokenizerFast")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizerFast")
    _import_structure["models.xlnet"].append("XLNetTokenizerFast")
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]


try:
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import dummy_sentencepiece_and_tokenizers_objects

    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

# Vision-specific objects
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import dummy_vision_objects

    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_base"] = ["ImageProcessingMixin"]
    _import_structure["image_processing_utils"] = ["BaseImageProcessor"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]
    _import_structure["models.aria"].extend(["AriaImageProcessor"])
    _import_structure["models.beit"].extend(["BeitFeatureExtractor", "BeitImageProcessor"])
    _import_structure["models.bit"].extend(["BitImageProcessor"])
    _import_structure["models.blip"].extend(["BlipImageProcessor"])
    _import_structure["models.bridgetower"].append("BridgeTowerImageProcessor")
    _import_structure["models.chameleon"].append("ChameleonImageProcessor")
    _import_structure["models.chinese_clip"].extend(["ChineseCLIPFeatureExtractor", "ChineseCLIPImageProcessor"])
    _import_structure["models.clip"].extend(["CLIPFeatureExtractor", "CLIPImageProcessor"])
    _import_structure["models.conditional_detr"].extend(
        ["ConditionalDetrFeatureExtractor", "ConditionalDetrImageProcessor"]
    )
    _import_structure["models.convnext"].extend(["ConvNextFeatureExtractor", "ConvNextImageProcessor"])
    _import_structure["models.deformable_detr"].extend(
        ["DeformableDetrFeatureExtractor", "DeformableDetrImageProcessor"]
    )
    _import_structure["models.deit"].extend(["DeiTFeatureExtractor", "DeiTImageProcessor"])
    _import_structure["models.deprecated.deta"].append("DetaImageProcessor")
    _import_structure["models.deprecated.efficientformer"].append("EfficientFormerImageProcessor")
    _import_structure["models.deprecated.tvlt"].append("TvltImageProcessor")
    _import_structure["models.deprecated.vit_hybrid"].extend(["ViTHybridImageProcessor"])
    _import_structure["models.depth_pro"].extend(["DepthProImageProcessor", "DepthProImageProcessorFast"])
    _import_structure["models.detr"].extend(["DetrFeatureExtractor", "DetrImageProcessor"])
    _import_structure["models.donut"].extend(["DonutFeatureExtractor", "DonutImageProcessor"])
    _import_structure["models.dpt"].extend(["DPTFeatureExtractor", "DPTImageProcessor"])
    _import_structure["models.efficientnet"].append("EfficientNetImageProcessor")
    _import_structure["models.emu3"].append("Emu3ImageProcessor")
    _import_structure["models.flava"].extend(["FlavaFeatureExtractor", "FlavaImageProcessor", "FlavaProcessor"])
    _import_structure["models.fuyu"].extend(["FuyuImageProcessor", "FuyuProcessor"])
    _import_structure["models.gemma3"].append("Gemma3ImageProcessor")
    _import_structure["models.glpn"].extend(["GLPNFeatureExtractor", "GLPNImageProcessor"])
    _import_structure["models.got_ocr2"].extend(["GotOcr2ImageProcessor"])
    _import_structure["models.grounding_dino"].extend(["GroundingDinoImageProcessor"])
    _import_structure["models.idefics"].extend(["IdeficsImageProcessor"])
    _import_structure["models.idefics2"].extend(["Idefics2ImageProcessor"])
    _import_structure["models.idefics3"].extend(["Idefics3ImageProcessor"])
    _import_structure["models.imagegpt"].extend(["ImageGPTFeatureExtractor", "ImageGPTImageProcessor"])
    _import_structure["models.instructblipvideo"].extend(["InstructBlipVideoImageProcessor"])
    _import_structure["models.layoutlmv2"].extend(["LayoutLMv2FeatureExtractor", "LayoutLMv2ImageProcessor"])
    _import_structure["models.layoutlmv3"].extend(["LayoutLMv3FeatureExtractor", "LayoutLMv3ImageProcessor"])
    _import_structure["models.levit"].extend(["LevitFeatureExtractor", "LevitImageProcessor"])
    _import_structure["models.llava"].append("LlavaImageProcessor")
    _import_structure["models.llava_next"].append("LlavaNextImageProcessor")
    _import_structure["models.llava_next_video"].append("LlavaNextVideoImageProcessor")
    _import_structure["models.llava_onevision"].extend(
        ["LlavaOnevisionImageProcessor", "LlavaOnevisionVideoProcessor"]
    )
    _import_structure["models.mask2former"].append("Mask2FormerImageProcessor")
    _import_structure["models.maskformer"].extend(["MaskFormerFeatureExtractor", "MaskFormerImageProcessor"])
    _import_structure["models.mllama"].extend(["MllamaImageProcessor"])
    _import_structure["models.mobilenet_v1"].extend(["MobileNetV1FeatureExtractor", "MobileNetV1ImageProcessor"])
    _import_structure["models.mobilenet_v2"].extend(["MobileNetV2FeatureExtractor", "MobileNetV2ImageProcessor"])
    _import_structure["models.mobilevit"].extend(["MobileViTFeatureExtractor", "MobileViTImageProcessor"])
    _import_structure["models.nougat"].append("NougatImageProcessor")
    _import_structure["models.oneformer"].extend(["OneFormerImageProcessor"])
    _import_structure["models.owlv2"].append("Owlv2ImageProcessor")
    _import_structure["models.owlvit"].extend(["OwlViTFeatureExtractor", "OwlViTImageProcessor"])
    _import_structure["models.perceiver"].extend(["PerceiverFeatureExtractor", "PerceiverImageProcessor"])
    _import_structure["models.pix2struct"].extend(["Pix2StructImageProcessor"])
    _import_structure["models.pixtral"].append("PixtralImageProcessor")
    _import_structure["models.poolformer"].extend(["PoolFormerFeatureExtractor", "PoolFormerImageProcessor"])
    _import_structure["models.prompt_depth_anything"].extend(["PromptDepthAnythingImageProcessor"])
    _import_structure["models.pvt"].extend(["PvtImageProcessor"])
    _import_structure["models.qwen2_vl"].extend(["Qwen2VLImageProcessor"])
    _import_structure["models.rt_detr"].extend(["RTDetrImageProcessor"])
    _import_structure["models.sam"].extend(["SamImageProcessor"])
    _import_structure["models.segformer"].extend(["SegformerFeatureExtractor", "SegformerImageProcessor"])
    _import_structure["models.seggpt"].extend(["SegGptImageProcessor"])
    _import_structure["models.siglip"].append("SiglipImageProcessor")
    _import_structure["models.siglip2"].append("Siglip2ImageProcessor")
    _import_structure["models.smolvlm"].extend(["SmolVLMImageProcessor"])
    _import_structure["models.superglue"].extend(["SuperGlueImageProcessor"])
    _import_structure["models.superpoint"].extend(["SuperPointImageProcessor"])
    _import_structure["models.swin2sr"].append("Swin2SRImageProcessor")
    _import_structure["models.textnet"].extend(["TextNetImageProcessor"])
    _import_structure["models.tvp"].append("TvpImageProcessor")
    _import_structure["models.video_llava"].append("VideoLlavaImageProcessor")
    _import_structure["models.videomae"].extend(["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"])
    _import_structure["models.vilt"].extend(["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"])
    _import_structure["models.vit"].extend(["ViTFeatureExtractor", "ViTImageProcessor"])
    _import_structure["models.vitmatte"].append("VitMatteImageProcessor")
    _import_structure["models.vitpose"].append("VitPoseImageProcessor")
    _import_structure["models.vivit"].append("VivitImageProcessor")
    _import_structure["models.yolos"].extend(["YolosFeatureExtractor", "YolosImageProcessor"])
    _import_structure["models.zoedepth"].append("ZoeDepthImageProcessor")

try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import dummy_torchvision_objects

    _import_structure["utils.dummy_torchvision_objects"] = [
        name for name in dir(dummy_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_utils_fast"] = ["BaseImageProcessorFast"]
    _import_structure["models.blip"].append("BlipImageProcessorFast")
    _import_structure["models.clip"].append("CLIPImageProcessorFast")
    _import_structure["models.convnext"].append("ConvNextImageProcessorFast")
    _import_structure["models.deformable_detr"].append("DeformableDetrImageProcessorFast")
    _import_structure["models.deit"].append("DeiTImageProcessorFast")
    _import_structure["models.depth_pro"].append("DepthProImageProcessorFast")
    _import_structure["models.detr"].append("DetrImageProcessorFast")
    _import_structure["models.gemma3"].append("Gemma3ImageProcessorFast")
    _import_structure["models.got_ocr2"].append("GotOcr2ImageProcessorFast")
    _import_structure["models.llama4"].append("Llama4ImageProcessorFast")
    _import_structure["models.llava"].append("LlavaImageProcessorFast")
    _import_structure["models.llava_next"].append("LlavaNextImageProcessorFast")
    _import_structure["models.llava_onevision"].append("LlavaOnevisionImageProcessorFast")
    _import_structure["models.phi4_multimodal"].append("Phi4MultimodalImageProcessorFast")
    _import_structure["models.pixtral"].append("PixtralImageProcessorFast")
    _import_structure["models.qwen2_vl"].append("Qwen2VLImageProcessorFast")
    _import_structure["models.rt_detr"].append("RTDetrImageProcessorFast")
    _import_structure["models.siglip"].append("SiglipImageProcessorFast")
    _import_structure["models.siglip2"].append("Siglip2ImageProcessorFast")
    _import_structure["models.vit"].append("ViTImageProcessorFast")

try:
    if not (is_torchvision_available() and is_timm_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import dummy_timm_and_torchvision_objects

    _import_structure["utils.dummy_timm_and_torchvision_objects"] = [
        name for name in dir(dummy_timm_and_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.timm_wrapper"].extend(["TimmWrapperImageProcessor"])

# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["model_debugging_utils"] = [
        "model_addition_debugger",
        "model_addition_debugger_context",
    ]
    _import_structure["activations"] = []
    _import_structure["cache_utils"] = [
        "Cache",
        "CacheConfig",
        "DynamicCache",
        "EncoderDecoderCache",
        "HQQQuantizedCache",
        "HybridCache",
        "MambaCache",
        "OffloadedCache",
        "OffloadedStaticCache",
        "QuantizedCache",
        "QuantizedCacheConfig",
        "QuantoQuantizedCache",
        "SinkCache",
        "SlidingWindowCache",
        "StaticCache",
    ]
    _import_structure["data.datasets"] = [
        "GlueDataset",
        "GlueDataTrainingArguments",
        "LineByLineTextDataset",
        "LineByLineWithRefDataset",
        "LineByLineWithSOPTextDataset",
        "SquadDataset",
        "SquadDataTrainingArguments",
        "TextDataset",
        "TextDatasetForNextSentencePrediction",
    ]
    _import_structure["generation"].extend(
        [
            "AlternatingCodebooksLogitsProcessor",
            "BayesianDetectorConfig",
            "BayesianDetectorModel",
            "BeamScorer",
            "BeamSearchScorer",
            "ClassifierFreeGuidanceLogitsProcessor",
            "ConstrainedBeamSearchScorer",
            "Constraint",
            "ConstraintListState",
            "DisjunctiveConstraint",
            "EncoderNoRepeatNGramLogitsProcessor",
            "EncoderRepetitionPenaltyLogitsProcessor",
            "EosTokenCriteria",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
            "ExponentialDecayLengthPenalty",
            "ForcedBOSTokenLogitsProcessor",
            "ForcedEOSTokenLogitsProcessor",
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "MinPLogitsWarper",
            "NoBadWordsLogitsProcessor",
            "NoRepeatNGramLogitsProcessor",
            "PhrasalConstraint",
            "PrefixConstrainedLogitsProcessor",
            "RepetitionPenaltyLogitsProcessor",
            "SequenceBiasLogitsProcessor",
            "StoppingCriteria",
            "StoppingCriteriaList",
            "StopStringCriteria",
            "SuppressTokensAtBeginLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "SynthIDTextWatermarkDetector",
            "SynthIDTextWatermarkingConfig",
            "SynthIDTextWatermarkLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WatermarkDetector",
            "WatermarkLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
        ]
    )

    _import_structure["modeling_flash_attention_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_rope_utils"] = ["ROPE_INIT_FUNCTIONS", "dynamic_rope_update"]
    _import_structure["modeling_utils"] = ["PreTrainedModel", "AttentionInterface"]

    # PyTorch models structure

    _import_structure["models.albert"].extend(
        [
            "AlbertForMaskedLM",
            "AlbertForMultipleChoice",
            "AlbertForPreTraining",
            "AlbertForQuestionAnswering",
            "AlbertForSequenceClassification",
            "AlbertForTokenClassification",
            "AlbertModel",
            "AlbertPreTrainedModel",
        ]
    )

    _import_structure["models.align"].extend(
        [
            "AlignModel",
            "AlignPreTrainedModel",
            "AlignTextModel",
            "AlignVisionModel",
        ]
    )
    _import_structure["models.altclip"].extend(
        [
            "AltCLIPModel",
            "AltCLIPPreTrainedModel",
            "AltCLIPTextModel",
            "AltCLIPVisionModel",
        ]
    )
    _import_structure["models.aria"].extend(
        [
            "AriaForConditionalGeneration",
            "AriaPreTrainedModel",
            "AriaTextForCausalLM",
            "AriaTextModel",
            "AriaTextPreTrainedModel",
        ]
    )
    _import_structure["models.audio_spectrogram_transformer"].extend(
        [
            "ASTForAudioClassification",
            "ASTModel",
            "ASTPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_XVECTOR_MAPPING",
            "MODEL_FOR_BACKBONE_MAPPING",
            "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MODEL_FOR_CTC_MAPPING",
            "MODEL_FOR_DEPTH_ESTIMATION_MAPPING",
            "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_IMAGE_MAPPING",
            "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING",
            "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING",
            "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING",
            "MODEL_FOR_KEYPOINT_DETECTION_MAPPING",
            "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MASK_GENERATION_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_OBJECT_DETECTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_RETRIEVAL_MAPPING",
            "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TEXT_ENCODING_MAPPING",
            "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING",
            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING",
            "MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING",
            "MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING",
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING",
            "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_VISION_2_SEQ_MAPPING",
            "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING",
            "MODEL_MAPPING",
            "MODEL_WITH_LM_HEAD_MAPPING",
            "AutoBackbone",
            "AutoModel",
            "AutoModelForAudioClassification",
            "AutoModelForAudioFrameClassification",
            "AutoModelForAudioXVector",
            "AutoModelForCausalLM",
            "AutoModelForCTC",
            "AutoModelForDepthEstimation",
            "AutoModelForDocumentQuestionAnswering",
            "AutoModelForImageClassification",
            "AutoModelForImageSegmentation",
            "AutoModelForImageTextToText",
            "AutoModelForImageToImage",
            "AutoModelForInstanceSegmentation",
            "AutoModelForKeypointDetection",
            "AutoModelForMaskedImageModeling",
            "AutoModelForMaskedLM",
            "AutoModelForMaskGeneration",
            "AutoModelForMultipleChoice",
            "AutoModelForNextSentencePrediction",
            "AutoModelForObjectDetection",
            "AutoModelForPreTraining",
            "AutoModelForQuestionAnswering",
            "AutoModelForSemanticSegmentation",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForTableQuestionAnswering",
            "AutoModelForTextEncoding",
            "AutoModelForTextToSpectrogram",
            "AutoModelForTextToWaveform",
            "AutoModelForTokenClassification",
            "AutoModelForUniversalSegmentation",
            "AutoModelForVideoClassification",
            "AutoModelForVision2Seq",
            "AutoModelForVisualQuestionAnswering",
            "AutoModelForZeroShotImageClassification",
            "AutoModelForZeroShotObjectDetection",
            "AutoModelWithLMHead",
        ]
    )
    _import_structure["models.autoformer"].extend(
        [
            "AutoformerForPrediction",
            "AutoformerModel",
            "AutoformerPreTrainedModel",
        ]
    )
    _import_structure["models.aya_vision"].extend(["AyaVisionForConditionalGeneration", "AyaVisionPreTrainedModel"])
    _import_structure["models.bamba"].extend(
        [
            "BambaForCausalLM",
            "BambaModel",
            "BambaPreTrainedModel",
        ]
    )
    _import_structure["models.bark"].extend(
        [
            "BarkCausalModel",
            "BarkCoarseModel",
            "BarkFineModel",
            "BarkModel",
            "BarkPreTrainedModel",
            "BarkSemanticModel",
        ]
    )
    _import_structure["models.bart"].extend(
        [
            "BartForCausalLM",
            "BartForConditionalGeneration",
            "BartForQuestionAnswering",
            "BartForSequenceClassification",
            "BartModel",
            "BartPretrainedModel",
            "BartPreTrainedModel",
            "PretrainedBartModel",
        ]
    )
    _import_structure["models.beit"].extend(
        [
            "BeitBackbone",
            "BeitForImageClassification",
            "BeitForMaskedImageModeling",
            "BeitForSemanticSegmentation",
            "BeitModel",
            "BeitPreTrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertForNextSentencePrediction",
            "BertForPreTraining",
            "BertForQuestionAnswering",
            "BertForSequenceClassification",
            "BertForTokenClassification",
            "BertLMHeadModel",
            "BertModel",
            "BertPreTrainedModel",
        ]
    )
    _import_structure["models.bert_generation"].extend(
        [
            "BertGenerationDecoder",
            "BertGenerationEncoder",
            "BertGenerationPreTrainedModel",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "BigBirdForCausalLM",
            "BigBirdForMaskedLM",
            "BigBirdForMultipleChoice",
            "BigBirdForPreTraining",
            "BigBirdForQuestionAnswering",
            "BigBirdForSequenceClassification",
            "BigBirdForTokenClassification",
            "BigBirdModel",
            "BigBirdPreTrainedModel",
        ]
    )
    _import_structure["models.bigbird_pegasus"].extend(
        [
            "BigBirdPegasusForCausalLM",
            "BigBirdPegasusForConditionalGeneration",
            "BigBirdPegasusForQuestionAnswering",
            "BigBirdPegasusForSequenceClassification",
            "BigBirdPegasusModel",
            "BigBirdPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.biogpt"].extend(
        [
            "BioGptForCausalLM",
            "BioGptForSequenceClassification",
            "BioGptForTokenClassification",
            "BioGptModel",
            "BioGptPreTrainedModel",
        ]
    )
    _import_structure["models.bit"].extend(
        [
            "BitBackbone",
            "BitForImageClassification",
            "BitModel",
            "BitPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "BlenderbotForCausalLM",
            "BlenderbotForConditionalGeneration",
            "BlenderbotModel",
            "BlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "BlenderbotSmallForCausalLM",
            "BlenderbotSmallForConditionalGeneration",
            "BlenderbotSmallModel",
            "BlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
            "BlipForConditionalGeneration",
            "BlipForImageTextRetrieval",
            "BlipForQuestionAnswering",
            "BlipModel",
            "BlipPreTrainedModel",
            "BlipTextModel",
            "BlipVisionModel",
        ]
    )
    _import_structure["models.blip_2"].extend(
        [
            "Blip2ForConditionalGeneration",
            "Blip2ForImageTextRetrieval",
            "Blip2Model",
            "Blip2PreTrainedModel",
            "Blip2QFormerModel",
            "Blip2TextModelWithProjection",
            "Blip2VisionModel",
            "Blip2VisionModelWithProjection",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "BloomForCausalLM",
            "BloomForQuestionAnswering",
            "BloomForSequenceClassification",
            "BloomForTokenClassification",
            "BloomModel",
            "BloomPreTrainedModel",
        ]
    )
    _import_structure["models.bridgetower"].extend(
        [
            "BridgeTowerForContrastiveLearning",
            "BridgeTowerForImageAndTextRetrieval",
            "BridgeTowerForMaskedLM",
            "BridgeTowerModel",
            "BridgeTowerPreTrainedModel",
        ]
    )
    _import_structure["models.bros"].extend(
        [
            "BrosForTokenClassification",
            "BrosModel",
            "BrosPreTrainedModel",
            "BrosProcessor",
            "BrosSpadeEEForTokenClassification",
            "BrosSpadeELForTokenClassification",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "CamembertForCausalLM",
            "CamembertForMaskedLM",
            "CamembertForMultipleChoice",
            "CamembertForQuestionAnswering",
            "CamembertForSequenceClassification",
            "CamembertForTokenClassification",
            "CamembertModel",
            "CamembertPreTrainedModel",
        ]
    )
    _import_structure["models.canine"].extend(
        [
            "CanineForMultipleChoice",
            "CanineForQuestionAnswering",
            "CanineForSequenceClassification",
            "CanineForTokenClassification",
            "CanineModel",
            "CaninePreTrainedModel",
        ]
    )
    _import_structure["models.chameleon"].extend(
        [
            "ChameleonForConditionalGeneration",
            "ChameleonModel",
            "ChameleonPreTrainedModel",
            "ChameleonProcessor",
            "ChameleonVQVAE",
        ]
    )
    _import_structure["models.chinese_clip"].extend(
        [
            "ChineseCLIPModel",
            "ChineseCLIPPreTrainedModel",
            "ChineseCLIPTextModel",
            "ChineseCLIPVisionModel",
        ]
    )
    _import_structure["models.clap"].extend(
        [
            "ClapAudioModel",
            "ClapAudioModelWithProjection",
            "ClapFeatureExtractor",
            "ClapModel",
            "ClapPreTrainedModel",
            "ClapTextModel",
            "ClapTextModelWithProjection",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "CLIPForImageClassification",
            "CLIPModel",
            "CLIPPreTrainedModel",
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPVisionModel",
            "CLIPVisionModelWithProjection",
        ]
    )
    _import_structure["models.clipseg"].extend(
        [
            "CLIPSegForImageSegmentation",
            "CLIPSegModel",
            "CLIPSegPreTrainedModel",
            "CLIPSegTextModel",
            "CLIPSegVisionModel",
        ]
    )
    _import_structure["models.clvp"].extend(
        [
            "ClvpDecoder",
            "ClvpEncoder",
            "ClvpForCausalLM",
            "ClvpModel",
            "ClvpModelForConditionalGeneration",
            "ClvpPreTrainedModel",
        ]
    )
    _import_structure["models.codegen"].extend(
        [
            "CodeGenForCausalLM",
            "CodeGenModel",
            "CodeGenPreTrainedModel",
        ]
    )
    _import_structure["models.cohere"].extend(["CohereForCausalLM", "CohereModel", "CoherePreTrainedModel"])
    _import_structure["models.cohere2"].extend(["Cohere2ForCausalLM", "Cohere2Model", "Cohere2PreTrainedModel"])
    _import_structure["models.colpali"].extend(
        [
            "ColPaliForRetrieval",
            "ColPaliPreTrainedModel",
        ]
    )
    _import_structure["models.conditional_detr"].extend(
        [
            "ConditionalDetrForObjectDetection",
            "ConditionalDetrForSegmentation",
            "ConditionalDetrModel",
            "ConditionalDetrPreTrainedModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "ConvBertForMaskedLM",
            "ConvBertForMultipleChoice",
            "ConvBertForQuestionAnswering",
            "ConvBertForSequenceClassification",
            "ConvBertForTokenClassification",
            "ConvBertModel",
            "ConvBertPreTrainedModel",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "ConvNextBackbone",
            "ConvNextForImageClassification",
            "ConvNextModel",
            "ConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "ConvNextV2Backbone",
            "ConvNextV2ForImageClassification",
            "ConvNextV2Model",
            "ConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.cpmant"].extend(
        [
            "CpmAntForCausalLM",
            "CpmAntModel",
            "CpmAntPreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "CTRLForSequenceClassification",
            "CTRLLMHeadModel",
            "CTRLModel",
            "CTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "CvtForImageClassification",
            "CvtModel",
            "CvtPreTrainedModel",
        ]
    )
    _import_structure["models.dab_detr"].extend(
        [
            "DabDetrForObjectDetection",
            "DabDetrModel",
            "DabDetrPreTrainedModel",
        ]
    )
    _import_structure["models.dac"].extend(
        [
            "DacModel",
            "DacPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
            "Data2VecAudioForAudioFrameClassification",
            "Data2VecAudioForCTC",
            "Data2VecAudioForSequenceClassification",
            "Data2VecAudioForXVector",
            "Data2VecAudioModel",
            "Data2VecAudioPreTrainedModel",
            "Data2VecTextForCausalLM",
            "Data2VecTextForMaskedLM",
            "Data2VecTextForMultipleChoice",
            "Data2VecTextForQuestionAnswering",
            "Data2VecTextForSequenceClassification",
            "Data2VecTextForTokenClassification",
            "Data2VecTextModel",
            "Data2VecTextPreTrainedModel",
            "Data2VecVisionForImageClassification",
            "Data2VecVisionForSemanticSegmentation",
            "Data2VecVisionModel",
            "Data2VecVisionPreTrainedModel",
        ]
    )
    _import_structure["models.dbrx"].extend(
        [
            "DbrxForCausalLM",
            "DbrxModel",
            "DbrxPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "DebertaForMaskedLM",
            "DebertaForQuestionAnswering",
            "DebertaForSequenceClassification",
            "DebertaForTokenClassification",
            "DebertaModel",
            "DebertaPreTrainedModel",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "DebertaV2ForMaskedLM",
            "DebertaV2ForMultipleChoice",
            "DebertaV2ForQuestionAnswering",
            "DebertaV2ForSequenceClassification",
            "DebertaV2ForTokenClassification",
            "DebertaV2Model",
            "DebertaV2PreTrainedModel",
        ]
    )
    _import_structure["models.decision_transformer"].extend(
        [
            "DecisionTransformerGPT2Model",
            "DecisionTransformerGPT2PreTrainedModel",
            "DecisionTransformerModel",
            "DecisionTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deepseek_v3"].extend(
        [
            "DeepseekV3ForCausalLM",
            "DeepseekV3Model",
            "DeepseekV3PreTrainedModel",
        ]
    )
    _import_structure["models.deformable_detr"].extend(
        [
            "DeformableDetrForObjectDetection",
            "DeformableDetrModel",
            "DeformableDetrPreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "DeiTForImageClassification",
            "DeiTForImageClassificationWithTeacher",
            "DeiTForMaskedImageModeling",
            "DeiTModel",
            "DeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.deta"].extend(
        [
            "DetaForObjectDetection",
            "DetaModel",
            "DetaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.efficientformer"].extend(
        [
            "EfficientFormerForImageClassification",
            "EfficientFormerForImageClassificationWithTeacher",
            "EfficientFormerModel",
            "EfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.ernie_m"].extend(
        [
            "ErnieMForInformationExtraction",
            "ErnieMForMultipleChoice",
            "ErnieMForQuestionAnswering",
            "ErnieMForSequenceClassification",
            "ErnieMForTokenClassification",
            "ErnieMModel",
            "ErnieMPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.gptsan_japanese"].extend(
        [
            "GPTSanJapaneseForConditionalGeneration",
            "GPTSanJapaneseModel",
            "GPTSanJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.graphormer"].extend(
        [
            "GraphormerForGraphClassification",
            "GraphormerModel",
            "GraphormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.jukebox"].extend(
        [
            "JukeboxModel",
            "JukeboxPreTrainedModel",
            "JukeboxPrior",
            "JukeboxVQVAE",
        ]
    )
    _import_structure["models.deprecated.mctct"].extend(
        [
            "MCTCTForCTC",
            "MCTCTModel",
            "MCTCTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mega"].extend(
        [
            "MegaForCausalLM",
            "MegaForMaskedLM",
            "MegaForMultipleChoice",
            "MegaForQuestionAnswering",
            "MegaForSequenceClassification",
            "MegaForTokenClassification",
            "MegaModel",
            "MegaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mmbt"].extend(["MMBTForClassification", "MMBTModel", "ModalEmbeddings"])
    _import_structure["models.deprecated.nat"].extend(
        [
            "NatBackbone",
            "NatForImageClassification",
            "NatModel",
            "NatPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.nezha"].extend(
        [
            "NezhaForMaskedLM",
            "NezhaForMultipleChoice",
            "NezhaForNextSentencePrediction",
            "NezhaForPreTraining",
            "NezhaForQuestionAnswering",
            "NezhaForSequenceClassification",
            "NezhaForTokenClassification",
            "NezhaModel",
            "NezhaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.open_llama"].extend(
        [
            "OpenLlamaForCausalLM",
            "OpenLlamaForSequenceClassification",
            "OpenLlamaModel",
            "OpenLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.qdqbert"].extend(
        [
            "QDQBertForMaskedLM",
            "QDQBertForMultipleChoice",
            "QDQBertForNextSentencePrediction",
            "QDQBertForQuestionAnswering",
            "QDQBertForSequenceClassification",
            "QDQBertForTokenClassification",
            "QDQBertLMHeadModel",
            "QDQBertModel",
            "QDQBertPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.realm"].extend(
        [
            "RealmEmbedder",
            "RealmForOpenQA",
            "RealmKnowledgeAugEncoder",
            "RealmPreTrainedModel",
            "RealmReader",
            "RealmRetriever",
            "RealmScorer",
        ]
    )
    _import_structure["models.deprecated.retribert"].extend(
        [
            "RetriBertModel",
            "RetriBertPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.speech_to_text_2"].extend(
        ["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"]
    )
    _import_structure["models.deprecated.trajectory_transformer"].extend(
        [
            "TrajectoryTransformerModel",
            "TrajectoryTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "AdaptiveEmbedding",
            "TransfoXLForSequenceClassification",
            "TransfoXLLMHeadModel",
            "TransfoXLModel",
            "TransfoXLPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.tvlt"].extend(
        [
            "TvltForAudioVisualClassification",
            "TvltForPreTraining",
            "TvltModel",
            "TvltPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.van"].extend(
        [
            "VanForImageClassification",
            "VanModel",
            "VanPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.vit_hybrid"].extend(
        [
            "ViTHybridForImageClassification",
            "ViTHybridModel",
            "ViTHybridPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.xlm_prophetnet"].extend(
        [
            "XLMProphetNetDecoder",
            "XLMProphetNetEncoder",
            "XLMProphetNetForCausalLM",
            "XLMProphetNetForConditionalGeneration",
            "XLMProphetNetModel",
            "XLMProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.depth_anything"].extend(
        [
            "DepthAnythingForDepthEstimation",
            "DepthAnythingPreTrainedModel",
        ]
    )
    _import_structure["models.depth_pro"].extend(
        [
            "DepthProForDepthEstimation",
            "DepthProModel",
            "DepthProPreTrainedModel",
        ]
    )
    _import_structure["models.detr"].extend(
        [
            "DetrForObjectDetection",
            "DetrForSegmentation",
            "DetrModel",
            "DetrPreTrainedModel",
        ]
    )
    _import_structure["models.diffllama"].extend(
        [
            "DiffLlamaForCausalLM",
            "DiffLlamaForQuestionAnswering",
            "DiffLlamaForSequenceClassification",
            "DiffLlamaForTokenClassification",
            "DiffLlamaModel",
            "DiffLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.dinat"].extend(
        [
            "DinatBackbone",
            "DinatForImageClassification",
            "DinatModel",
            "DinatPreTrainedModel",
        ]
    )
    _import_structure["models.dinov2"].extend(
        [
            "Dinov2Backbone",
            "Dinov2ForImageClassification",
            "Dinov2Model",
            "Dinov2PreTrainedModel",
        ]
    )
    _import_structure["models.dinov2_with_registers"].extend(
        [
            "Dinov2WithRegistersBackbone",
            "Dinov2WithRegistersForImageClassification",
            "Dinov2WithRegistersModel",
            "Dinov2WithRegistersPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "DistilBertForMaskedLM",
            "DistilBertForMultipleChoice",
            "DistilBertForQuestionAnswering",
            "DistilBertForSequenceClassification",
            "DistilBertForTokenClassification",
            "DistilBertModel",
            "DistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.donut"].extend(
        [
            "DonutSwinModel",
            "DonutSwinPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoder",
            "DPRPretrainedContextEncoder",
            "DPRPreTrainedModel",
            "DPRPretrainedQuestionEncoder",
            "DPRPretrainedReader",
            "DPRQuestionEncoder",
            "DPRReader",
        ]
    )
    _import_structure["models.dpt"].extend(
        [
            "DPTForDepthEstimation",
            "DPTForSemanticSegmentation",
            "DPTModel",
            "DPTPreTrainedModel",
        ]
    )
    _import_structure["models.efficientnet"].extend(
        [
            "EfficientNetForImageClassification",
            "EfficientNetModel",
            "EfficientNetPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "ElectraForCausalLM",
            "ElectraForMaskedLM",
            "ElectraForMultipleChoice",
            "ElectraForPreTraining",
            "ElectraForQuestionAnswering",
            "ElectraForSequenceClassification",
            "ElectraForTokenClassification",
            "ElectraModel",
            "ElectraPreTrainedModel",
        ]
    )
    _import_structure["models.emu3"].extend(
        [
            "Emu3ForCausalLM",
            "Emu3ForConditionalGeneration",
            "Emu3PreTrainedModel",
            "Emu3TextModel",
            "Emu3VQVAE",
        ]
    )
    _import_structure["models.encodec"].extend(
        [
            "EncodecModel",
            "EncodecPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    _import_structure["models.ernie"].extend(
        [
            "ErnieForCausalLM",
            "ErnieForMaskedLM",
            "ErnieForMultipleChoice",
            "ErnieForNextSentencePrediction",
            "ErnieForPreTraining",
            "ErnieForQuestionAnswering",
            "ErnieForSequenceClassification",
            "ErnieForTokenClassification",
            "ErnieModel",
            "ErniePreTrainedModel",
        ]
    )
    _import_structure["models.esm"].extend(
        [
            "EsmFoldPreTrainedModel",
            "EsmForMaskedLM",
            "EsmForProteinFolding",
            "EsmForSequenceClassification",
            "EsmForTokenClassification",
            "EsmModel",
            "EsmPreTrainedModel",
        ]
    )
    _import_structure["models.falcon"].extend(
        [
            "FalconForCausalLM",
            "FalconForQuestionAnswering",
            "FalconForSequenceClassification",
            "FalconForTokenClassification",
            "FalconModel",
            "FalconPreTrainedModel",
        ]
    )
    _import_structure["models.falcon_mamba"].extend(
        [
            "FalconMambaForCausalLM",
            "FalconMambaModel",
            "FalconMambaPreTrainedModel",
        ]
    )
    _import_structure["models.fastspeech2_conformer"].extend(
        [
            "FastSpeech2ConformerHifiGan",
            "FastSpeech2ConformerModel",
            "FastSpeech2ConformerPreTrainedModel",
            "FastSpeech2ConformerWithHifiGan",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "FlaubertForMultipleChoice",
            "FlaubertForQuestionAnswering",
            "FlaubertForQuestionAnsweringSimple",
            "FlaubertForSequenceClassification",
            "FlaubertForTokenClassification",
            "FlaubertModel",
            "FlaubertPreTrainedModel",
            "FlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.flava"].extend(
        [
            "FlavaForPreTraining",
            "FlavaImageCodebook",
            "FlavaImageModel",
            "FlavaModel",
            "FlavaMultimodalModel",
            "FlavaPreTrainedModel",
            "FlavaTextModel",
        ]
    )
    _import_structure["models.fnet"].extend(
        [
            "FNetForMaskedLM",
            "FNetForMultipleChoice",
            "FNetForNextSentencePrediction",
            "FNetForPreTraining",
            "FNetForQuestionAnswering",
            "FNetForSequenceClassification",
            "FNetForTokenClassification",
            "FNetModel",
            "FNetPreTrainedModel",
        ]
    )
    _import_structure["models.focalnet"].extend(
        [
            "FocalNetBackbone",
            "FocalNetForImageClassification",
            "FocalNetForMaskedImageModeling",
            "FocalNetModel",
            "FocalNetPreTrainedModel",
        ]
    )
    _import_structure["models.fsmt"].extend(["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"])
    _import_structure["models.funnel"].extend(
        [
            "FunnelBaseModel",
            "FunnelForMaskedLM",
            "FunnelForMultipleChoice",
            "FunnelForPreTraining",
            "FunnelForQuestionAnswering",
            "FunnelForSequenceClassification",
            "FunnelForTokenClassification",
            "FunnelModel",
            "FunnelPreTrainedModel",
        ]
    )
    _import_structure["models.fuyu"].extend(["FuyuForCausalLM", "FuyuPreTrainedModel"])
    _import_structure["models.gemma"].extend(
        [
            "GemmaForCausalLM",
            "GemmaForSequenceClassification",
            "GemmaForTokenClassification",
            "GemmaModel",
            "GemmaPreTrainedModel",
        ]
    )
    _import_structure["models.gemma2"].extend(
        [
            "Gemma2ForCausalLM",
            "Gemma2ForSequenceClassification",
            "Gemma2ForTokenClassification",
            "Gemma2Model",
            "Gemma2PreTrainedModel",
        ]
    )
    _import_structure["models.gemma3"].extend(
        [
            "Gemma3ForCausalLM",
            "Gemma3ForConditionalGeneration",
            "Gemma3PreTrainedModel",
            "Gemma3TextModel",
        ]
    )
    _import_structure["models.git"].extend(
        [
            "GitForCausalLM",
            "GitModel",
            "GitPreTrainedModel",
            "GitVisionModel",
        ]
    )
    _import_structure["models.glm"].extend(
        [
            "GlmForCausalLM",
            "GlmForSequenceClassification",
            "GlmForTokenClassification",
            "GlmModel",
            "GlmPreTrainedModel",
        ]
    )
    _import_structure["models.llama4"].extend(
        [
            "Llama4ForCausalLM",
            "Llama4ForConditionalGeneration",
            "Llama4TextModel",
            "Llama4VisionModel",
            "Llama4PreTrainedModel",
        ]
    )
    _import_structure["models.glpn"].extend(
        [
            "GLPNForDepthEstimation",
            "GLPNModel",
            "GLPNPreTrainedModel",
        ]
    )
    _import_structure["models.got_ocr2"].extend(
        [
            "GotOcr2ForConditionalGeneration",
            "GotOcr2PreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "GPT2DoubleHeadsModel",
            "GPT2ForQuestionAnswering",
            "GPT2ForSequenceClassification",
            "GPT2ForTokenClassification",
            "GPT2LMHeadModel",
            "GPT2Model",
            "GPT2PreTrainedModel",
        ]
    )
    _import_structure["models.gpt_bigcode"].extend(
        [
            "GPTBigCodeForCausalLM",
            "GPTBigCodeForSequenceClassification",
            "GPTBigCodeForTokenClassification",
            "GPTBigCodeModel",
            "GPTBigCodePreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neo"].extend(
        [
            "GPTNeoForCausalLM",
            "GPTNeoForQuestionAnswering",
            "GPTNeoForSequenceClassification",
            "GPTNeoForTokenClassification",
            "GPTNeoModel",
            "GPTNeoPreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neox"].extend(
        [
            "GPTNeoXForCausalLM",
            "GPTNeoXForQuestionAnswering",
            "GPTNeoXForSequenceClassification",
            "GPTNeoXForTokenClassification",
            "GPTNeoXModel",
            "GPTNeoXPreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neox_japanese"].extend(
        [
            "GPTNeoXJapaneseForCausalLM",
            "GPTNeoXJapaneseModel",
            "GPTNeoXJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "GPTJForCausalLM",
            "GPTJForQuestionAnswering",
            "GPTJForSequenceClassification",
            "GPTJModel",
            "GPTJPreTrainedModel",
        ]
    )
    _import_structure["models.granite"].extend(
        [
            "GraniteForCausalLM",
            "GraniteModel",
            "GranitePreTrainedModel",
        ]
    )
    _import_structure["models.granitemoe"].extend(
        [
            "GraniteMoeForCausalLM",
            "GraniteMoeModel",
            "GraniteMoePreTrainedModel",
        ]
    )

    _import_structure["models.granitemoeshared"].extend(
        [
            "GraniteMoeSharedForCausalLM",
            "GraniteMoeSharedModel",
            "GraniteMoeSharedPreTrainedModel",
        ]
    )
    _import_structure["models.grounding_dino"].extend(
        [
            "GroundingDinoForObjectDetection",
            "GroundingDinoModel",
            "GroundingDinoPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "GroupViTModel",
            "GroupViTPreTrainedModel",
            "GroupViTTextModel",
            "GroupViTVisionModel",
        ]
    )
    _import_structure["models.helium"].extend(
        [
            "HeliumForCausalLM",
            "HeliumForSequenceClassification",
            "HeliumForTokenClassification",
            "HeliumModel",
            "HeliumPreTrainedModel",
        ]
    )
    _import_structure["models.hiera"].extend(
        [
            "HieraBackbone",
            "HieraForImageClassification",
            "HieraForPreTraining",
            "HieraModel",
            "HieraPreTrainedModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "HubertForCTC",
            "HubertForSequenceClassification",
            "HubertModel",
            "HubertPreTrainedModel",
        ]
    )
    _import_structure["models.ibert"].extend(
        [
            "IBertForMaskedLM",
            "IBertForMultipleChoice",
            "IBertForQuestionAnswering",
            "IBertForSequenceClassification",
            "IBertForTokenClassification",
            "IBertModel",
            "IBertPreTrainedModel",
        ]
    )
    _import_structure["models.idefics"].extend(
        [
            "IdeficsForVisionText2Text",
            "IdeficsModel",
            "IdeficsPreTrainedModel",
            "IdeficsProcessor",
        ]
    )
    _import_structure["models.idefics2"].extend(
        [
            "Idefics2ForConditionalGeneration",
            "Idefics2Model",
            "Idefics2PreTrainedModel",
            "Idefics2Processor",
        ]
    )
    _import_structure["models.idefics3"].extend(
        [
            "Idefics3ForConditionalGeneration",
            "Idefics3Model",
            "Idefics3PreTrainedModel",
            "Idefics3Processor",
            "Idefics3VisionConfig",
            "Idefics3VisionTransformer",
        ]
    )
    _import_structure["models.ijepa"].extend(
        [
            "IJepaForImageClassification",
            "IJepaModel",
            "IJepaPreTrainedModel",
        ]
    )
    _import_structure["models.imagegpt"].extend(
        [
            "ImageGPTForCausalImageModeling",
            "ImageGPTForImageClassification",
            "ImageGPTModel",
            "ImageGPTPreTrainedModel",
        ]
    )
    _import_structure["models.informer"].extend(
        [
            "InformerForPrediction",
            "InformerModel",
            "InformerPreTrainedModel",
        ]
    )
    _import_structure["models.instructblip"].extend(
        [
            "InstructBlipForConditionalGeneration",
            "InstructBlipPreTrainedModel",
            "InstructBlipQFormerModel",
            "InstructBlipVisionModel",
        ]
    )
    _import_structure["models.instructblipvideo"].extend(
        [
            "InstructBlipVideoForConditionalGeneration",
            "InstructBlipVideoPreTrainedModel",
            "InstructBlipVideoQFormerModel",
            "InstructBlipVideoVisionModel",
        ]
    )
    _import_structure["models.jamba"].extend(
        [
            "JambaForCausalLM",
            "JambaForSequenceClassification",
            "JambaModel",
            "JambaPreTrainedModel",
        ]
    )
    _import_structure["models.jetmoe"].extend(
        [
            "JetMoeForCausalLM",
            "JetMoeForSequenceClassification",
            "JetMoeModel",
            "JetMoePreTrainedModel",
        ]
    )
    _import_structure["models.kosmos2"].extend(
        [
            "Kosmos2ForConditionalGeneration",
            "Kosmos2Model",
            "Kosmos2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
            "LayoutLMForMaskedLM",
            "LayoutLMForQuestionAnswering",
            "LayoutLMForSequenceClassification",
            "LayoutLMForTokenClassification",
            "LayoutLMModel",
            "LayoutLMPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv2"].extend(
        [
            "LayoutLMv2ForQuestionAnswering",
            "LayoutLMv2ForSequenceClassification",
            "LayoutLMv2ForTokenClassification",
            "LayoutLMv2Model",
            "LayoutLMv2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "LayoutLMv3ForQuestionAnswering",
            "LayoutLMv3ForSequenceClassification",
            "LayoutLMv3ForTokenClassification",
            "LayoutLMv3Model",
            "LayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(
        [
            "LEDForConditionalGeneration",
            "LEDForQuestionAnswering",
            "LEDForSequenceClassification",
            "LEDModel",
            "LEDPreTrainedModel",
        ]
    )
    _import_structure["models.levit"].extend(
        [
            "LevitForImageClassification",
            "LevitForImageClassificationWithTeacher",
            "LevitModel",
            "LevitPreTrainedModel",
        ]
    )
    _import_structure["models.lilt"].extend(
        [
            "LiltForQuestionAnswering",
            "LiltForSequenceClassification",
            "LiltForTokenClassification",
            "LiltModel",
            "LiltPreTrainedModel",
        ]
    )
    _import_structure["models.llama"].extend(
        [
            "LlamaForCausalLM",
            "LlamaForQuestionAnswering",
            "LlamaForSequenceClassification",
            "LlamaForTokenClassification",
            "LlamaModel",
            "LlamaPreTrainedModel",
        ]
    )
    _import_structure["models.llava"].extend(
        [
            "LlavaForConditionalGeneration",
            "LlavaPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next"].extend(
        [
            "LlavaNextForConditionalGeneration",
            "LlavaNextPreTrainedModel",
        ]
    )
    _import_structure["models.phi4_multimodal"].extend(
        [
            "Phi4MultimodalForCausalLM",
            "Phi4MultimodalPreTrainedModel",
            "Phi4MultimodalAudioModel",
            "Phi4MultimodalAudioPreTrainedModel",
            "Phi4MultimodalModel",
            "Phi4MultimodalVisionModel",
            "Phi4MultimodalVisionPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next_video"].extend(
        [
            "LlavaNextVideoForConditionalGeneration",
            "LlavaNextVideoPreTrainedModel",
        ]
    )
    _import_structure["models.llava_onevision"].extend(
        [
            "LlavaOnevisionForConditionalGeneration",
            "LlavaOnevisionPreTrainedModel",
        ]
    )
    _import_structure["models.longformer"].extend(
        [
            "LongformerForMaskedLM",
            "LongformerForMultipleChoice",
            "LongformerForQuestionAnswering",
            "LongformerForSequenceClassification",
            "LongformerForTokenClassification",
            "LongformerModel",
            "LongformerPreTrainedModel",
        ]
    )
    _import_structure["models.longt5"].extend(
        [
            "LongT5EncoderModel",
            "LongT5ForConditionalGeneration",
            "LongT5Model",
            "LongT5PreTrainedModel",
        ]
    )
    _import_structure["models.luke"].extend(
        [
            "LukeForEntityClassification",
            "LukeForEntityPairClassification",
            "LukeForEntitySpanClassification",
            "LukeForMaskedLM",
            "LukeForMultipleChoice",
            "LukeForQuestionAnswering",
            "LukeForSequenceClassification",
            "LukeForTokenClassification",
            "LukeModel",
            "LukePreTrainedModel",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "LxmertEncoder",
            "LxmertForPreTraining",
            "LxmertForQuestionAnswering",
            "LxmertModel",
            "LxmertPreTrainedModel",
            "LxmertVisualFeatureEncoder",
        ]
    )
    _import_structure["models.m2m_100"].extend(
        [
            "M2M100ForConditionalGeneration",
            "M2M100Model",
            "M2M100PreTrainedModel",
        ]
    )
    _import_structure["models.mamba"].extend(
        [
            "MambaForCausalLM",
            "MambaModel",
            "MambaPreTrainedModel",
        ]
    )
    _import_structure["models.mamba2"].extend(
        [
            "Mamba2ForCausalLM",
            "Mamba2Model",
            "Mamba2PreTrainedModel",
        ]
    )
    _import_structure["models.marian"].extend(
        ["MarianForCausalLM", "MarianModel", "MarianMTModel", "MarianPreTrainedModel"]
    )
    _import_structure["models.markuplm"].extend(
        [
            "MarkupLMForQuestionAnswering",
            "MarkupLMForSequenceClassification",
            "MarkupLMForTokenClassification",
            "MarkupLMModel",
            "MarkupLMPreTrainedModel",
        ]
    )
    _import_structure["models.mask2former"].extend(
        [
            "Mask2FormerForUniversalSegmentation",
            "Mask2FormerModel",
            "Mask2FormerPreTrainedModel",
        ]
    )
    _import_structure["models.maskformer"].extend(
        [
            "MaskFormerForInstanceSegmentation",
            "MaskFormerModel",
            "MaskFormerPreTrainedModel",
            "MaskFormerSwinBackbone",
        ]
    )
    _import_structure["models.mbart"].extend(
        [
            "MBartForCausalLM",
            "MBartForConditionalGeneration",
            "MBartForQuestionAnswering",
            "MBartForSequenceClassification",
            "MBartModel",
            "MBartPreTrainedModel",
        ]
    )
    _import_structure["models.megatron_bert"].extend(
        [
            "MegatronBertForCausalLM",
            "MegatronBertForMaskedLM",
            "MegatronBertForMultipleChoice",
            "MegatronBertForNextSentencePrediction",
            "MegatronBertForPreTraining",
            "MegatronBertForQuestionAnswering",
            "MegatronBertForSequenceClassification",
            "MegatronBertForTokenClassification",
            "MegatronBertModel",
            "MegatronBertPreTrainedModel",
        ]
    )
    _import_structure["models.mgp_str"].extend(
        [
            "MgpstrForSceneTextRecognition",
            "MgpstrModel",
            "MgpstrPreTrainedModel",
        ]
    )
    _import_structure["models.mimi"].extend(
        [
            "MimiModel",
            "MimiPreTrainedModel",
        ]
    )
    _import_structure["models.mistral"].extend(
        [
            "MistralForCausalLM",
            "MistralForQuestionAnswering",
            "MistralForSequenceClassification",
            "MistralForTokenClassification",
            "MistralModel",
            "MistralPreTrainedModel",
        ]
    )
    _import_structure["models.mistral3"].extend(
        [
            "Mistral3ForConditionalGeneration",
            "Mistral3PreTrainedModel",
        ]
    )
    _import_structure["models.mixtral"].extend(
        [
            "MixtralForCausalLM",
            "MixtralForQuestionAnswering",
            "MixtralForSequenceClassification",
            "MixtralForTokenClassification",
            "MixtralModel",
            "MixtralPreTrainedModel",
        ]
    )
    _import_structure["models.mllama"].extend(
        [
            "MllamaForCausalLM",
            "MllamaForConditionalGeneration",
            "MllamaPreTrainedModel",
            "MllamaProcessor",
            "MllamaTextModel",
            "MllamaVisionModel",
        ]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "MobileBertForMaskedLM",
            "MobileBertForMultipleChoice",
            "MobileBertForNextSentencePrediction",
            "MobileBertForPreTraining",
            "MobileBertForQuestionAnswering",
            "MobileBertForSequenceClassification",
            "MobileBertForTokenClassification",
            "MobileBertModel",
            "MobileBertPreTrainedModel",
        ]
    )
    _import_structure["models.mobilenet_v1"].extend(
        [
            "MobileNetV1ForImageClassification",
            "MobileNetV1Model",
            "MobileNetV1PreTrainedModel",
        ]
    )
    _import_structure["models.mobilenet_v2"].extend(
        [
            "MobileNetV2ForImageClassification",
            "MobileNetV2ForSemanticSegmentation",
            "MobileNetV2Model",
            "MobileNetV2PreTrainedModel",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "MobileViTForImageClassification",
            "MobileViTForSemanticSegmentation",
            "MobileViTModel",
            "MobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevitv2"].extend(
        [
            "MobileViTV2ForImageClassification",
            "MobileViTV2ForSemanticSegmentation",
            "MobileViTV2Model",
            "MobileViTV2PreTrainedModel",
        ]
    )
    _import_structure["models.modernbert"].extend(
        [
            "ModernBertForMaskedLM",
            "ModernBertForQuestionAnswering",
            "ModernBertForSequenceClassification",
            "ModernBertForTokenClassification",
            "ModernBertModel",
            "ModernBertPreTrainedModel",
        ]
    )
    _import_structure["models.moonshine"].extend(
        [
            "MoonshineForConditionalGeneration",
            "MoonshineModel",
            "MoonshinePreTrainedModel",
        ]
    )
    _import_structure["models.moshi"].extend(
        [
            "MoshiForCausalLM",
            "MoshiForConditionalGeneration",
            "MoshiModel",
            "MoshiPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "MPNetForMaskedLM",
            "MPNetForMultipleChoice",
            "MPNetForQuestionAnswering",
            "MPNetForSequenceClassification",
            "MPNetForTokenClassification",
            "MPNetModel",
            "MPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mpt"].extend(
        [
            "MptForCausalLM",
            "MptForQuestionAnswering",
            "MptForSequenceClassification",
            "MptForTokenClassification",
            "MptModel",
            "MptPreTrainedModel",
        ]
    )
    _import_structure["models.mra"].extend(
        [
            "MraForMaskedLM",
            "MraForMultipleChoice",
            "MraForQuestionAnswering",
            "MraForSequenceClassification",
            "MraForTokenClassification",
            "MraModel",
            "MraPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(
        [
            "MT5EncoderModel",
            "MT5ForConditionalGeneration",
            "MT5ForQuestionAnswering",
            "MT5ForSequenceClassification",
            "MT5ForTokenClassification",
            "MT5Model",
            "MT5PreTrainedModel",
        ]
    )
    _import_structure["models.musicgen"].extend(
        [
            "MusicgenForCausalLM",
            "MusicgenForConditionalGeneration",
            "MusicgenModel",
            "MusicgenPreTrainedModel",
            "MusicgenProcessor",
        ]
    )
    _import_structure["models.musicgen_melody"].extend(
        [
            "MusicgenMelodyForCausalLM",
            "MusicgenMelodyForConditionalGeneration",
            "MusicgenMelodyModel",
            "MusicgenMelodyPreTrainedModel",
        ]
    )
    _import_structure["models.mvp"].extend(
        [
            "MvpForCausalLM",
            "MvpForConditionalGeneration",
            "MvpForQuestionAnswering",
            "MvpForSequenceClassification",
            "MvpModel",
            "MvpPreTrainedModel",
        ]
    )
    _import_structure["models.nemotron"].extend(
        [
            "NemotronForCausalLM",
            "NemotronForQuestionAnswering",
            "NemotronForSequenceClassification",
            "NemotronForTokenClassification",
            "NemotronModel",
            "NemotronPreTrainedModel",
        ]
    )
    _import_structure["models.nllb_moe"].extend(
        [
            "NllbMoeForConditionalGeneration",
            "NllbMoeModel",
            "NllbMoePreTrainedModel",
            "NllbMoeSparseMLP",
            "NllbMoeTop2Router",
        ]
    )
    _import_structure["models.nystromformer"].extend(
        [
            "NystromformerForMaskedLM",
            "NystromformerForMultipleChoice",
            "NystromformerForQuestionAnswering",
            "NystromformerForSequenceClassification",
            "NystromformerForTokenClassification",
            "NystromformerModel",
            "NystromformerPreTrainedModel",
        ]
    )
    _import_structure["models.olmo"].extend(
        [
            "OlmoForCausalLM",
            "OlmoModel",
            "OlmoPreTrainedModel",
        ]
    )
    _import_structure["models.olmo2"].extend(
        [
            "Olmo2ForCausalLM",
            "Olmo2Model",
            "Olmo2PreTrainedModel",
        ]
    )
    _import_structure["models.olmoe"].extend(
        [
            "OlmoeForCausalLM",
            "OlmoeModel",
            "OlmoePreTrainedModel",
        ]
    )
    _import_structure["models.omdet_turbo"].extend(
        [
            "OmDetTurboForObjectDetection",
            "OmDetTurboPreTrainedModel",
        ]
    )
    _import_structure["models.oneformer"].extend(
        [
            "OneFormerForUniversalSegmentation",
            "OneFormerModel",
            "OneFormerPreTrainedModel",
        ]
    )
    _import_structure["models.openai"].extend(
        [
            "OpenAIGPTDoubleHeadsModel",
            "OpenAIGPTForSequenceClassification",
            "OpenAIGPTLMHeadModel",
            "OpenAIGPTModel",
            "OpenAIGPTPreTrainedModel",
        ]
    )
    _import_structure["models.opt"].extend(
        [
            "OPTForCausalLM",
            "OPTForQuestionAnswering",
            "OPTForSequenceClassification",
            "OPTModel",
            "OPTPreTrainedModel",
        ]
    )
    _import_structure["models.owlv2"].extend(
        [
            "Owlv2ForObjectDetection",
            "Owlv2Model",
            "Owlv2PreTrainedModel",
            "Owlv2TextModel",
            "Owlv2VisionModel",
        ]
    )
    _import_structure["models.owlvit"].extend(
        [
            "OwlViTForObjectDetection",
            "OwlViTModel",
            "OwlViTPreTrainedModel",
            "OwlViTTextModel",
            "OwlViTVisionModel",
        ]
    )
    _import_structure["models.paligemma"].extend(
        [
            "PaliGemmaForConditionalGeneration",
            "PaliGemmaPreTrainedModel",
            "PaliGemmaProcessor",
        ]
    )
    _import_structure["models.patchtsmixer"].extend(
        [
            "PatchTSMixerForPrediction",
            "PatchTSMixerForPretraining",
            "PatchTSMixerForRegression",
            "PatchTSMixerForTimeSeriesClassification",
            "PatchTSMixerModel",
            "PatchTSMixerPreTrainedModel",
        ]
    )
    _import_structure["models.patchtst"].extend(
        [
            "PatchTSTForClassification",
            "PatchTSTForPrediction",
            "PatchTSTForPretraining",
            "PatchTSTForRegression",
            "PatchTSTModel",
            "PatchTSTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "PegasusForCausalLM",
            "PegasusForConditionalGeneration",
            "PegasusModel",
            "PegasusPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus_x"].extend(
        [
            "PegasusXForConditionalGeneration",
            "PegasusXModel",
            "PegasusXPreTrainedModel",
        ]
    )
    _import_structure["models.perceiver"].extend(
        [
            "PerceiverForImageClassificationConvProcessing",
            "PerceiverForImageClassificationFourier",
            "PerceiverForImageClassificationLearned",
            "PerceiverForMaskedLM",
            "PerceiverForMultimodalAutoencoding",
            "PerceiverForOpticalFlow",
            "PerceiverForSequenceClassification",
            "PerceiverModel",
            "PerceiverPreTrainedModel",
        ]
    )
    _import_structure["models.persimmon"].extend(
        [
            "PersimmonForCausalLM",
            "PersimmonForSequenceClassification",
            "PersimmonForTokenClassification",
            "PersimmonModel",
            "PersimmonPreTrainedModel",
        ]
    )
    _import_structure["models.phi"].extend(
        [
            "PhiForCausalLM",
            "PhiForSequenceClassification",
            "PhiForTokenClassification",
            "PhiModel",
            "PhiPreTrainedModel",
        ]
    )
    _import_structure["models.phi3"].extend(
        [
            "Phi3ForCausalLM",
            "Phi3ForSequenceClassification",
            "Phi3ForTokenClassification",
            "Phi3Model",
            "Phi3PreTrainedModel",
        ]
    )
    _import_structure["models.phimoe"].extend(
        [
            "PhimoeForCausalLM",
            "PhimoeForSequenceClassification",
            "PhimoeModel",
            "PhimoePreTrainedModel",
        ]
    )
    _import_structure["models.pix2struct"].extend(
        [
            "Pix2StructForConditionalGeneration",
            "Pix2StructPreTrainedModel",
            "Pix2StructTextModel",
            "Pix2StructVisionModel",
        ]
    )
    _import_structure["models.pixtral"].extend(["PixtralPreTrainedModel", "PixtralVisionModel"])
    _import_structure["models.plbart"].extend(
        [
            "PLBartForCausalLM",
            "PLBartForConditionalGeneration",
            "PLBartForSequenceClassification",
            "PLBartModel",
            "PLBartPreTrainedModel",
        ]
    )
    _import_structure["models.poolformer"].extend(
        [
            "PoolFormerForImageClassification",
            "PoolFormerModel",
            "PoolFormerPreTrainedModel",
        ]
    )
    _import_structure["models.pop2piano"].extend(
        [
            "Pop2PianoForConditionalGeneration",
            "Pop2PianoPreTrainedModel",
        ]
    )
    _import_structure["models.prompt_depth_anything"].extend(
        [
            "PromptDepthAnythingForDepthEstimation",
            "PromptDepthAnythingPreTrainedModel",
        ]
    )
    _import_structure["models.prophetnet"].extend(
        [
            "ProphetNetDecoder",
            "ProphetNetEncoder",
            "ProphetNetForCausalLM",
            "ProphetNetForConditionalGeneration",
            "ProphetNetModel",
            "ProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.pvt"].extend(
        [
            "PvtForImageClassification",
            "PvtModel",
            "PvtPreTrainedModel",
        ]
    )
    _import_structure["models.pvt_v2"].extend(
        [
            "PvtV2Backbone",
            "PvtV2ForImageClassification",
            "PvtV2Model",
            "PvtV2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2"].extend(
        [
            "Qwen2ForCausalLM",
            "Qwen2ForQuestionAnswering",
            "Qwen2ForSequenceClassification",
            "Qwen2ForTokenClassification",
            "Qwen2Model",
            "Qwen2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_5_vl"].extend(
        [
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2_5_VLModel",
            "Qwen2_5_VLPreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_audio"].extend(
        [
            "Qwen2AudioEncoder",
            "Qwen2AudioForConditionalGeneration",
            "Qwen2AudioPreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_moe"].extend(
        [
            "Qwen2MoeForCausalLM",
            "Qwen2MoeForQuestionAnswering",
            "Qwen2MoeForSequenceClassification",
            "Qwen2MoeForTokenClassification",
            "Qwen2MoeModel",
            "Qwen2MoePreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_vl"].extend(
        [
            "Qwen2VLForConditionalGeneration",
            "Qwen2VLModel",
            "Qwen2VLPreTrainedModel",
        ]
    )
    _import_structure["models.qwen3"].extend(
        [
            "Qwen3ForCausalLM",
            "Qwen3ForQuestionAnswering",
            "Qwen3ForSequenceClassification",
            "Qwen3ForTokenClassification",
            "Qwen3Model",
            "Qwen3PreTrainedModel",
        ]
    )
    _import_structure["models.qwen3_moe"].extend(
        [
            "Qwen3MoeForCausalLM",
            "Qwen3MoeForQuestionAnswering",
            "Qwen3MoeForSequenceClassification",
            "Qwen3MoeForTokenClassification",
            "Qwen3MoeModel",
            "Qwen3MoePreTrainedModel",
        ]
    )
    _import_structure["models.rag"].extend(
        [
            "RagModel",
            "RagPreTrainedModel",
            "RagSequenceForGeneration",
            "RagTokenForGeneration",
        ]
    )
    _import_structure["models.recurrent_gemma"].extend(
        [
            "RecurrentGemmaForCausalLM",
            "RecurrentGemmaModel",
            "RecurrentGemmaPreTrainedModel",
        ]
    )
    _import_structure["models.reformer"].extend(
        [
            "ReformerForMaskedLM",
            "ReformerForQuestionAnswering",
            "ReformerForSequenceClassification",
            "ReformerModel",
            "ReformerModelWithLMHead",
            "ReformerPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "RegNetForImageClassification",
            "RegNetModel",
            "RegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "RemBertForCausalLM",
            "RemBertForMaskedLM",
            "RemBertForMultipleChoice",
            "RemBertForQuestionAnswering",
            "RemBertForSequenceClassification",
            "RemBertForTokenClassification",
            "RemBertModel",
            "RemBertPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "ResNetBackbone",
            "ResNetForImageClassification",
            "ResNetModel",
            "ResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "RobertaForCausalLM",
            "RobertaForMaskedLM",
            "RobertaForMultipleChoice",
            "RobertaForQuestionAnswering",
            "RobertaForSequenceClassification",
            "RobertaForTokenClassification",
            "RobertaModel",
            "RobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "RobertaPreLayerNormForCausalLM",
            "RobertaPreLayerNormForMaskedLM",
            "RobertaPreLayerNormForMultipleChoice",
            "RobertaPreLayerNormForQuestionAnswering",
            "RobertaPreLayerNormForSequenceClassification",
            "RobertaPreLayerNormForTokenClassification",
            "RobertaPreLayerNormModel",
            "RobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roc_bert"].extend(
        [
            "RoCBertForCausalLM",
            "RoCBertForMaskedLM",
            "RoCBertForMultipleChoice",
            "RoCBertForPreTraining",
            "RoCBertForQuestionAnswering",
            "RoCBertForSequenceClassification",
            "RoCBertForTokenClassification",
            "RoCBertModel",
            "RoCBertPreTrainedModel",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "RoFormerForCausalLM",
            "RoFormerForMaskedLM",
            "RoFormerForMultipleChoice",
            "RoFormerForQuestionAnswering",
            "RoFormerForSequenceClassification",
            "RoFormerForTokenClassification",
            "RoFormerModel",
            "RoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.rt_detr"].extend(
        [
            "RTDetrForObjectDetection",
            "RTDetrModel",
            "RTDetrPreTrainedModel",
            "RTDetrResNetBackbone",
            "RTDetrResNetPreTrainedModel",
        ]
    )
    _import_structure["models.rt_detr_v2"].extend(
        ["RTDetrV2ForObjectDetection", "RTDetrV2Model", "RTDetrV2PreTrainedModel"]
    )
    _import_structure["models.rwkv"].extend(
        [
            "RwkvForCausalLM",
            "RwkvModel",
            "RwkvPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "SamModel",
            "SamPreTrainedModel",
            "SamVisionModel",
        ]
    )
    _import_structure["models.seamless_m4t"].extend(
        [
            "SeamlessM4TCodeHifiGan",
            "SeamlessM4TForSpeechToSpeech",
            "SeamlessM4TForSpeechToText",
            "SeamlessM4TForTextToSpeech",
            "SeamlessM4TForTextToText",
            "SeamlessM4THifiGan",
            "SeamlessM4TModel",
            "SeamlessM4TPreTrainedModel",
            "SeamlessM4TTextToUnitForConditionalGeneration",
            "SeamlessM4TTextToUnitModel",
        ]
    )
    _import_structure["models.seamless_m4t_v2"].extend(
        [
            "SeamlessM4Tv2ForSpeechToSpeech",
            "SeamlessM4Tv2ForSpeechToText",
            "SeamlessM4Tv2ForTextToSpeech",
            "SeamlessM4Tv2ForTextToText",
            "SeamlessM4Tv2Model",
            "SeamlessM4Tv2PreTrainedModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "SegformerDecodeHead",
            "SegformerForImageClassification",
            "SegformerForSemanticSegmentation",
            "SegformerModel",
            "SegformerPreTrainedModel",
        ]
    )
    _import_structure["models.seggpt"].extend(
        [
            "SegGptForImageSegmentation",
            "SegGptModel",
            "SegGptPreTrainedModel",
        ]
    )
    _import_structure["models.sew"].extend(
        [
            "SEWForCTC",
            "SEWForSequenceClassification",
            "SEWModel",
            "SEWPreTrainedModel",
        ]
    )
    _import_structure["models.sew_d"].extend(
        [
            "SEWDForCTC",
            "SEWDForSequenceClassification",
            "SEWDModel",
            "SEWDPreTrainedModel",
        ]
    )
    _import_structure["models.shieldgemma2"].append("ShieldGemma2ForImageClassification")
    _import_structure["models.siglip"].extend(
        [
            "SiglipForImageClassification",
            "SiglipModel",
            "SiglipPreTrainedModel",
            "SiglipTextModel",
            "SiglipVisionModel",
        ]
    )
    _import_structure["models.siglip2"].extend(
        [
            "Siglip2ForImageClassification",
            "Siglip2Model",
            "Siglip2PreTrainedModel",
            "Siglip2TextModel",
            "Siglip2VisionModel",
        ]
    )
    _import_structure["models.smolvlm"].extend(
        [
            "SmolVLMForConditionalGeneration",
            "SmolVLMModel",
            "SmolVLMPreTrainedModel",
            "SmolVLMProcessor",
            "SmolVLMVisionConfig",
            "SmolVLMVisionTransformer",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].extend(["SpeechEncoderDecoderModel"])
    _import_structure["models.speech_to_text"].extend(
        [
            "Speech2TextForConditionalGeneration",
            "Speech2TextModel",
            "Speech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.speecht5"].extend(
        [
            "SpeechT5ForSpeechToSpeech",
            "SpeechT5ForSpeechToText",
            "SpeechT5ForTextToSpeech",
            "SpeechT5HifiGan",
            "SpeechT5Model",
            "SpeechT5PreTrainedModel",
        ]
    )
    _import_structure["models.splinter"].extend(
        [
            "SplinterForPreTraining",
            "SplinterForQuestionAnswering",
            "SplinterModel",
            "SplinterPreTrainedModel",
        ]
    )
    _import_structure["models.squeezebert"].extend(
        [
            "SqueezeBertForMaskedLM",
            "SqueezeBertForMultipleChoice",
            "SqueezeBertForQuestionAnswering",
            "SqueezeBertForSequenceClassification",
            "SqueezeBertForTokenClassification",
            "SqueezeBertModel",
            "SqueezeBertPreTrainedModel",
        ]
    )
    _import_structure["models.stablelm"].extend(
        [
            "StableLmForCausalLM",
            "StableLmForSequenceClassification",
            "StableLmForTokenClassification",
            "StableLmModel",
            "StableLmPreTrainedModel",
        ]
    )
    _import_structure["models.starcoder2"].extend(
        [
            "Starcoder2ForCausalLM",
            "Starcoder2ForSequenceClassification",
            "Starcoder2ForTokenClassification",
            "Starcoder2Model",
            "Starcoder2PreTrainedModel",
        ]
    )
    _import_structure["models.superglue"].extend(
        [
            "SuperGlueForKeypointMatching",
            "SuperGluePreTrainedModel",
        ]
    )
    _import_structure["models.superpoint"].extend(
        [
            "SuperPointForKeypointDetection",
            "SuperPointPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "SwiftFormerForImageClassification",
            "SwiftFormerModel",
            "SwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "SwinBackbone",
            "SwinForImageClassification",
            "SwinForMaskedImageModeling",
            "SwinModel",
            "SwinPreTrainedModel",
        ]
    )
    _import_structure["models.swin2sr"].extend(
        [
            "Swin2SRForImageSuperResolution",
            "Swin2SRModel",
            "Swin2SRPreTrainedModel",
        ]
    )
    _import_structure["models.swinv2"].extend(
        [
            "Swinv2Backbone",
            "Swinv2ForImageClassification",
            "Swinv2ForMaskedImageModeling",
            "Swinv2Model",
            "Swinv2PreTrainedModel",
        ]
    )
    _import_structure["models.switch_transformers"].extend(
        [
            "SwitchTransformersEncoderModel",
            "SwitchTransformersForConditionalGeneration",
            "SwitchTransformersModel",
            "SwitchTransformersPreTrainedModel",
            "SwitchTransformersSparseMLP",
            "SwitchTransformersTop1Router",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "T5EncoderModel",
            "T5ForConditionalGeneration",
            "T5ForQuestionAnswering",
            "T5ForSequenceClassification",
            "T5ForTokenClassification",
            "T5Model",
            "T5PreTrainedModel",
        ]
    )
    _import_structure["models.table_transformer"].extend(
        [
            "TableTransformerForObjectDetection",
            "TableTransformerModel",
            "TableTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TapasForMaskedLM",
            "TapasForQuestionAnswering",
            "TapasForSequenceClassification",
            "TapasModel",
            "TapasPreTrainedModel",
        ]
    )
    _import_structure["models.textnet"].extend(
        [
            "TextNetBackbone",
            "TextNetForImageClassification",
            "TextNetModel",
            "TextNetPreTrainedModel",
        ]
    )
    _import_structure["models.time_series_transformer"].extend(
        [
            "TimeSeriesTransformerForPrediction",
            "TimeSeriesTransformerModel",
            "TimeSeriesTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.timesformer"].extend(
        [
            "TimesformerForVideoClassification",
            "TimesformerModel",
            "TimesformerPreTrainedModel",
        ]
    )
    _import_structure["models.timm_backbone"].extend(["TimmBackbone"])
    _import_structure["models.timm_wrapper"].extend(
        ["TimmWrapperForImageClassification", "TimmWrapperModel", "TimmWrapperPreTrainedModel"]
    )
    _import_structure["models.trocr"].extend(
        [
            "TrOCRForCausalLM",
            "TrOCRPreTrainedModel",
        ]
    )
    _import_structure["models.tvp"].extend(
        [
            "TvpForVideoGrounding",
            "TvpModel",
            "TvpPreTrainedModel",
        ]
    )
    _import_structure["models.udop"].extend(
        [
            "UdopEncoderModel",
            "UdopForConditionalGeneration",
            "UdopModel",
            "UdopPreTrainedModel",
        ],
    )
    _import_structure["models.umt5"].extend(
        [
            "UMT5EncoderModel",
            "UMT5ForConditionalGeneration",
            "UMT5ForQuestionAnswering",
            "UMT5ForSequenceClassification",
            "UMT5ForTokenClassification",
            "UMT5Model",
            "UMT5PreTrainedModel",
        ]
    )
    _import_structure["models.unispeech"].extend(
        [
            "UniSpeechForCTC",
            "UniSpeechForPreTraining",
            "UniSpeechForSequenceClassification",
            "UniSpeechModel",
            "UniSpeechPreTrainedModel",
        ]
    )
    _import_structure["models.unispeech_sat"].extend(
        [
            "UniSpeechSatForAudioFrameClassification",
            "UniSpeechSatForCTC",
            "UniSpeechSatForPreTraining",
            "UniSpeechSatForSequenceClassification",
            "UniSpeechSatForXVector",
            "UniSpeechSatModel",
            "UniSpeechSatPreTrainedModel",
        ]
    )
    _import_structure["models.univnet"].extend(
        [
            "UnivNetModel",
        ]
    )
    _import_structure["models.upernet"].extend(
        [
            "UperNetForSemanticSegmentation",
            "UperNetPreTrainedModel",
        ]
    )
    _import_structure["models.video_llava"].extend(
        [
            "VideoLlavaForConditionalGeneration",
            "VideoLlavaPreTrainedModel",
            "VideoLlavaProcessor",
        ]
    )
    _import_structure["models.videomae"].extend(
        [
            "VideoMAEForPreTraining",
            "VideoMAEForVideoClassification",
            "VideoMAEModel",
            "VideoMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vilt"].extend(
        [
            "ViltForImageAndTextRetrieval",
            "ViltForImagesAndTextClassification",
            "ViltForMaskedLM",
            "ViltForQuestionAnswering",
            "ViltForTokenClassification",
            "ViltModel",
            "ViltPreTrainedModel",
        ]
    )
    _import_structure["models.vipllava"].extend(
        [
            "VipLlavaForConditionalGeneration",
            "VipLlavaPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["VisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["VisionTextDualEncoderModel"])
    _import_structure["models.visual_bert"].extend(
        [
            "VisualBertForMultipleChoice",
            "VisualBertForPreTraining",
            "VisualBertForQuestionAnswering",
            "VisualBertForRegionToPhraseAlignment",
            "VisualBertForVisualReasoning",
            "VisualBertModel",
            "VisualBertPreTrainedModel",
        ]
    )
    _import_structure["models.vit"].extend(
        [
            "ViTForImageClassification",
            "ViTForMaskedImageModeling",
            "ViTModel",
            "ViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "ViTMAEForPreTraining",
            "ViTMAEModel",
            "ViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vit_msn"].extend(
        [
            "ViTMSNForImageClassification",
            "ViTMSNModel",
            "ViTMSNPreTrainedModel",
        ]
    )
    _import_structure["models.vitdet"].extend(
        [
            "VitDetBackbone",
            "VitDetModel",
            "VitDetPreTrainedModel",
        ]
    )
    _import_structure["models.vitmatte"].extend(
        [
            "VitMatteForImageMatting",
            "VitMattePreTrainedModel",
        ]
    )
    _import_structure["models.vitpose"].extend(
        [
            "VitPoseForPoseEstimation",
            "VitPosePreTrainedModel",
        ]
    )
    _import_structure["models.vitpose_backbone"].extend(
        [
            "VitPoseBackbone",
            "VitPoseBackbonePreTrainedModel",
        ]
    )
    _import_structure["models.vits"].extend(
        [
            "VitsModel",
            "VitsPreTrainedModel",
        ]
    )
    _import_structure["models.vivit"].extend(
        [
            "VivitForVideoClassification",
            "VivitModel",
            "VivitPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "Wav2Vec2ForAudioFrameClassification",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForMaskedLM",
            "Wav2Vec2ForPreTraining",
            "Wav2Vec2ForSequenceClassification",
            "Wav2Vec2ForXVector",
            "Wav2Vec2Model",
            "Wav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_bert"].extend(
        [
            "Wav2Vec2BertForAudioFrameClassification",
            "Wav2Vec2BertForCTC",
            "Wav2Vec2BertForSequenceClassification",
            "Wav2Vec2BertForXVector",
            "Wav2Vec2BertModel",
            "Wav2Vec2BertPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_conformer"].extend(
        [
            "Wav2Vec2ConformerForAudioFrameClassification",
            "Wav2Vec2ConformerForCTC",
            "Wav2Vec2ConformerForPreTraining",
            "Wav2Vec2ConformerForSequenceClassification",
            "Wav2Vec2ConformerForXVector",
            "Wav2Vec2ConformerModel",
            "Wav2Vec2ConformerPreTrainedModel",
        ]
    )
    _import_structure["models.wavlm"].extend(
        [
            "WavLMForAudioFrameClassification",
            "WavLMForCTC",
            "WavLMForSequenceClassification",
            "WavLMForXVector",
            "WavLMModel",
            "WavLMPreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "WhisperForAudioClassification",
            "WhisperForCausalLM",
            "WhisperForConditionalGeneration",
            "WhisperModel",
            "WhisperPreTrainedModel",
        ]
    )
    _import_structure["models.x_clip"].extend(
        [
            "XCLIPModel",
            "XCLIPPreTrainedModel",
            "XCLIPTextModel",
            "XCLIPVisionModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "XGLMForCausalLM",
            "XGLMModel",
            "XGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "XLMForMultipleChoice",
            "XLMForQuestionAnswering",
            "XLMForQuestionAnsweringSimple",
            "XLMForSequenceClassification",
            "XLMForTokenClassification",
            "XLMModel",
            "XLMPreTrainedModel",
            "XLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "XLMRobertaForCausalLM",
            "XLMRobertaForMaskedLM",
            "XLMRobertaForMultipleChoice",
            "XLMRobertaForQuestionAnswering",
            "XLMRobertaForSequenceClassification",
            "XLMRobertaForTokenClassification",
            "XLMRobertaModel",
            "XLMRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta_xl"].extend(
        [
            "XLMRobertaXLForCausalLM",
            "XLMRobertaXLForMaskedLM",
            "XLMRobertaXLForMultipleChoice",
            "XLMRobertaXLForQuestionAnswering",
            "XLMRobertaXLForSequenceClassification",
            "XLMRobertaXLForTokenClassification",
            "XLMRobertaXLModel",
            "XLMRobertaXLPreTrainedModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "XLNetForMultipleChoice",
            "XLNetForQuestionAnswering",
            "XLNetForQuestionAnsweringSimple",
            "XLNetForSequenceClassification",
            "XLNetForTokenClassification",
            "XLNetLMHeadModel",
            "XLNetModel",
            "XLNetPreTrainedModel",
        ]
    )
    _import_structure["models.xmod"].extend(
        [
            "XmodForCausalLM",
            "XmodForMaskedLM",
            "XmodForMultipleChoice",
            "XmodForQuestionAnswering",
            "XmodForSequenceClassification",
            "XmodForTokenClassification",
            "XmodModel",
            "XmodPreTrainedModel",
        ]
    )
    _import_structure["models.yolos"].extend(
        [
            "YolosForObjectDetection",
            "YolosModel",
            "YolosPreTrainedModel",
        ]
    )
    _import_structure["models.yoso"].extend(
        [
            "YosoForMaskedLM",
            "YosoForMultipleChoice",
            "YosoForQuestionAnswering",
            "YosoForSequenceClassification",
            "YosoForTokenClassification",
            "YosoModel",
            "YosoPreTrainedModel",
        ]
    )
    _import_structure["models.zamba"].extend(
        [
            "ZambaForCausalLM",
            "ZambaForSequenceClassification",
            "ZambaModel",
            "ZambaPreTrainedModel",
        ]
    )
    _import_structure["models.zamba2"].extend(
        [
            "Zamba2ForCausalLM",
            "Zamba2ForSequenceClassification",
            "Zamba2Model",
            "Zamba2PreTrainedModel",
        ]
    )
    _import_structure["models.zoedepth"].extend(
        [
            "ZoeDepthForDepthEstimation",
            "ZoeDepthPreTrainedModel",
        ]
    )
    _import_structure["optimization"] = [
        "Adafactor",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_inverse_sqrt_schedule",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
        "get_wsd_schedule",
    ]
    _import_structure["pytorch_utils"] = [
        "Conv1D",
        "apply_chunking_to_forward",
        "prune_layer",
    ]
    _import_structure["sagemaker"] = []
    _import_structure["time_series_utils"] = []

try:
    if not (
        is_librosa_available()
        and is_essentia_available()
        and is_scipy_available()
        and is_torch_available()
        and is_pretty_midi_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import (
        dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects,
    )

    _import_structure["utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects"] = [
        name
        for name in dir(dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects)
        if not name.startswith("_")
    ]
else:
    _import_structure["models.pop2piano"].append("Pop2PianoFeatureExtractor")
    _import_structure["models.pop2piano"].append("Pop2PianoTokenizer")
    _import_structure["models.pop2piano"].append("Pop2PianoProcessor")

try:
    if not is_torchaudio_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.utils import (
        dummy_torchaudio_objects,
    )

    _import_structure["utils.dummy_torchaudio_objects"] = [
        name for name in dir(dummy_torchaudio_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyFeatureExtractor")
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyProcessor")


from . import ms_utils
from .masking_utils import create_causal_mask, create_sliding_window_causal_mask
from .modeling_utils import construct_pipeline_parallel_model, _load_pretrained_model_wrapper, \
    _get_resolved_checkpoint_files_wrapper
from .tokenization_utils import apply_chat_template_wrapper
from .trainer import training_step
from .generation import *

# redirect mindnlp.transformers to transformers
import transformers
sys.modules[__name__] = _LazyModule(
    'transformers',
    transformers.__file__,
    _import_structure,
    module_spec=__spec__,
    extra_objects={"__version__": transformers.__version__},
)


# patch transformers
def not_supported():
    return False

def empty_fn(*args, **kwargs):
    pass

transformers.utils.import_utils._torch_fx_available = False
transformers.utils.import_utils.is_torch_sdpa_available = not_supported


from ..utils.decorators import dtype_wrapper, patch_dtype_wrapper, patch_wrappers

patch_dtype_wrapper(transformers.AutoModel, 'from_pretrained')
patch_dtype_wrapper(transformers.modeling_utils.PreTrainedModel, 'from_pretrained',
                    [transformers.modeling_utils.restore_default_torch_dtype]
                    )
patch_wrappers(transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model',
                [_load_pretrained_model_wrapper])

transformers.modeling_utils._get_resolved_checkpoint_files = _get_resolved_checkpoint_files_wrapper(
    transformers.modeling_utils._get_resolved_checkpoint_files
)

transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template = apply_chat_template_wrapper(
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template
)

transformers.pipelines.pipeline = dtype_wrapper(transformers.pipelines.pipeline)
transformers.modeling_utils.caching_allocator_warmup = empty_fn
transformers.masking_utils.create_causal_mask = create_causal_mask
transformers.masking_utils.create_sliding_window_causal_mask = create_sliding_window_causal_mask

transformers.trainer.Trainer.training_step = training_step
# for ORANGE_PI
if ON_ORANGE_PI:
    transformers.generation.logits_process.InfNanRemoveLogitsProcessor.__call__ = InfNanRemoveLogitsProcessor_call

# add mindnlp.transformers modules/attrs to lazymodule
# setattr(sys.modules[__name__], 'test_ms_model', test_ms_model)
