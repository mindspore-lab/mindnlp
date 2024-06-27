# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Models init
"""

from . import (
    albert,
    align,
    altclip,
    audio_spectrogram_transformer,
    auto,
    owlv2,
    autoformer,
    baichuan,
    bark,
    bart,
    barthez,
    bartpho,
    beit,
    bert,
    bert_generation,
    bert_japanese,
    bertweet,
    bge_m3,
    big_bird,
    bigbird_pegasus,
    biogpt,
    bit,
    blenderbot,
    blenderbot_small,
    blip,
    blip_2,
    bloom,
    bridgetower,
    bros,
    byt5,
    camembert,
    canine,
    clap,
    clip,
    codegen,
    cohere,
    cogvlm,
    convbert,
    convnext,
    convnextv2,
    cpm,
    cpmant,
    ctrl,
    cpmbee,
    cvt,
    data2vec,
    deberta,
    deberta_v2,
    decision_transformer,
    detr,
    deit,
    distilbert,
    dpr,
    efficientnet,
    efficientformer,
    electra,
    encodec,
    esm,
    ernie,
    ernie_m,
    falcon,
    fastspeech2_conformer,
    chatglm,
    chatglm2,
    chatglm3,
    flava,
    focalnet,
    funnel,
    gemma,
    git,
    gpt,
    gpt2,
    gpt_bigcode,
    gptj,
    gpt_neo,
    gpt_neox,
    gpt_neox_japanese,
    gpt_pangu,
    graphormer,
    groupvit,
    hubert,
    imagegpt,
    ibert,
    jetmoe,
    layoutlm,
    layoutlmv2,
    led,
    lilt,
    llama,
    llava,
    llava_next,
    longformer,
    luke,
    mamba,
    marian,
    mask2former,
    mbart,
    mctct,
    megatron_bert,
    minicpm,
    mistral,
    mixtral,
    mobilebert,
    mobilenet_v1,
    mobilenet_v2,
    mobilevit,
    mpnet,
    mpt,
    musicgen,
    musicgen_melody,
    mvp,
    nezha,
    nystromformer,
    olmo,
    oneformer,
    openelm,
    opt,
    owlvit,
    pegasus,
    phi,
    phi3,
    poolformer,
    pop2piano,
    qdqbert,
    qwen2,
    qwen2_moe,
    reformer,
    rembert,
    resnet,
    roberta,
    roc_bert,
    rwkv,
    sam,
    seamless_m4t,
    seamless_m4t_v2,
    segformer,
    seggpt,
    sew,
    sew_d,
    speech_encoder_decoder,
    speecht5,
    stablelm,
    splinter,
    squeezebert,
    starcoder2,
    swiftformer,
    swin,
    switch_transformers,
    swin2sr,
    t5,
    tapas,
    time_series_transformer,
    timesformer,
    tinybert,
    videomae,
    vipllava,
    vision_encoder_decoder,
    vision_text_dual_encoder,
    visual_bert,
    vit,
    vit_hybrid,
    vit_msn,
    vitdet,
    wav2vec2,
    wav2vec2_conformer,
    wav2vec2_bert,
    wav2vec2_with_lm,
    wavlm,
    whisper,
    x_clip,
    xlm,
    xlm_roberta,
    xlm_roberta_xl,
    xlnet,
)

from .owlv2 import *
from .albert import *
from .align import *
from .altclip import *
from .audio_spectrogram_transformer import *
from .auto import *
from .autoformer import *
from .baichuan import *
from .bark import *
from .bart import *
from .barthez import *
from .bartpho import *
from .beit import *
from .bert import *
from .bert_generation import *
from .bert_japanese import *
from .bertweet import *
from .bge_m3 import *
from .big_bird import *
from .bigbird_pegasus import *
from .biogpt import *
from .bit import *
from .blenderbot import *
from .blenderbot_small import *
from .blip import *
from .blip_2 import *
from .bloom import *
from .bridgetower import *
from .bros import *
from .byt5 import *
from .camembert import *
from .canine import *
from .clap import *
from .clip import *
from .codegen import *
from .cohere import *
from .cogvlm import *
from .convbert import *
from .convnext import *
from .convnextv2 import *
from .cpm import *
from .ctrl import *
from .cpmant import *
from .cpmbee import *
from .cvt import *
from .data2vec import *
from .deberta import *
from .deberta_v2 import *
from .decision_transformer import *
from .detr import *
from .deit import *
from .distilbert import *
from .dpr import *
from .efficientnet import *
from .efficientformer import *
from .electra import *
from .encodec import *
from .esm import *
from .ernie import *
from .ernie_m import *
from .chatglm import *
from .chatglm2 import *
from .chatglm3 import *
from .falcon import *
from .flava import *
from .focalnet import *
from .funnel import *
from .fastspeech2_conformer import *
from .gemma import *
from .git import *
from .gpt import *
from .gptj import *
from .gpt_neo import *
from .gpt_neox import *
from .gpt_neox_japanese import *
from .gpt_bigcode import *
from .gpt_pangu import *
from .gpt2 import *
from .graphormer import *
from .groupvit import *
from .ibert import *
from .hubert import *
from .imagegpt import *
from .jetmoe import *
from .layoutlm import *
from .layoutlmv2 import *
from .led import *
from .lilt import *
from .llama import *
from .llava import *
from .llava_next import *
from .longformer import *
from .luke import *
from .mamba import *
from .marian import *
from .mask2former import *
from .mbart import *
from .mctct import *
from .megatron_bert import *
from .minicpm import *
from .mistral import *
from .mixtral import *
from .mobilebert import *
from .mobilenet_v1 import *
from .mobilenet_v2 import *
from .mobilevit import *
from .mpnet import *
from .mpt import *
from .musicgen import *
from .musicgen_melody import *
from .mvp import *
from .nezha import *
from .nystromformer import *
from .olmo import *
from .oneformer import *
from .openelm import *
from .opt import *
from .owlvit import *
from .pegasus import *
from .phi import *
from .phi3 import *
from .poolformer import *
from .pop2piano import *
from .qdqbert import *
from .qwen2 import *
from .qwen2_moe import *
from .reformer import *
from .rembert import *
from .resnet import *
from .roberta import *
from .roc_bert import *
from .rwkv import *
from .sam import *
from .seamless_m4t import *
from .seamless_m4t_v2 import *
from .segformer import *
from .seggpt import *
from .sew import *
from .sew_d import *
from .speech_encoder_decoder import *
from .speecht5 import *
from .stablelm import *
from .splinter import *
from .squeezebert import *
from .starcoder2 import *
from .swiftformer import *
from .swin import *
from .switch_transformers import *
from .swin2sr import *
from .tinybert import *
from .t5 import *
from .tapas import *
from .time_series_transformer import *
from .timesformer import *
from .videomae import *
from .vipllava import *
from .vision_encoder_decoder import *
from .vision_text_dual_encoder import *
from .visual_bert import *
from .vit import *
from .vit_hybrid import *
from .vit_msn import *
from .vitdet import *
from .whisper import *
from .wav2vec2 import *
from .wav2vec2_conformer import *
from .wav2vec2_bert import *
from .wav2vec2_with_lm import *
from .wavlm import *
from .x_clip import *
from .xlm import *
from .xlm_roberta import *
from .xlm_roberta_xl import *
from .xlnet import *

__all__ = []
__all__.extend(albert.__all__)
__all__.extend(align.__all__)
__all__.extend(altclip.__all__)
__all__.extend(audio_spectrogram_transformer.__all__)
__all__.extend(auto.__all__)
__all__.extend(autoformer.__all__)
__all__.extend(baichuan.__all__)
__all__.extend(bark.__all__)
__all__.extend(bart.__all__)
__all__.extend(barthez.__all__)
__all__.extend(bartpho.__all__)
__all__.extend(beit.__all__)
__all__.extend(bert.__all__)
__all__.extend(bert_generation.__all__)
__all__.extend(bert_japanese.__all__)
__all__.extend(bertweet.__all__)
__all__.extend(bge_m3.__all__)
__all__.extend(big_bird.__all__)
__all__.extend(bigbird_pegasus.__all__)
__all__.extend(biogpt.__all__)
__all__.extend(bit.__all__)
__all__.extend(blenderbot.__all__)
__all__.extend(blenderbot_small.__all__)
__all__.extend(blip.__all__)
__all__.extend(blip_2.__all__)
__all__.extend(bloom.__all__)
__all__.extend(bridgetower.__all__)
__all__.extend(bros.__all__)
__all__.extend(byt5.__all__)
__all__.extend(camembert.__all__)
__all__.extend(canine.__all__)
__all__.extend(clap.__all__)
__all__.extend(clip.__all__)
__all__.extend(codegen.__all__)
__all__.extend(cohere.__all__)
__all__.extend(cogvlm.__all__)
__all__.extend(convbert.__all__)
__all__.extend(convnext.__all__)
__all__.extend(convnextv2.__all__)
__all__.extend(cpm.__all__)
__all__.extend(ctrl.__all__)
__all__.extend(cpmant.__all__)
__all__.extend(cpmbee.__all__)
__all__.extend(cvt.__all__)
__all__.extend(data2vec.__all__)
__all__.extend(deberta.__all__)
__all__.extend(deberta_v2.__all__)
__all__.extend(decision_transformer.__all__)
__all__.extend(deit.__all__)
__all__.extend(distilbert.__all__)
__all__.extend(detr.__all__)
__all__.extend(dpr.__all__)
__all__.extend(efficientnet.__all__)
__all__.extend(efficientformer.__all__)
__all__.extend(electra.__all__)
__all__.extend(encodec.__all__)
__all__.extend(ernie.__all__)
__all__.extend(ernie_m.__all__)
__all__.extend(esm.__all__)
__all__.extend(chatglm.__all__)
__all__.extend(falcon.__all__)
__all__.extend(flava.__all__)
__all__.extend(focalnet.__all__)
__all__.extend(funnel.__all__)
__all__.extend(fastspeech2_conformer.__all__)
__all__.extend(chatglm2.__all__)
__all__.extend(chatglm3.__all__)
__all__.extend(gpt.__all__)
__all__.extend(gptj.__all__)
__all__.extend(gemma.__all__)
__all__.extend(git.__all__)
__all__.extend(gpt_neo.__all__)
__all__.extend(gpt_neox.__all__)
__all__.extend(gpt_neox_japanese.__all__)
__all__.extend(gpt_pangu.__all__)
__all__.extend(gpt_bigcode.__all__)
__all__.extend(gpt2.__all__)
__all__.extend(graphormer.__all__)
__all__.extend(groupvit.__all__)
__all__.extend(ibert.__all__)
__all__.extend(hubert.__all__)
__all__.extend(imagegpt.__all__)
__all__.extend(jetmoe.__all__)
__all__.extend(layoutlm.__all__)
__all__.extend(layoutlmv2.__all__)
__all__.extend(led.__all__)
__all__.extend(lilt.__all__)
__all__.extend(llama.__all__)
__all__.extend(llava.__all__)
__all__.extend(llava_next.__all__)
__all__.extend(longformer.__all__)
__all__.extend(luke.__all__)
__all__.extend(mamba.__all__)
__all__.extend(marian.__all__)
__all__.extend(mask2former.__all__)
__all__.extend(mbart.__all__)
__all__.extend(mctct.__all__)
__all__.extend(megatron_bert.__all__)
__all__.extend(minicpm.__all__)
__all__.extend(mistral.__all__)
__all__.extend(mixtral.__all__)
__all__.extend(mobilebert.__all__)
__all__.extend(mobilenet_v1.__all__)
__all__.extend(mobilenet_v2.__all__)
__all__.extend(mobilevit.__all__)
__all__.extend(mpnet.__all__)
__all__.extend(mpt.__all__)
__all__.extend(musicgen.__all__)
__all__.extend(musicgen_melody.__all__)
__all__.extend(mvp.__all__)
__all__.extend(nezha.__all__)
__all__.extend(nystromformer.__all__)
__all__.extend(olmo.__all__)
__all__.extend(oneformer.__all__)
__all__.extend(openelm.__all__)
__all__.extend(opt.__all__)
__all__.extend(owlvit.__all__)
__all__.extend(pegasus.__all__)
__all__.extend(phi.__all__)
__all__.extend(phi3.__all__)
__all__.extend(poolformer.__all__)
__all__.extend(pop2piano.__all__)
__all__.extend(qdqbert.__all__)
__all__.extend(qwen2.__all__)
__all__.extend(qwen2_moe.__all__)
__all__.extend(reformer.__all__)
__all__.extend(rembert.__all__)
__all__.extend(resnet.__all__)
__all__.extend(roberta.__all__)
__all__.extend(roc_bert.__all__)
__all__.extend(rwkv.__all__)
__all__.extend(sam.__all__)
__all__.extend(seamless_m4t.__all__)
__all__.extend(seamless_m4t_v2.__all__)
__all__.extend(segformer.__all__)
__all__.extend(seggpt.__all__)
__all__.extend(sew.__all__)
__all__.extend(sew_d.__all__)
__all__.extend(speech_encoder_decoder.__all__)
__all__.extend(speecht5.__all__)
__all__.extend(stablelm.__all__)
__all__.extend(splinter.__all__)
__all__.extend(squeezebert.__all__)
__all__.extend(starcoder2.__all__)
__all__.extend(swiftformer.__all__)
__all__.extend(owlv2.__all__)
__all__.extend(swin.__all__)
__all__.extend(switch_transformers.__all__)
__all__.extend(swin2sr.__all__)
__all__.extend(t5.__all__)
__all__.extend(time_series_transformer.__all__)
__all__.extend(timesformer.__all__)
__all__.extend(tinybert.__all__)
__all__.extend(videomae.__all__)
__all__.extend(vipllava.__all__)
__all__.extend(vision_encoder_decoder.__all__)
__all__.extend(vision_text_dual_encoder.__all__)
__all__.extend(visual_bert.__all__)
__all__.extend(vit.__all__)
__all__.extend(vit_hybrid.__all__)
__all__.extend(vit_msn.__all__)
__all__.extend(vitdet.__all__)
__all__.extend(whisper.__all__)
__all__.extend(wav2vec2.__all__)
__all__.extend(wav2vec2_conformer.__all__)
__all__.extend(wav2vec2_bert.__all__)
__all__.extend(wav2vec2_with_lm.__all__)
__all__.extend(wavlm.__all__)
__all__.extend(x_clip.__all__)
__all__.extend(xlm.__all__)
__all__.extend(xlm_roberta.__all__)
__all__.extend(xlm_roberta_xl.__all__)
__all__.extend(xlnet.__all__)
