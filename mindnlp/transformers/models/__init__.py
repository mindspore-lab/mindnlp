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
    chatglm,
    chatglm2,
    chatglm3,
    chatglm4,
    chinese_clip,
    clap,
    clip,
    clipseg,
    clvp,
    codegen,
    cohere,
    conditional_detr,
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
    dbrx,
    deberta,
    deberta_v2,
    decision_transformer,
    deformable_detr,
    deepseek_v2,
    detr,
    deta,
    deit,
    depth_anything,
    dinov2,
    distilbert,
    donut,
    dpr,
    dpt,
    efficientnet,
    efficientformer,
    electra,
    encodec,
    esm,
    ernie,
    ernie_m,
    falcon,
    fastspeech2_conformer,
    flava,
    flaubert,
    florence2,
    focalnet,
    fnet,
    funnel,
    fsmt,
    gemma,
    gemma2,
    git,
    openai,
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
    instructblip,
    ibert,
    idefics,
    jamba,
    jetmoe,
    kosmos2,
    layoutlm,
    layoutlmv2,
    layoutlmv3,
    led,
    lilt,
    llama,
    llava,
    llava_next,
    longformer,
    luke,
    lxmert,
    mamba,
    marian,
    markuplm,
    m2m_100,
    mask2former,
    mbart,
    mbart50,
    mctct,
    megatron_bert,
    mgp_str,
    minicpm,
    mistral,
    mixtral,
    mobilebert,
    mobilenet_v1,
    mobilenet_v2,
    mobilevit,
    mpnet,
    mpt,
    mllama,
    mluke,
    mt5,
    musicgen,
    musicgen_melody,
    mvp,
    nezha,
    nllb,
    nllb_moe,
    nougat,
    nystromformer,
    olmo,
    oneformer,
    openelm,
    opt,
    owlvit,
    patchtst,
    pegasus,
    pegasus_x,
    perceiver,
    persimmon,
    phi,
    phi3,
    pix2struct,
    plbart,
    poolformer,
    pop2piano,
    prophetnet,
    qdqbert,
    qwen2,
    qwen2_moe,
    qwen2_vl,
    rag,
    realm,
    reformer,
    rembert,
    resnet,
    roberta,
    roberta_prelayernorm,
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
    superpoint,
    swiftformer,
    swin,
    switch_transformers,
    swin2sr,
    t5,
    tapas,
    tapex,
    time_series_transformer,
    timesformer,
    tinybert,
    trocr,
    tvlt,
    udop,
    upernet,
    umt5,
    unispeech_sat,
    univnet,
    videomae,
    vipllava,
    vision_encoder_decoder,
    vision_text_dual_encoder,
    visual_bert,
    vit,
    vit_hybrid,
    vit_mae,
    vit_msn,
    vitdet,
    vitmatte,
    vits,
    vivit,
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
    xlm_prophetnet,
    xlnet,
    xmod,
    vilt,
    yolos,
    fuyu,
)
from .fuyu import *
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
from .chatglm import *
from .chatglm2 import *
from .chatglm3 import *
from .chatglm4 import *
from .chinese_clip import *
from .clap import *
from .clip import *
from .clipseg import *
from .clvp import *
from .codegen import *
from .cohere import *
from .conditional_detr import *
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
from .dbrx import *
from .deberta import *
from .deberta_v2 import *
from .decision_transformer import *
from .deformable_detr import *
from .deepseek_v2 import *
from .depth_anything import *
from .detr import *
from .deta import *
from .deit import *
from .dinov2 import *
from .distilbert import *
from .donut import *
from .dpr import *
from .dpt import *
from .efficientnet import *
from .efficientformer import *
from .electra import *
from .encodec import *
from .esm import *
from .ernie import *
from .ernie_m import *
from .falcon import *
from .flava import *
from .flaubert import *
from .florence2 import *
from .focalnet import *
from .fnet import *
from .funnel import *
from .fsmt import *
from .fastspeech2_conformer import *
from .gemma import *
from .gemma2 import *
from .git import *
from .openai import *
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
from .idefics import *
from .hubert import *
from .imagegpt import *
from .instructblip import *
from .jamba import *
from .jetmoe import *
from .kosmos2 import *
from .layoutlm import *
from .layoutlmv2 import *
from .layoutlmv3 import *
from .led import *
from .lilt import *
from .llama import *
from .llava import *
from .llava_next import *
from .longformer import *
from .luke import *
from .lxmert import *
from .m2m_100 import *
from .mamba import *
from .marian import *
from .markuplm import *
from .mask2former import *
from .mbart import *
from .mbart50 import *
from .mctct import *
from .megatron_bert import *
from .mgp_str import *
from .minicpm import *
from .mistral import *
from .mixtral import *
from .mobilebert import *
from .mobilenet_v1 import *
from .mobilenet_v2 import *
from .mobilevit import *
from .mpnet import *
from .mllama import *
from .mluke import *
from .mpt import *
from .mt5 import *
from .musicgen import *
from .musicgen_melody import *
from .mvp import *
from .nezha import *
from .nllb import *
from .nllb_moe import *
from .nougat import *
from .nystromformer import *
from .olmo import *
from .oneformer import *
from .openelm import *
from .opt import *
from .owlvit import *
from .patchtst import *
from .pegasus import *
from .pegasus_x import *
from .perceiver import *
from .persimmon import *
from .phi import *
from .phi3 import *
from .pix2struct import *
from .plbart import *
from .poolformer import *
from .pop2piano import *
from .prophetnet import *
from .qdqbert import *
from .qwen2 import *
from .qwen2_moe import *
from .qwen2_vl import *
from .rag import *
from .realm import *
from .reformer import *
from .rembert import *
from .resnet import *
from .roberta import *
from .roberta_prelayernorm import *
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
from .superpoint import *
from .swiftformer import *
from .swin import *
from .switch_transformers import *
from .swin2sr import *
from .tinybert import *
from .t5 import *
from .tapas import *
from .tapex import *
from .time_series_transformer import *
from .timesformer import *
from .trocr import *
from .tvlt import *
from .udop import *
from .upernet import *
from .unispeech_sat import *
from .univnet import *
from .videomae import *
from .vilt import *
from .vipllava import *
from .vision_encoder_decoder import *
from .vision_text_dual_encoder import *
from .visual_bert import *
from .vit import *
from .vits import *
from .vit_hybrid import *
from .vit_mae import *
from .vit_msn import *
from .vitdet import *
from .vitmatte import *
from .vivit import *
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
from .xlm_prophetnet import *
from .xlnet import *
from .umt5 import *
from .xmod import *
from .yolos import *



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
__all__.extend(chatglm.__all__)
__all__.extend(chatglm2.__all__)
__all__.extend(chatglm3.__all__)
__all__.extend(chatglm4.__all__)
__all__.extend(chinese_clip.__all__)
__all__.extend(clap.__all__)
__all__.extend(clip.__all__)
__all__.extend(clipseg.__all__)
__all__.extend(clvp.__all__)
__all__.extend(codegen.__all__)
__all__.extend(cohere.__all__)
__all__.extend(conditional_detr.__all__)
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
__all__.extend(dbrx.__all__)
__all__.extend(deberta.__all__)
__all__.extend(deberta_v2.__all__)
__all__.extend(decision_transformer.__all__)
__all__.extend(deformable_detr.__all__)
__all__.extend(deepseek_v2.__all__)
__all__.extend(deit.__all__)
__all__.extend(depth_anything.__all__)
__all__.extend(dinov2.__all__)
__all__.extend(distilbert.__all__)
__all__.extend(donut.__all__)
__all__.extend(detr.__all__)
__all__.extend(deta.__all__)
__all__.extend(dpr.__all__)
__all__.extend(dpt.__all__)
__all__.extend(efficientnet.__all__)
__all__.extend(efficientformer.__all__)
__all__.extend(electra.__all__)
__all__.extend(encodec.__all__)
__all__.extend(ernie.__all__)
__all__.extend(ernie_m.__all__)
__all__.extend(esm.__all__)
__all__.extend(falcon.__all__)
__all__.extend(flava.__all__)
__all__.extend(flaubert.__all__)
__all__.extend(florence2.__all__)
__all__.extend(fnet.__all__)
__all__.extend(focalnet.__all__)
__all__.extend(funnel.__all__)
__all__.extend(fsmt.__all__)
__all__.extend(fastspeech2_conformer.__all__)
__all__.extend(openai.__all__)
__all__.extend(gptj.__all__)
__all__.extend(gemma.__all__)
__all__.extend(gemma2.__all__)
__all__.extend(git.__all__)
__all__.extend(gpt_neo.__all__)
__all__.extend(gpt_neox.__all__)
__all__.extend(gpt_neox_japanese.__all__)
__all__.extend(gpt_pangu.__all__)
__all__.extend(gpt_bigcode.__all__)
__all__.extend(gpt2.__all__)
__all__.extend(graphormer.__all__)
__all__.extend(groupvit.__all__)
__all__.extend(hubert.__all__)
__all__.extend(ibert.__all__)
__all__.extend(idefics.__all__)
__all__.extend(imagegpt.__all__)
__all__.extend(instructblip.__all__)
__all__.extend(jamba.__all__)
__all__.extend(jetmoe.__all__)
__all__.extend(kosmos2.__all__)
__all__.extend(layoutlm.__all__)
__all__.extend(layoutlmv2.__all__)
__all__.extend(layoutlmv3.__all__)
__all__.extend(led.__all__)
__all__.extend(lilt.__all__)
__all__.extend(llama.__all__)
__all__.extend(llava.__all__)
__all__.extend(llava_next.__all__)
__all__.extend(longformer.__all__)
__all__.extend(luke.__all__)
__all__.extend(lxmert.__all__)
__all__.extend(m2m_100.__all__)
__all__.extend(mamba.__all__)
__all__.extend(marian.__all__)
__all__.extend(markuplm.__all__)
__all__.extend(mask2former.__all__)
__all__.extend(mbart.__all__)
__all__.extend(mbart50.__all__)
__all__.extend(mctct.__all__)
__all__.extend(megatron_bert.__all__)
__all__.extend(mgp_str.__all__)
__all__.extend(minicpm.__all__)
__all__.extend(mistral.__all__)
__all__.extend(mixtral.__all__)
__all__.extend(mllama.__all__)
__all__.extend(mluke.__all__)
__all__.extend(mobilebert.__all__)
__all__.extend(mobilenet_v1.__all__)
__all__.extend(mobilenet_v2.__all__)
__all__.extend(mobilevit.__all__)
__all__.extend(mpnet.__all__)
__all__.extend(mpt.__all__)
__all__.extend(mt5.__all__)
__all__.extend(musicgen.__all__)
__all__.extend(musicgen_melody.__all__)
__all__.extend(mvp.__all__)
__all__.extend(nezha.__all__)
__all__.extend(nllb.__all__)
__all__.extend(nllb_moe.__all__)
__all__.extend(nougat.__all__)
__all__.extend(nystromformer.__all__)
__all__.extend(olmo.__all__)
__all__.extend(oneformer.__all__)
__all__.extend(openelm.__all__)
__all__.extend(opt.__all__)
__all__.extend(owlvit.__all__)
__all__.extend(patchtst.__all__)
__all__.extend(pegasus.__all__)
__all__.extend(pegasus_x.__all__)
__all__.extend(perceiver.__all__)
__all__.extend(persimmon.__all__)
__all__.extend(phi.__all__)
__all__.extend(phi3.__all__)
__all__.extend(pix2struct.__all__)
__all__.extend(plbart.__all__)
__all__.extend(poolformer.__all__)
__all__.extend(pop2piano.__all__)
__all__.extend(prophetnet.__all__)
__all__.extend(qdqbert.__all__)
__all__.extend(qwen2.__all__)
__all__.extend(qwen2_moe.__all__)
__all__.extend(qwen2_vl.__all__)
__all__.extend(rag.__all__)
__all__.extend(realm.__all__)
__all__.extend(reformer.__all__)
__all__.extend(rembert.__all__)
__all__.extend(resnet.__all__)
__all__.extend(roberta.__all__)
__all__.extend(roberta_prelayernorm.__all__)
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
__all__.extend(superpoint.__all__)
__all__.extend(t5.__all__)
__all__.extend(tapas.__all__)
__all__.extend(tapex.__all__)
__all__.extend(time_series_transformer.__all__)
__all__.extend(timesformer.__all__)
__all__.extend(tinybert.__all__)
__all__.extend(trocr.__all__)
__all__.extend(tvlt.__all__)
__all__.extend(udop.__all__)
__all__.extend(upernet.__all__)
__all__.extend(unispeech_sat.__all__)
__all__.extend(univnet.__all__)
__all__.extend(videomae.__all__)
__all__.extend(vilt.__all__)
__all__.extend(vipllava.__all__)
__all__.extend(vision_encoder_decoder.__all__)
__all__.extend(vision_text_dual_encoder.__all__)
__all__.extend(visual_bert.__all__)
__all__.extend(vit.__all__)
__all__.extend(vits.__all__)
__all__.extend(vit_hybrid.__all__)
__all__.extend(vit_mae.__all__)
__all__.extend(vit_msn.__all__)
__all__.extend(vitdet.__all__)
__all__.extend(vitmatte.__all__)
__all__.extend(vivit.__all__)
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
__all__.extend(xlm_prophetnet.__all__)
__all__.extend(xlnet.__all__)
__all__.extend(umt5.__all__)
__all__.extend(xmod.__all__)
__all__.extend(fuyu.__all__)
__all__.extend(yolos.__all__)
