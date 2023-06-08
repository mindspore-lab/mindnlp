# Copyright 2022 Huawei Technologies Co., Ltd
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
# pylint: disable=W0237
# pylint: disable=E0401
# pylint: disable=R1714
"""
mindspore opt model
"""
import random
import mindspore
import numpy as np
from mindspore import nn
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
#from mindnlp.abc.backbones.pretrained import PreTrainedModel

from ..utils.activations import ACT2FN

from ...abc import PreTrainedModel
#from ..utils.mixin import CellUtilMixin
#from ..utils import logging
#from dataclasses import dataclass
#from collections import OrderedDict
#from dataclasses import fields
from .opt_config import OPTConfig

#logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

def _make_causal_mask(input_ids_shape, dtype: mindspore.dtype, past_key_values_length: int = 0):
    """
    Make casual mask for bi-directional self-attention
    """
    bsz, tgt_len = input_ids_shape
    #这里报错，可能是因为后面 input_shape = mindspore.ops.shape(input_ids) 返回值的问题
    #mask = mindspore.numpy.full((tgt_len,tgt_len),mindspore.tensor(mindspore.finfo(dtype).min))
    #mask = mindspore.numpy.full((tgt_len,tgt_len),mindspore.Tensor(np.finfo(dtype).min))
    mask = mindspore.numpy.full((tgt_len,tgt_len),mindspore.Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min))
    #尝试解决finfo问题
    #print("mask1 of ms is ", mask)
    maskshape = mindspore.ops.shape(mask)[-1]
    #mask_cond = mindspore.ops.arange(mask.shqpe(-1))
    #mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.shape(-1),1),0)
    #修改shape问题
    mask_cond = mindspore.ops.arange(maskshape)
    #print("cond of ms is ",mask_cond)
    #print("bool of ms is ", mask_cond < (mask_cond + 1).view(maskshape,1))#pass
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(maskshape,1),0)
    mask = mask.to(dtype)
    #print("mask2 of ms is ", mask)
    if past_key_values_length > 0:
        mask = mindspore.ops.Concat([mindspore.ops.Zeros((tgt_len, past_key_values_length), dtype),mask], dim=-1)
    multiples = (bsz, 1, 1, 1)
    return mindspore.ops.tile(mask[None, None, :, :], multiples)

def _expand_mask(mask: mindspore.Tensor, dtype: mindspore.dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    #bsz, src_len = mask.size()
    #print("here is mask")
    #print(mindspore.ops.shape(mask))
    #print(mask)
    #print(type(mask))#这里都能出错<class 'mindspore.common.tensor.Tensor'> <class 'mindspore.common._stub_tensor.StubTensor'>
    _, src_len = mindspore.ops.shape(mask)
    tgt_len = tgt_len if tgt_len is not None else src_len
    #print(tgt_len)
    #出错原因在于expand不支持在GPU上运行?
    #maskshape = np.array([bsz, 1, tgt_len, src_len])
    #maskshape = mindspore.Tensor(maskshape)
    #expanded_mask = mask[:, None, None, :].expand(maskshape).to(dtype)
    #must be a Tensor but got Tuple[Int64*4].
    #expanded_mask = mindspore.ops.expand(mindspore.Tensor(mask[:, None, None, :]),mindspore.Tensor(maskshape)).to(dtype)
    multiples = (1, 1, tgt_len,1)
    #print(multiples)
    mask = mindspore.Tensor(mask)
    #print(mindspore.ops.shape(mask[:, None, None, :]))
    #print("here is unsqueeze")
    mask = mindspore.ops.unsqueeze(mindspore.Tensor(mask),dim = 1)
    mask = mindspore.ops.unsqueeze(mindspore.Tensor(mask),dim = 1)
    #print(mindspore.ops.shape(mindspore.ops.unsqueeze(mindspore.Tensor(mask),dim = 1)))
    #expanded_mask = mindspore.Tensor(mask[:, None, None, :])
    #print(expanded_mask)
    expanded_mask = mindspore.ops.tile(mindspore.Tensor(mask),multiples)
    inverted_mask = 1.0 - expanded_mask
    mask_bool = inverted_mask.astype(mindspore.bool_)
    #print(mask_bool)
    return mindspore.ops.masked_fill(inverted_mask, mask_bool, mindspore.Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min))


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def construct(self, attention_mask, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        #attention_mask: LongTensor?
        attention_mask = attention_mask.to(mindspore.int64)# equivalent to torch.tensor.long()?

        # create positions depending on attention_mask
        temptype = attention_mask.dtype
        #positions = (mindspore.Tensor.cumsum(attention_mask, axis=1).type_as(attention_mask) * attention_mask).long() - 1
        positions = (mindspore.Tensor.cumsum(attention_mask, axis=1).astype(temptype) * attention_mask).long() - 1
        #problem 不确定astype,这里需要进一步检验
        #cumsum 即式 cumulative sum，累计一个和

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().construct(positions + self.offset)


class OPTAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.is_decoder = is_decoder

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)#QKV的投影都只是一个线性层
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        #return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return mindspore.ops.transpose(tensor.view(bsz, seq_len, self.num_heads, self.head_dim),(0,2,1,3))
    #problem
    #这里trasnpose的区别非常关键
    #相当于torch.permute算子
    #https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/migration_guide/typical_api_comparision.html?highlight=transpose
    #同时不管contiguous
    #batch表示有批量中的句子数量，seqlen表示句子中的单词数量，numheads表示头的数量，head_dim表示每个头下单词特征向量的维度
    #这里如果传入一个QKV矩阵，相当于直接通过view被拆分为了多头！！！
    def construct(
        self,
        hidden_states: mindspore.Tensor,
        key_value_states = None,
        past_key_value = None,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False,
    ):
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = mindspore.ops.shape(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = mindspore.ops.concat((past_key_value[0], key_states), axis=2)
            value_states = mindspore.ops.concat((past_key_value[1], value_states), axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)
            #这里解释了上面的past_key_value是个什么东西，相当于把之前的KV存下来?而past_key_value[0]，past_key_value[1]相当于直接使用之前的KV state
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)

        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = mindspore.ops.shape(key_states)[1]
        #attn_weights = mindspore.ops.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = mindspore.ops.bmm(query_states, mindspore.ops.transpose(key_states,(0,2,1)))
        #problem 检查这里是不是3个维度
        #bmm是指batch matrix-matrix product，三维张量b*n*m 与b*m*p,其实就是对每个batch中的 n*m m*p矩阵进行乘法运算

        if mindspore.ops.shape(attn_weights) != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {mindspore.ops.shape(attn_weights)}"
            )
        if attention_mask is not None:
            #if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            if mindspore.ops.shape(attention_mask) != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = mindspore.ops.maximum(attn_weights, mindspore.Tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min))
            #print("test max", mindspore.Tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min))
            #attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            #mask = mindspore.numpy.full((tgt_len,tgt_len),mindspore.Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == mindspore.float16:
            #attn_weights = mindspore.ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(mindspore.float16)
            attn_weights = mindspore.ops.softmax(attn_weights, axis=-1).to(mindspore.float16)
            #problem官方文档中没有dtype选项
        else:
            attn_weights = mindspore.ops.softmax(attn_weights, axis=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        #attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = mindspore.ops.dropout(attn_weights, p=self.dropout, training=self.training)
        #官方文档中没有training选项，实际使用并不需要train，怎么处理？

        attn_output = mindspore.ops.bmm(attn_probs, value_states)#先dropout然后再bmm,这里对完成QK后的结果进行dropout然后再乘上V

        if mindspore.ops.shape(attn_output) != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {mindspore.ops.shape(attn_output)}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        #attn_output = attn_output.transpose(1, 2)
        attn_output = mindspore.ops.transpose(attn_output,(0,2,1,3))

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)#最后还有个线性层？

        return attn_output, attn_weights_reshaped, past_key_value#除了计算完的output之外还有 shape与pask_key_value


class OPTDecoderLayer(nn.Cell):
    """OPTDecoderLayer"""
    def __init__(self, config: OPTConfig): #调用了config
        super().__init__()
        self.embed_dim = config.hidden_size #embed_dim使用config中的参数
        self.self_attn = OPTAttention(#调用上面的attention机制，参数基本上是config中设置好的
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before#关于layernorm的参数，bool变量决定是否进行?
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]#设置激活函数
        #pytorch 中 layernorm可以传入一个数 但是mindspore中只能传入一个tuple或者list
        """
        self.self_attn_layer_norm = nn.LayerNorm(#设置layernorm 直接调用LayerNorm 相比于pytorch中的nn.LayerNorm函数，没有elementwise_affine参数
            [self.embed_dim], begin_norm_axis = -1, begin_params_axis = -1
        )
        """
        self.self_attn_layer_norm = nn.LayerNorm(#设置layernorm，直接调用LayerNorm 相比于pytorch中的nn.LayerNorm函数，没有elementwise_affine参数
            [self.embed_dim], epsilon=1e-5
        )
        self.fc1 = nn.Dense(self.embed_dim, config.ffn_dim, has_bias=config.enable_bias)#两个线性层embed_dim->ffn_dim,然后再ffn_dim->embed_dim
        self.fc2 = nn.Dense(config.ffn_dim, self.embed_dim, has_bias=config.enable_bias)#没看太懂具体是要干什么
        self.final_layer_norm = nn.LayerNorm([self.embed_dim], epsilon=1e-5)#最后再来一层layernorm

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask = None,#区分重点
        layer_head_mask = None,#
        output_attentions = False,
        use_cache = False,
        past_key_value = None,
    ):
        """
        print("mindspore decoder layer")
        print("hidden state is ", hidden_states)
        print("attention mask is ", attention_mask)
        print("layer_head mask is :", layer_head_mask)
        print("output attention is ", output_attentions)
        print("use cache is ",use_cache)
        print("past key value is ", past_key_value)
        
        """
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        #print("hidden states of mindspore after layer norm")
        #print(hidden_states)
        # Self Attention
        #这里的selfattention就是上面的OPTAttention,mask也在这个过程中完成？
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,#传入attention mask
            layer_head_mask=layer_head_mask,#这里传入layer_head_mask
            output_attentions=output_attentions,
        )
        #print("hidden state of mindspore after attention!")
        #print(hidden_states)
        hidden_states = mindspore.ops.dropout(hidden_states, p=self.dropout, training=self.training)
        #print("hidden state of mindspore after dropout!")
        #print(hidden_states)
        #problem dropout 的training问题
        #hiddenstate进行完attention进行dropout然后加上residual
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)#先通过f1投影
        hidden_states = self.activation_fn(hidden_states)#激活函数?

        hidden_states = self.fc2(hidden_states)#通过f2投影回来
        hidden_states = mindspore.ops.dropout(hidden_states, p=self.dropout, training=self.training)#投影回来时候dropout

        hidden_states = (residual + hidden_states).view(hidden_states_shape)#residual代表上个状态的hidden_state 其实也就是飞线连接

        # 350m applies layer norm AFTER attention

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
class OPTPreTrainedModel(PreTrainedModel):
    """name changed PreTrainedModel->PretrainedModel,CHANGED BACK"""
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Dense):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OPTDecoder)):
            module.gradient_checkpointing = value

    def post_init(self):
        pass
    #self added


class OPTDecoder(OPTPreTrainedModel):#这里继承的是OPTPreTrainedModel?
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        #把word embedding 与 position设置好
        #self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, use_one_hot = False, padding_idx = 1)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim,padding_idx = self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            #self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
            self.project_out = nn.Dense(config.hidden_size, config.word_embed_proj_dim, has_bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            #self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
            self.project_in = nn.Dense(config.word_embed_proj_dim, config.hidden_size, has_bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        #注意mindspore与pytorch两者的差异
        #OPTconfig中 layer_norm_elementwise_affine=True
        #https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/note/api_mapping/pytorch_diff/LayerNorm.html?highlight=layernorm
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                #[config.hidden_size], begin_norm_axis = -1, begin_params_axis = -1
                #[config.hidden_size], begin_norm_axis = 0, begin_params_axis = 0
                [config.hidden_size], epsilon=1e-5
            )
        else:
            self.final_layer_norm = None

        #self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.CellList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        """
        self.layers在这一步就设置好了所有layer的信息.每一层是一个OPTDecoderLayer,
        通过for语句中的config.num_hidden_layers来迭代设置层数
        """
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        """
        mask 机制
        """
        #print("from prepare test direct outputs", attention_mask)
        #print("input shape is: ", input_shape, "input embeds is: ", inputs_embeds)#pass
        combined_attention_mask = None
        #print("inputshape is", input_shape[-1])
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )#.to(inputs_embeds.device)
        #print("combined mask of ms is: ", combined_attention_mask) #FAIL
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            #print("from prepare test type", type(attention_mask))
            #print("from prepare test direct output", attention_mask)
            #pass
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])#.to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def construct(
        self,
        input_ids = None,
        attention_mask = None,#attention mask 和headmask又是什么关系
        head_mask = None,#传入head_mask,而且这个headmask指定了每一层的mask,通过head_mask[idx]来传递每一层的mask
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """forward"""
        #problem
        #print("this is input_ids", input_ids)
        #print("this is attention", attention_mask)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is not None:
            #input_shape = input_ids.shape()
            input_shape = mindspore.ops.shape(input_ids)
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            #attention_mask = mindspore.ops.ones(inputs_embeds.shape[:2], dtype=mindspore.bool, device=inputs_embeds.device)
            #attention_mask = mindspore.ops.ones(inputs_embeds.shape[:2], dtype=mindspore.int64)
            attention_mask = mindspore.ops.ones(inputs_embeds.shape[:2], dtype=mindspore.bool_)
        #print("attention mask is",attention_mask)    #pass
        #problem1 None->StubTensor,是否应该通过dtype来修正
        #print("this is attention mask changed",attention_mask)
        #print(type(attention_mask))
        #print("attention_mask direct ouput",attention_mask)
        #这里可以直接输出，但是传入到expand_mask中就不能输出了
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
        #print("past key value of ms is: ", past_key_values) #pass both are none
        #调用_prepare_decoder_attention_mask，使用causal mask，然后再下面传入decoder_layer
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        #print("attention mask is",attention_mask)#FAIL 理论上应该是阶梯状那种 now pass
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds
        #print("hidden_states of mindspore is ",hidden_states)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                #logger.warning_once(
                #    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                #)
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            #print("hiddenstate before mslayer ", hidden_states) pass
            #print("attentionmask before mslayer ", attention_mask)#FAIL
            #print("past key values:", past_key_value)
            if self.gradient_checkpointing and self.training:
                #checkpoint
                print("Training = true???")

            else:
                layer_outputs = decoder_layer(#经过decoder层，其中每个decoder_layer是从self.layers中取出来的
                    hidden_states,
                    attention_mask=attention_mask,#将attention_mask传入!!!
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),#这里传入mask机制,与attention_mask有什么区别
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            print("hiddenstate after mindspore layer: ")
            print(hidden_states)
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        output = (hidden_states,)+(next_cache,)+\
            (all_hidden_states,)+(all_self_attns,)
        return output

class OPTModel(OPTPreTrainedModel):
    """opt model"""
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)#调用上面定义好的整个optdecoder
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        """decoder"""
        return self.decoder

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,#来传递每一层的mask
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """forward"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(#调用optdecoder
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        output = (decoder_outputs.last_hidden_state,)+(decoder_outputs.past_key_values,)+\
            (decoder_outputs.hidden_states,)+(decoder_outputs.attentions,)
        return output

class OPTForCausalLM(OPTPreTrainedModel):
    """task"""
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Dense(config.word_embed_proj_dim, config.vocab_size, has_bias=False)#线性层投影，从embedding的维度投影到vocabulary的维度，意思是要做最后的输出？

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """set decoder"""
        self.model.decoder = decoder

    def get_decoder(self):
        """use decoder"""
        return self.model.decoder

    #@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """forward"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        output = (loss,)+(logits,)+\
            (outputs.past_key_values,)+(outputs.hidden_states,)+(outputs.attentions,)
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

class OPTForSequenceClassification(OPTPreTrainedModel):
    """tasks"""
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Dense(config.word_embed_proj_dim, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """forward"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_lengths = input_ids.shape[:2]
        else:
            batch_size, sequence_lengths = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                #sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
                sequence_lengths = mindspore.ops.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[mindspore.ops.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == mindspore.int64 or labels.dtype == mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output


        output = (loss,) + (pooled_logits,) + (transformer_outputs.past_key_values,) + (transformer_outputs.hidden_states,)\
            + (transformer_outputs.attentions,)
        return output

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

class OPTForQuestionAnswering(OPTPreTrainedModel):
    """tasks"""
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Dense(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    #@add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    #@replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        start_positions   = None,
        end_positions = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """forward"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

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
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output


        output = (total_loss,) + (start_logits,) + (end_logits,) + (transformer_outputs.hidden_states,) + (transformer_outputs.attentions,)
        return output
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value
    