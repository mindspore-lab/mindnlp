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

"""MindNLP Openai-GPT-1 model"""
import mindspore
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore import nn 
from mindspore import ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, Normal, Zero
import math


# mindspore版本gelu函数
def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + ops.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * ops.pow(x, 3))))


# mindspore激活函数
ACT_FNS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": gelu_new,
    "swish": nn.SiLU
    }


# mindspore实现线性层
class Conv1D(nn.Cell):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features. 输出特征数量
        nx (:obj:`int`): The number of input features. 输入特征数量
    """
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = initializer(Normal(sigma=0.02, mean=0.0), shape=[nx, nf], dtype=mindspore.float32)
        self.weight = Parameter(w)
        self.bias = Parameter(Tensor(shape=(nf), dtype=mindspore.float32, init=Zero()), name='bias')
        
    def construct(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x = ops.addmm(self.bias, x.view(-1, x.shape[-1]), self.weight)
        x = x.view(*size_out)
        return x
    


#mindspore版本实现MLP
class MLP(nn.Cell):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd # 768,输入层
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(keep_prob = 1- config.resid_pdrop)

    def construct(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
    

# mindspore版本实现attention
class Attention(nn.Cell):
    def __init__(self, nx, n_positions, config, scale=False):
        super().__init__()
        n_state = nx 
        if n_state % config.n_head != 0:
            raise ValueError(f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}")
        
        # 不参与反向传播的网络参数
        self.bias = ops.Tril(ops.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions)
        
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(1-config.attn_pdrop)
        self.resid_dropout = nn.Dropout(1-config.resid_pdrop)
        self.pruned_heads = set()


    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = ops.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.shape[-1])

        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : w.shape[-2], : w.shape[-1]]
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = ops.softmax(w,dim=-1)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [ops.matmul(w,v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.transpose(0, 2, 1, 3)
        new_x_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.view(*new_x_shape) 

    def split_heads(self, x, k=False):
        new_x_shape = x.shape[:-1] + (self.n_head, x.shape[-1] // self.n_head)
        x = x.view(*new_x_shape) 
        if k:
            return x.transpose(0, 2, 3, 1)
        else:
            return x.transpose(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)

# mindspore实现transformer模块
class Block(nn.Module):
    def __init__(self, n_positions, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_positions, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        attn_outputs = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        a = attn_outputs[0]

        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = [h] + attn_outputs[1:]
        return outputs
    

# OpenAIGPTModel模型实现
class OpenAIGPTModel(nn.Cell):
    """
    "The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top."
    """
    def __init__(self, config):
        super().__init__(config)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.CellList([Block(config.n_positions, config, scale=True) for _ in range(config.n_layer)])
        self.position_ids = ops.arange(config.n_positions)

    def construct(
        self,
        input_ids,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            # Code is different from when we had a single embedding matrix  from position and token embeddings
            position_ids = self.position_ids[None, : input_shape[-1]]

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            # attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            attention_mask = (1.0 - attention_mask) * Tensor(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min, self.dtype)

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(hidden_states, attention_mask, head_mask[i], output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = hidden_states.view(*output_shape)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions


class OpenAIGPTLMHeadModel(nn.Cell):
    """
    OpenAI GPT Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings).
    """
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, bias=False)


    def construct(
        self,
        input_ids,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        return loss,lm_logits,transformer_outputs.hidden_states, transformer_outputs.attentions



"""
OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks.
The two heads are two linear layers. 
The language modeling head has its weights tied to the input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).
"""
# class OpenAIGPTDoubleHeadsModel(nn.Cell):
#     def __init__(self, config):
#         super().__init__(config)

#         config.num_labels = 1
#         self.transformer = OpenAIGPTModel(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
#         # self.multiple_choice_head = SequenceSummary(config)

#     def construct(
#         self,
#         input_ids,
#         attention_mask = None,
#         token_type_ids = None,
#         position_ids = None,
#         head_mask = None,
#         inputs_embeds = None,
#         mc_token_ids = None,
#         labels = None,
#         mc_labels = None,
#         output_attentions = None,
#         output_hidden_states = None
#     ):
#         transformer_outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states
#         )
#         hidden_states = transformer_outputs[0]

#         lm_logits = self.lm_head(hidden_states)
#         mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

#         lm_loss, mc_loss = None, None
#         if mc_labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
#         if labels is not None:
#             shift_logits = lm_logits[..., :-1, :]
#             shift_labels = labels[..., 1:]
#             loss_fct = nn.CrossEntropyLoss()
#             lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#         if not return_dict:
#             output = (lm_logits, mc_logits) + transformer_outputs[1:]
#             if mc_loss is not None:
#                 output = (mc_loss,) + output
#             return ((lm_loss,) + output) if lm_loss is not None else output

#         return OpenAIGPTDoubleHeadsModelOutput(
#             loss=lm_loss,
#             mc_loss=mc_loss,
#             logits=lm_logits,
#             mc_logits=mc_logits,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )

"""
    The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer).
    [`OpenAIGPTForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the
    last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding
    token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
    it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take
    the last value in each row of the batch).
"""
# class OpenAIGPTForSequenceClassification(nn.Cell):
    
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.transformer = OpenAIGPTModel(config)
#         self.score = nn.Dense(config.n_embd, self.num_labels, bias=False)

#     def construct(
#         self,
#         input_ids,
#         attention_mask = None,
#         token_type_ids = None,
#         position_ids = None,
#         head_mask = None,
#         inputs_embeds = None,
#         labels = None,
#         output_attentions = None,
#         output_hidden_states = None
#     ):
    
#         transformer_outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states
#         )

#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size, sequence_length = input_ids.shape[:2]
#         else:
#             batch_size, sequence_length = inputs_embeds.shape[:2]

#         # Ensure the batch size is > 1 if there is no padding.
#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

#         if self.config.pad_token_id is None:
#             sequence_lengths = -1
#         else:
#             if input_ids is not None:
#                 sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
#             else:
#                 sequence_lengths = -1
#                 logger.warning(
#                     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                     "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#                 )

#         pooled_logits = logits[range(batch_size), sequence_lengths]

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=pooled_logits,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )
