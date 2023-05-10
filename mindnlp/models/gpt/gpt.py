# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
# pylint: disable=C0103
# pylint: disable=C0415
# pylint: disable=W0223

"""MindNLP gpt model"""
import os
import logging
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal
from mindnlp.models.gpt.gpt_config import GPTConfig
from mindnlp._legacy.nn import Dropout, Matmul
from mindnlp._legacy.functional import split, softmax, arange
from mindnlp.abc import PreTrainedModel
from mindnlp.configs import MINDNLP_MODEL_URL_BASE
from ..utils.utils import Conv1D, prune_conv1d_layer, find_pruneable_heads_and_indices
from ..utils.utils import SequenceSummary
from ..utils.activations import ACT2FN
from .gpt_config import GPTConfig, GPT_SUPPORT_LIST


PRETRAINED_MODEL_ARCHIVE_MAP = {
    model: MINDNLP_MODEL_URL_BASE.format('gpt', model) for model in GPT_SUPPORT_LIST
}


def torch_to_mindspore(pth_file, **kwargs):
    """torch to mindspore."""
    prefix = kwargs.get("prefix", "")

    try:
        import torch
    except Exception as exc:
        raise ImportError("'import torch' failed, please install torch by "
                          "`pip install torch` or instructions from 'https://pytorch.org'") \
        from exc

    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

    for k, v in state_dict.items():
        if 'ln' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'embed' in k:
            k = k.replace('weight', 'embedding_table')
        if prefix:
            k = prefix + "." + k
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    ms_ckpt_path = pth_file.replace('pytorch_model.bin','mindspore.ckpt')
    if not os.path.exists(ms_ckpt_path):
        try:
            save_checkpoint(ms_ckpt, ms_ckpt_path)
        except Exception as exc:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.') \
            from exc

    return ms_ckpt_path


class MLP(nn.Cell):
    r"""
    GPT MLP
	"""

    def __init__(self, n_state, config):
        super().__init__()
        n_embd = config.n_embd
        self.c_fc = Conv1D(n_state, n_embd)
        self.c_proj = Conv1D(n_embd, n_state)
        self.act = ACT2FN[config.afn]
        self.dropout = Dropout(p=config.resid_pdrop)

    def construct(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Attention(nn.Cell):
    r"""
    GPT Attention
    """

    def __init__(self, nx, n_positions, config, scale=False):
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        if n_state % config.n_head != 0:
            raise ValueError(f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}")

        self.bias = Tensor(np.tril(np.ones((n_positions, n_positions))), mindspore.float32).view(1, 1, n_positions, n_positions)
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, n_state)
        self.c_attn = Conv1D(n_state * 3, n_state)
        self.c_proj = Conv1D(n_state, n_state)
        self.attn_dropout = Dropout(p=config.attn_pdrop)
        self.resid_dropout = Dropout(p=config.resid_pdrop)
        self.matmul = Matmul()
        self.pruned_heads = set()

        self.output_attentions = config.output_attentions

    def prune_heads(self, heads):
        """
        Prunes heads of the model.
        """
        if len(heads) == 0:
            return
        head_size = self.split_size//self.n_head
        heads, index = find_pruneable_heads_and_indices(heads, self.n_head, head_size, self.pruned_heads)
        index_attn = ops.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, axis=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, axis=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = self.matmul(q, k)
        if self.scale:
            w = w / ops.sqrt(ops.scalar_to_tensor(v.shape[-1]))
        b = self.bias[:, :, : w.shape[-2], : w.shape[-1]]
        w = w * b + -1e9 * (1 - b)

        if attention_mask is not None:
            w = w + attention_mask

        w = softmax(w)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask

        outputs = (self.matmul(w, v),)
        if self.output_attentions:
            outputs += (w,)
        return outputs


    def merge_heads(self, x):
        """merge heads"""
        x = x.transpose(0, 2, 1, 3)
        new_x_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.view(new_x_shape)

    def split_heads(self, x, k=False):
        """split heads"""
        new_x_shape = x.shape[:-1] + (self.n_head, x.shape[-1] // self.n_head)
        x = x.view(new_x_shape)
        if k:
            return x.transpose(0, 2, 3, 1)
        return x.transpose(0, 2, 1, 3)

    def construct(self, x, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = split(x, self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = (a,) + attn_outputs[1:]
        return outputs


class Block(nn.Cell):
    r"""
    GPT Block
    """

    def __init__(self, n_positions, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_positions, config, scale)
        self.ln_1 = nn.LayerNorm((nx,), epsilon=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm((nx,), epsilon=config.layer_norm_epsilon)

    def construct(self, x, attention_mask=None, head_mask=None):
        output_attn = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        a = output_attn[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = (h,) + output_attn[1:]
        return outputs


class GPTPreTrainedModel(PreTrainedModel):
    """BertPretrainedModel"""
    convert_torch_to_mindspore = torch_to_mindspore
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_class = GPTConfig
    base_model_prefix = 'transformer'

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            embedding_table = initializer(Normal(self.config.initializer_range),
                                                 cell.embedding_table.shape,
                                                 cell.embedding_table.dtype)
            if cell.padding_idx is not None:
                embedding_table[cell.padding_idx] = 0
            cell.embedding_table.set_data(embedding_table)
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

class GPTModel(GPTPreTrainedModel):
    """
    The bare GPT transformer model outputting raw hidden-states without any specific head on top
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([Block(config.n_positions, config, scale=True) for _ in range(config.n_layer)])
        self.position_ids = ops.arange(config.n_positions)

        self.n_layer = self.config.n_layer
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

    def get_input_embeddings(self):
        """
        return the input embeddings layer
        """
        return self.tokens_embed

    def set_input_embeddings(self, value):
        """
        set the input embeddings layer
        """
        self.tokens_embed = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
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
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * Tensor(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min,
                                                             self.dtype)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        all_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(hidden_states, attention_mask, head_mask[i])
            hidden_states = outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = hidden_states.view(*output_shape)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return (hidden_states, all_hidden_states, all_attentions)


class GPTLMHeadModel(GPTPreTrainedModel):
    r"""
    GPT Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings).
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = GPTModel(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)

    def get_output_embeddings(self):
        """
        Returns the embeddings of the obtained output
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Define the embeddings of the output
        """
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        output = (lm_logits,) + transformer_outputs[1:]
        if loss is not None:
            output = (loss,) + output
        return output


class GPTDoubleHeadsModel(GPTPreTrainedModel):
    """
    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        config.num_labels = 1
        self.transformer = GPTModel(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the embeddings of the obtained output
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Define the embeddings of the output
        """
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        mc_token_ids = None,
        labels = None,
        mc_labels = None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        lm_loss, mc_loss = None, None
        if mc_labels is not None:
            mc_loss = ops.cross_entropy(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            lm_loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        output = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_loss is not None:
            output = (mc_loss,) + output
        if lm_loss is not None:
            output = (lm_loss,) + output
        return output

class GPTForSequenceClassification(GPTPreTrainedModel):
    """
    The Original GPT Model transformer with a sequence classification head on top (linear layer).
    GPTForSequenceClassification uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the
    last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding
    token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
    it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take
    the last value in each row of the batch).
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.transformer = GPTModel(config)
        self.score = nn.Dense(config.n_embd, self.num_labels, has_bias=False)

        self.pad_token_id = self.config.pad_token_id
        problem_type = config.problem_type
        if problem_type is None:
            self.loss = None
        else:
            if self.num_labels == 1:
                self.problem_type = "regression"
                self.loss = nn.MSELoss()
            elif self.num_labels > 1:
                self.problem_type = "single_label_classification"
                self.loss = nn.CrossEntropyLoss()
            else:
                self.problem_type = "multi_label_classification"
                self.loss = nn.BCEWithLogitsLoss()

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in
            `[0, ...,config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        # Ensure the batch size is > 1 if there is no padding.
        if self.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # reduce sum not support int on Ascend.
                sequence_lengths = ops.ne(input_ids, self.pad_token_id) \
                        .astype(mindspore.float32).sum(-1) \
                        .astype(mindspore.int32) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[arange(batch_size), sequence_lengths]

        loss = None

        output = (pooled_logits,) + transformer_outputs[1:]

        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss(pooled_logits.squeeze(), labels.squeeze())
            elif self.num_labels > 1:
                loss = self.loss(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = self.loss(pooled_logits, labels)

        if loss is not None:
            output = (loss,) + output
        return output


__all__ = ['MLP', 'Attention', 'Block', 'GPTModel', 'GPTLMHeadModel',
           'GPTDoubleHeadsModel', 'GPTForSequenceClassification']
