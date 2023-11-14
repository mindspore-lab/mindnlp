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
"""MindNLP CPM model"""
from typing import Optional, Tuple
import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, dtype_to_nptype

from mindnlp._legacy.nn import Dropout
from mindnlp._legacy.functional import arange

from .cpm_config import CpmConfig
from ..gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Block

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/pytorch_model.bin"
}

class CpmPreTrainedModel(GPT2PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CpmConfig

    base_model_prefix = "transformer"


class CpmModel(CpmPreTrainedModel):
    r"""
    Cpm Model
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm((self.embed_dim,), epsilon=config.layer_norm_epsilon)

        self.add_cross_attention = self.config.add_cross_attention
        self.num_hidden_layers = self.config.num_hidden_layers
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.use_cache = self.config.use_cache

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        return the input embeddings layer
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        """
        set the input embeddings layer
        """
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def construct(
            self,
            input_ids: Tensor,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]
        if position_ids is None:
            position_ids = arange(past_length, input_shape[-1] + past_length, 1, dtype=mindspore.int64)
            position_ids = position_ids.expand_dims(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.astype(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * Tensor(np.finfo(dtype_to_nptype(self.dtype)).min, self.dtype)

        if self.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        presents = () if self.use_cache else None
        all_self_attentions = ()
        all_cross_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=past_key_values[i],
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=self.use_cache,
            )

            hidden_states = outputs[0]
            if self.use_cache:
                presents = presents + (outputs[1],)

            if self.output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if self.use_cache else 1],)
                if self.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if self.use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_attentions:
            outputs += (all_hidden_states, all_self_attentions)
            if self.add_cross_attention:
                outputs += (all_cross_attentions,)
        return outputs


class CpmLMHeadModel(CpmPreTrainedModel):
    r"""
    CPM LMHead Model
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = CpmModel(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        ignore_index = kwargs.pop('ignore_index', -1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # Initialize weights and apply final processing
        self.post_init()


    def get_output_embeddings(self):
        """
        return the output embeddings layer
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        set the output embeddings layer
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        prepare inputs for generation task
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].expand_dims(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].expand_dims(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].expand_dims(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def construct(
            self,
            input_ids: Tensor,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        output = (lm_logits,) + transformer_outputs[1:]
        if loss is not None:
            output = (loss,) + output
        return output

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.astype(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


__all__ = ['CpmLMHeadModel', 'CpmModel']
