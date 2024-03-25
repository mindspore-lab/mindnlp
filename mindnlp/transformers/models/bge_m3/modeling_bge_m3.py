# Copyright 2024 Huawei Technologies Co., Ltd
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
Bge M3 Config.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops

from ...modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput
from ..xlm_roberta import (
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)

from .configuration_bge_m3 import BgeM3Config


@dataclass
class BgeM3ModelOutput(ModelOutput):
    last_hidden_state: mindspore.Tensor = None
    pooler_output: mindspore.Tensor = None
    dense_output: mindspore.Tensor = None
    colbert_output: Optional[List[mindspore.Tensor]] = None
    sparse_output: Optional[Dict[int, float]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    cross_attentions: Optional[Tuple[mindspore.Tensor]] = None


class BgeM3Model(XLMRobertaPreTrainedModel):
    config_class = BgeM3Config

    def __init__(self, config: BgeM3Config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.colbert_linear = nn.Dense(
            config.hidden_size,
            config.hidden_size if config.colbert_dim is None else config.colbert_dim,
        )
        self.sparse_linear = nn.Dense(config.hidden_size, 1)
        self.sentence_pooling_method = config.sentence_pooling_method

        self.init_weights()

    # Copied from FlagEmbedding
    def dense_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = ops.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    # Copied from FlagEmbedding
    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = False):
        token_weights = ops.relu(self.sparse_linear(hidden_state))
        if not return_embedding:
            return token_weights

        sparse_embedding = ops.zeros(
            (input_ids.shape[0],
            input_ids.shape[1],
            self.config.vocab_size),
            dtype=token_weights.dtype,
        )
        sparse_embedding = ops.scatter(sparse_embedding, axis=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = self.config.unused_tokens
        sparse_embedding = ops.max(sparse_embedding, axis=1)[0]
        sparse_embedding[:, unused_tokens] *= 0.0
        return sparse_embedding

    # Copied from FlagEmbedding
    def colbert_embedding(self, last_hidden_state, mask):
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    # Modified from FlagEmbedding
    def _process_token_weights(self, token_weights, input_ids, mask):
        token_weights = token_weights.squeeze(-1)
        # conver to dict
        all_result = []
        unused_tokens = self.config.unused_tokens
        unused_tokens = mindspore.tensor(unused_tokens)

        # Get valid matrix
        valid_indices = ~mindspore.numpy.isin(input_ids, unused_tokens)
        # w>0
        valid_indices = (valid_indices & (token_weights > 0)).bool()
        valid_indices = (valid_indices & mask).bool()

        for i, valid in enumerate(valid_indices):
            result = defaultdict(int)

            # Get valid weight and ids
            valid_weights = token_weights[i][valid]
            valid_ids = input_ids[i][valid]

            # Get unique token
            unique_ids, inverse_indices = ops.unique(valid_ids)

            # Get max weight for each token
            for i in range(unique_ids.shape[0]):
                id_mask = inverse_indices == i
                result[str(unique_ids[i].item())] = valid_weights[id_mask].max().item()

            all_result.append(result)

        return all_result

    # Copied from FlagEmbedding
    def _process_colbert_vecs(self, colbert_vecs, tokens_num) -> List[mindspore.Tensor]:
        # delte the vectors of padding tokens
        vecs = []
        for i in range(len(tokens_num)):
            vecs.append(colbert_vecs[i, : tokens_num[i] - 1])
        return vecs

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BgeM3ModelOutput]:
        roberta_output: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = roberta_output.last_hidden_state
        dense_output = self.dense_embedding(last_hidden_state, attention_mask)

        tokens_num = attention_mask.sum(axis=1)
        colbert_output = self.colbert_embedding(last_hidden_state, attention_mask)
        colbert_output = self._process_colbert_vecs(colbert_output, tokens_num)

        sparse_output = self.sparse_embedding(last_hidden_state, input_ids)
        sparse_output = self._process_token_weights(sparse_output, input_ids, attention_mask)

        if not return_dict:
            return (
                last_hidden_state,
                roberta_output.pooler_output,
                dense_output,
                colbert_output,
                sparse_output,
                roberta_output.hidden_states,
                roberta_output.past_key_values,
                roberta_output.attentions,
                roberta_output.cross_attentions,
            )

        return BgeM3ModelOutput(
            last_hidden_state=last_hidden_state,
            dense_output=dense_output,
            pooler_output=roberta_output.pooler_output,
            colbert_output=colbert_output,
            sparse_output=sparse_output,
            hidden_states=roberta_output.hidden_states,
            past_key_values=roberta_output.past_key_values,
            attentions=roberta_output.attentions,
            cross_attentions=roberta_output.cross_attentions,
        )

__all__ = ['BgeM3Model']
