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
# pylint: disable=W0223

"""roberta model, base on bert."""
import mindspore
from mindspore import nn, ops
from mindspore import Parameter
from mindspore.common.initializer import initializer

from mindnlp.configs import MINDNLP_MODEL_URL_BASE
from mindnlp._legacy.nn import Dropout
from mindnlp.models.bert.bert import BertEmbeddings, BertModel, BertPreTrainedModel
from .roberta_config import RobertaConfig, ROBERTA_SUPPORT_LIST


PRETRAINED_MODEL_ARCHIVE_MAP = {
    model: MINDNLP_MODEL_URL_BASE.format('roberta', model) for model in ROBERTA_SUPPORT_LIST
}

class RobertaEmbeddings(BertEmbeddings):
    """Roberta embeddings"""
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = 1

    def construct(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = ops.arange(self.padding_idx + 1, seq_length + self.padding_idx+1, dtype=mindspore.int64)
            position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        return super().construct(input_ids, token_type_ids, position_ids)

class RobertaPreTrainedModel(BertPreTrainedModel):
    """Roberta Pretrained Model."""
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_class = RobertaConfig

class RobertaModel(BertModel):
    """Roberta Model"""
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)

class RobertaLMHead(nn.Cell):
    """RobertaLMHead"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)

        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.bias = Parameter(initializer('zeros', config.vocab_size), 'bias')
        self.gelu = nn.GELU()

    def construct(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x) + self.bias
        return x

class RobertaClassificationHead(nn.Cell):
    """RobertaClassificationHead"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size, activation='tanh')
        self.dropout = Dropout(p=1-config.hidden_dropout_prob)
        self.out_proj = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForMaskedLM(RobertaPreTrainedModel):
    """RobertaForMaskedLM"""
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.embedding_table
        self.vocab_size = self.config.vocab_size

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.vocab_size),
                                               masked_lm_labels.view(-1), ignore_index=-1)
            outputs = (masked_lm_loss,) + outputs

        return outputs

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    """RobertaForSequenceClassification"""
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss = ops.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class RobertaForMultipleChoice(RobertaPreTrainedModel):
    """RobertaForMultipleChoice"""
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.roberta = RobertaModel(config)
        self.dropout = Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)

    def construct(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                  position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = ops.cross_entropy(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

__all__ = ['RobertaEmbeddings', 'RobertaModel', 'RobertaLMHead', 'RobertaPreTrainedModel',
           'RobertaForMaskedLM', 'RobertaClassificationHead', 'RobertaForMultipleChoice',
           'RobertaForSequenceClassification']
