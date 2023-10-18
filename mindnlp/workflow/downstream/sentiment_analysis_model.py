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

""" Sentiment Analysis Model """

# pylint: disable=arguments-differ

from mindspore import nn
from mindnlp.transformers import BertModel

class BertForSentimentAnalysis(nn.Cell):
    """Bert Model for classification tasks"""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, \
        position_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits
