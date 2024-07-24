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


from mindspore import nn
from mindnlp.transformers import BertModel

class BertForSentimentAnalysis(nn.Module):
    """Bert Model for classification tasks"""
    def __init__(self, config):
        """
        Initializes a BertForSentimentAnalysis instance.
        
        Args:
            self: Instance of the BertForSentimentAnalysis class.
            config: An object containing configuration parameters for the model.
                    Expected to be an instance of a class that has the following attributes:
                    - num_labels: An integer representing the number of labels for classification.
                    - hidden_size: An integer representing the size of hidden layers.
        
        Returns:
            None. This method initializes the BertForSentimentAnalysis instance with the provided configuration.
        
        Raises:
            - TypeError: If the config parameter is not in the expected format or is missing required attributes.
            - ValueError: If the num_labels or hidden_size attributes in the config parameter are invalid.
            - RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, \
        position_ids=None, head_mask=None):
        """
        Constructs the sentiment analysis model using the BERT architecture.
        
        Args:
            self (BertForSentimentAnalysis): An instance of the BertForSentimentAnalysis class.
            input_ids (Tensor): The input sequence of token IDs, of shape (batch_size, sequence_length).
            attention_mask (Tensor, optional): The attention mask, indicating which tokens should be attended to, of shape (batch_size, sequence_length). Defaults to None.
            token_type_ids (Tensor, optional): The token type IDs, indicating the segment IDs for the tokens in the input sequence, of shape (batch_size, sequence_length). Defaults to None.
            position_ids (Tensor, optional): The position IDs, indicating the position of each token in the input sequence, of shape (batch_size, sequence_length). Defaults to None.
            head_mask (Tensor, optional): The head mask, indicating which attention heads to mask out, of shape (batch_size, num_heads). Defaults to None.
        
        Returns:
            logits (Tensor): The output logits representing the sentiment scores for each input sequence, of shape (batch_size, num_classes).
        
        Raises:
            None.
        """
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
