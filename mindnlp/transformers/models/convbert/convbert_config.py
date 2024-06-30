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
""" ConvBERT model configuration"""

from ...configuration_utils import PretrainedConfig


CONVBERT_SUPPORT_LIST = ["YituTech/conv-bert-base",
                         "YituTech/conv-bert-medium-small",
                         "YituTech/conv-bert-small"]

CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/config.json",
    "YituTech/conv-bert-medium-small": (
        "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/config.json"
    ),
    "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/config.json",
}

__all__ = ['CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP', 'ConvBertConfig']


class ConvBertConfig(PretrainedConfig):
    r"""
    ConvBert Config
    """
    model_type = "convbert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        embedding_size=768,
        head_ratio=2,
        conv_kernel_size=9,
        num_groups=1,
        classifier_dropout=None,
        **kwargs,
    ):
        """
        Initializes a new instance of the ConvBertConfig class.
        
        Args:
            self: The current instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 30522.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 768.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 12.
            intermediate_size (int, optional): The size of the intermediate layers. Defaults to 3072.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'gelu'.
            hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): The dropout probability for the attention layers. Defaults to 0.1.
            max_position_embeddings (int, optional): The maximum position embeddings. Defaults to 512.
            type_vocab_size (int, optional): The size of the type vocabulary. Defaults to 2.
            initializer_range (float, optional): The range for the weight initializer. Defaults to 0.02.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-12.
            pad_token_id (int, optional): The ID of the padding token. Defaults to 1.
            bos_token_id (int, optional): The ID of the beginning-of-sequence token. Defaults to 0.
            eos_token_id (int, optional): The ID of the end-of-sequence token. Defaults to 2.
            embedding_size (int, optional): The size of the embeddings. Defaults to 768.
            head_ratio (int, optional): The ratio of heads to hidden size. Defaults to 2.
            conv_kernel_size (int, optional): The size of the convolutional kernel. Defaults to 9.
            num_groups (int, optional): The number of groups for grouped convolution. Defaults to 1.
            classifier_dropout (float, optional): The dropout probability for the classifier layer. Defaults to None.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.head_ratio = head_ratio
        self.conv_kernel_size = conv_kernel_size
        self.num_groups = num_groups
        self.classifier_dropout = classifier_dropout
