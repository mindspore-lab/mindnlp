# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MBART model configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

MBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mbart-large-cc25": "https://hf-mirror.com/facebook/mbart-large-cc25/resolve/main/config.json",
    # See all MBART models at https://hf-mirror.com/models?filter=mbart
}


class MBartConfig(PretrainedConfig):
    """
    Configuration for MBart
    """
    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
            self,
            vocab_size=50265,
            max_position_embeddings=1024,
            encoder_layers=12,
            encoder_ffn_dim=4096,
            encoder_attention_heads=16,
            decoder_layers=12,
            decoder_ffn_dim=4096,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=1024,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            classifier_dropout=0.0,
            scale_embedding=False,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            forced_eos_token_id=2,
            **kwargs,
    ):
        """
        Initializes an instance of the MBartConfig class.
        
        Args:
            self: The instance of the class.
            vocab_size (int): The size of the vocabulary. Defaults to 50265.
            max_position_embeddings (int): The maximum number of positional embeddings. Defaults to 1024.
            encoder_layers (int): The number of encoder layers. Defaults to 12.
            encoder_ffn_dim (int): The dimension of the encoder feed-forward network. Defaults to 4096.
            encoder_attention_heads (int): The number of encoder attention heads. Defaults to 16.
            decoder_layers (int): The number of decoder layers. Defaults to 12.
            decoder_ffn_dim (int): The dimension of the decoder feed-forward network. Defaults to 4096.
            decoder_attention_heads (int): The number of decoder attention heads. Defaults to 16.
            encoder_layerdrop (float): The probability of dropping an encoder layer. Defaults to 0.0.
            decoder_layerdrop (float): The probability of dropping a decoder layer. Defaults to 0.0.
            use_cache (bool): Whether to use cache during decoding. Defaults to True.
            is_encoder_decoder (bool): Whether the model is an encoder-decoder. Defaults to True.
            activation_function (str): The activation function to be used. Defaults to 'gelu'.
            d_model (int): The dimension of the model. Defaults to 1024.
            dropout (float): The dropout rate. Defaults to 0.1.
            attention_dropout (float): The dropout rate for attention layers. Defaults to 0.0.
            activation_dropout (float): The dropout rate for activation layers. Defaults to 0.0.
            init_std (float): The standard deviation for weight initialization. Defaults to 0.02.
            classifier_dropout (float): The dropout rate for classifier layers. Defaults to 0.0.
            scale_embedding (bool): Whether to scale the embeddings. Defaults to False.
            pad_token_id (int): The ID for padding token. Defaults to 1.
            bos_token_id (int): The ID for beginning-of-sentence token. Defaults to 0.
            eos_token_id (int): The ID for end-of-sentence token. Defaults to 2.
            forced_eos_token_id (int): The ID for forced end-of-sentence token. Defaults to 2.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )


__all__ = ['MBartConfig']
