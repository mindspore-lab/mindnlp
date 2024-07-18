# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
""" XLM-ProphetNet model configuration"""


from typing import Callable, Optional, Union

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP={
"microsoft/xprophetnet-large-wiki100-cased":"https://hf-mirror.com/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json"
}
class XLMProphetNetConfig(PretrainedConfig):


    model_type = "xlm-prophetnet"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "num_encoder_attention_heads",
    }

    def __init__(
        self,
        activation_dropout: Optional[float] = 0.1,
        activation_function: Optional[Union[str, Callable]] = "gelu",
        vocab_size: Optional[int] = 30522,
        hidden_size: Optional[int] = 1024,
        encoder_ffn_dim: Optional[int] = 4096,
        num_encoder_layers: Optional[int] = 12,
        num_encoder_attention_heads: Optional[int] = 16,
        decoder_ffn_dim: Optional[int] = 4096,
        num_decoder_layers: Optional[int] = 12,
        num_decoder_attention_heads: Optional[int] = 16,
        attention_dropout: Optional[float] = 0.1,
        dropout: Optional[float] = 0.1,
        max_position_embeddings: Optional[int] = 512,
        init_std: Optional[float] = 0.02,
        is_encoder_decoder: Optional[bool] = True,
        add_cross_attention: Optional[bool] = True,
        decoder_start_token_id: Optional[int] = 0,
        ngram: Optional[int] = 2,
        num_buckets: Optional[int] = 32,
        relative_max_distance: Optional[int] = 128,
        disable_ngram_loss: Optional[bool] = False,
        eps: Optional[float] = 0.0,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # parameters for xlmprophetnet
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.disable_ngram_loss = disable_ngram_loss
        self.eps = eps

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            add_cross_attention=add_cross_attention,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

    @property
    def num_hidden_layers(self) -> int:
        return self.num_encoder_layers + self.num_decoder_layers

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and"
            " `num_decoder_layers`."
        )
__all__ = ['XLMProphetNetConfig']
