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
""" OPT model configuration"""
from mindnlp.configs import MINDNLP_CONFIG_URL_BASE
from ...configuration_utils import PreTrainedConfig


__all__ = ['OPTConfig']

OPT_SUPPORT_LIST = ["opt-350m"]

CONFIG_ARCHIVE_MAP = {
    model: MINDNLP_CONFIG_URL_BASE.format('opt', model) for model in OPT_SUPPORT_LIST
}


OPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
}


class OPTConfig(PreTrainedConfig):
    """
    Configuration for opt
    """
    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]
    pretrained_config_archive_map = CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=12,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before
        # We keep these variables at `True` for backward compatibility.
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine
        self.initializer_range = initializer_range

        # Note that the only purpose of `_remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        self._remove_final_layer_norm = _remove_final_layer_norm
