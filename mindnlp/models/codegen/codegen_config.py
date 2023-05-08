# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# ============================================================================
""" CodeGen model configuration"""

from mindnlp.abc import PreTrainedConfig
from mindnlp.configs import MINDNLP_CONFIG_URL_BASE

__all__ = ['CodeGenConfig']

CODEGEN_SUPPORT_LIST = ["Salesforce/codegen-350M-mono"]

CONFIG_ARCHIVE_MAP = {
    model: MINDNLP_CONFIG_URL_BASE.format('codegen', model) for model in CODEGEN_SUPPORT_LIST}


class CodeGenConfig(PreTrainedConfig):
    r"""
    CodeGen config
    """
    pretrained_config_archive_map = CONFIG_ARCHIVE_MAP

    model_type = "codegen"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size=504,
            n_positions=512,
            n_ctx=512,
            n_embd=512,
            n_layer=28,
            n_head=16,
            rotary_dim=64,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.01,
            embd_pdrop=0.01,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=502,
            eos_token_id=502,
            tie_word_embeddings=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
