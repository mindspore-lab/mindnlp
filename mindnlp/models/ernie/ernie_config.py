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
"""
Ernie config
"""

import re

from mindnlp.abc import PreTrainedConfig
from mindnlp.configs import MINDNLP_CONFIG_URL_BASE

ERNIE_SUPPORT_LIST = [
    "uie-base",
    "uie-medium",
    "uie-mini",
    "uie-micro",
    "uie-nano",
    "uie-base-en",
    "uie-senta-base",
    "uie-senta-medium",
    "uie-senta-mini",
    "uie-senta-micro",
    "uie-senta-nano",
    "uie-base-answer-extractor",
    "uie-base-qa-filter",
    "ernie-3.0-base-zh",
    "ernie-3.0-xbase-zh",
    "ernie-3.0-medium-zh",
    "ernie-3.0-mini-zh",
    "ernie-3.0-micro-zh",
    "ernie-3.0-nano-zh",
    "ernie-3.0-tiny-base-v1-zh",
    "ernie-3.0-tiny-medium-v1-zh",
    "ernie-3.0-tiny-mini-v1-zh",
    "ernie-3.0-tiny-micro-v1-zh",
    "ernie-3.0-tiny-nano-v1-zh"
]

CONFIG_ARCHIVE_MAP = {
    model: MINDNLP_CONFIG_URL_BASE.format(re.search(r"^[^-]*", model).group(), model)
    for model in ERNIE_SUPPORT_LIST
}


class ErnieConfig(PreTrainedConfig):
    """
    Configuration for Ernie.
    """

    pretrained_config_archive_map = CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        task_id=0,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        task_type_vocab_size: int = 3,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        fuse: bool = False,
        layer_norm_eps=1e-12,
        use_cache=False,
        use_task_id=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.task_id = task_id
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_task_id = use_task_id


class UIEConfig(PreTrainedConfig):
    """
    Configuration for UIE.
    """

    pretrained_config_archive_map = CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        vocab_size: int = 40000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        task_id=0,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        task_type_vocab_size: int = 3,
        type_vocab_size: int = 4,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        fuse: bool = False,
        layer_norm_eps=1e-12,
        use_cache=False,
        use_task_id=True,
        enable_recompute=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.task_id = task_id
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_task_id = use_task_id
        self.enable_recompute = enable_recompute


__all__ = ["ErnieConfig", "UIEConfig"]
