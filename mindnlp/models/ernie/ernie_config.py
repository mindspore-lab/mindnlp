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
UIE config
"""

from typing import Dict
from mindnlp.abc import PreTrainedConfig

ERNIE_PRETRAINED_INIT_CONFIGURATION = {
    "uie-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-medium": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-mini": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-micro": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-nano": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 312,
        "intermediate_size": 1248,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-base-en": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "uie-senta-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-senta-medium": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-senta-mini": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-senta-micro": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-senta-nano": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 312,
        "intermediate_size": 1248,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-base-answer-extractor": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "uie-base-qa-filter": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    }
}
ERNIE_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "uie-base": "",
        "uie-medium": "",
        "uie-mini": "",
        "uie-micro": "",
        "uie-nano": "",
        "uie-base-en": "",
        "uie-senta-base": "",
        "uie-senta-medium": "",
        "uie-senta-mini": "",
        "uie-senta-micro": "",
        "uie-senta-nano": "",
        "uie-base-answer-extractor": "",
        "uie-base-qa-filter": "",
    }
}


class ErnieConfig(PreTrainedConfig):
    """
    Configuration for Ernie.
    """
    model_type = "ernie"
    attribute_map: Dict[str, str] = {
        "dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION

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
class UIEConfig(PreTrainedConfig):
    """
    Configuration for UIE.
    """
    model_type = "ernie"
    attribute_map: Dict[str, str] = {
        "dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION

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
