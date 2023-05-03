# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""
GLM Model Config
"""

from mindnlp.abc.backbones.pretrained import PreTrainedConfig
from mindnlp.models.glm import glm

class GLMConfig(PreTrainedConfig):
    """
    GLMConfig
    """

    def __init__(
        self,
        num_layers=12,
        vocab_size=50257,
        hidden_size=512,
        num_attention_heads=4,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        max_sequence_length=100,
        max_memory_length=0,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        init_method_std = 0.02,
        relative_encoding=False,
        layernorm_epsilon=1.0e-5,
        block_position_encoding=False,
        output_predict=True,
        spell_length=None,
        spell_func='lstm',
        attention_scale=1.0,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.max_memory_length = max_memory_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.init_method_std = init_method_std
        self.relative_encoding = relative_encoding
        self.layernorm_epsilon = layernorm_epsilon
        self.block_position_encoding = block_position_encoding
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale
        self.output_predict = output_predict

        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size, spell_func)


