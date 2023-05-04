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
"""Test GLM"""
import unittest
import numpy as np

import mindspore
from mindspore import Tensor

from mindnlp.models.glm import glm,glm_config

class TestModelingGlm(unittest.TestCase):
    """
    Test Glm
    """
    def setUp(self):
        """
        Set up.
        """
        self.config = glm_config.GLMConfig()

    def test_position_embedding(self):
        """
        test position embedding
        """
        model = glm.PositionalEmbedding(hidden_size=512)
        pos_seq = Tensor(np.random.randint(0, 10, (10,)), mindspore.float32)
        output = model(pos_seq)
        assert output.shape == (1,10,512)

    def test_promptspell(self):
        """
        test promptspell
        """
        model = glm.PromptSpell(spell_length=10, hidden_size=20, spell_func="lstm")
        output = model()

        assert output.shape == (1,10,20)

    def test_glm_selfattention(self):
        """
        test glm selfattention
        """
        config = self.config
        model = glm.GLMSelfAttention(config=config)
        selfattention_input = Tensor(np.random.randint(0, 10, (2,50,512)), mindspore.float32)
        output = model(selfattention_input)

        assert output.shape == (2, 50, 512)

    def test_glm_crossattention(self):
        """
        test glm cross-attention
        """
        config = self.config
        model = glm.GLMCrossAttention(config=config)
        hidden_states_input = Tensor(np.random.randint(0, 10, (2,50,512)), mindspore.float32)
        encoder_states_input = Tensor(np.random.randint(0, 10, (2,50,512)), mindspore.float32)
        output = model(hidden_states_input, encoder_states_input)

        assert output.shape == (2, 50, 512)

    def test_glm_mlp(self):
        """
        test glm mlp
        """
        config = self.config
        model = glm.GlmMLP(config=config)
        mlp_input = Tensor(np.random.randint(0, 10, (2,50,512)), mindspore.float32)
        output = model(mlp_input)
        assert output.shape == (2, 50, 512)

    def test_glm_transformer_layer(self):
        """
        test glm transformer layer
        """
        model = glm.GLMTransformerLayer(config=self.config)
        transformer_layer_input = Tensor(np.random.randint(0, 10, (2,100,512)), mindspore.float32)
        ltor_mask_input = Tensor(np.random.randint(0, 1, (1,1,100,100)), mindspore.float32)
        output = model(transformer_layer_input, ltor_mask_input)

        assert output.shape == (2, 100, 512)

    def test_glm_decoderlayer(self):
        """
        test glm decoderlayer
        """
        model = glm.GLMDecoderLayer(config=self.config)
        hidden_states_input = Tensor(np.random.randint(0, 10, (2,100,512)), mindspore.float32)
        encoder_states_input = Tensor(np.random.randint(0, 1, (2,100,512)), mindspore.float32)
        ltor_mask_input = Tensor(np.random.randint(0, 1, (1,1,100,100)), mindspore.float32)
        output = model(hidden_states_input, encoder_states_input, ltor_mask_input)
        assert output.shape == (2, 100, 512)

    def test_glm_transformer(self):
        """
        test glm transformer
        """
        model = glm.GLMTransformer(config=self.config)
        batch_size = 10
        max_sequence_length = 100
        hidden_size = 512
        hidden_states_input = Tensor(np.random.randint(0, 10, (10, 100, 512)), dtype = mindspore.float32)
        position_ids_input = Tensor(np.random.randint(0, 1, (10, 100)), dtype = mindspore.int32)
        attention_mask_input = Tensor(np.random.randint(0, 1, (10, 1, 1, 100)), dtype = mindspore.int32)
        output = model(hidden_states_input, position_ids_input, attention_mask_input)

        assert output[0].shape == (batch_size,max_sequence_length,hidden_size)

    def test_glm_model(self):
        """
        test glm model
        """
        model = glm.GLMModel(self.config)
        batch_size = 10
        max_sequence_length = 100
        hidden_size = 512
        mems = None
        input_ids_input = Tensor(np.random.randint(0, 10,
                                (batch_size, max_sequence_length)), dtype = mindspore.int32)
        position_ids_input = Tensor(np.random.randint(0, 1,
                                (batch_size, max_sequence_length)), dtype = mindspore.int32)
        attention_mask_input = Tensor(np.random.randint(0, 1,
                                (batch_size, 1, 1, max_sequence_length)), dtype = mindspore.int32)

        output_tuple = model(input_ids_input, position_ids_input, attention_mask_input, mems)
        output = output_tuple[0]
        assert output.shape == (batch_size, max_sequence_length, hidden_size)
