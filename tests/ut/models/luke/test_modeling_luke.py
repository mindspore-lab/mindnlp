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
"""Test LUKE"""
import unittest

import mindspore
import numpy as np
from mindspore import Tensor

from mindnlp.models.luke import luke_config, luke


class TestModelingLUKE(unittest.TestCase):
    r"""
    Test LUKE
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_luke_embeddings(self):
        r"""
        Test LukeEmbeddings
        """
        config = luke_config.LukeConfig()
        model = luke.LukeEmbeddings(config)
        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        outputs = model(input_ids)
        assert outputs.shape == (1, 512, 768)

    def test_luke_entity_embeddings(self):
        r"""
        Test LukeEntityEmbeddings
        """
        config = luke_config.LukeConfig()
        model = luke.LukeEmbeddings(config)
        entity_ids = Tensor(np.random.randn(1, ), mindspore.int32)
        position_ids = Tensor(np.random.randn(1, 512), mindspore.int32)
        outputs = model(entity_ids, position_ids)
        assert outputs.shape == (1, 512, 768)

    def test_luke_self_attention(self):
        r"""
        Test LukeSelfAttention
        """
        config = luke_config.LukeConfig()
        model = luke.LukeSelfAttention(config)
        word_hidden_states = Tensor(np.random.randn(2, 768), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(2, 768), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 256, 768)
        assert outputs[1].shape == (1, 512, 768)

    def test_luke_self_output(self):
        r"""
        Test LukeSelfOutput
        """
        config = luke_config.LukeConfig()
        model = luke.LukeSelfOutput(config)
        hidden_states = Tensor(np.random.randn(2, 768), mindspore.float32)
        input_tensor = Tensor(np.random.randn(2, 768), mindspore.float32)
        outputs = model(hidden_states, input_tensor)
        assert outputs.shape == (2, 768)

    def test_luke_attention(self):
        r"""
        Test LukeAttention
        """
        config = luke_config.LukeConfig()
        model = luke.LukeAttention(config)
        word_hidden_states = Tensor(np.random.randn(1, 384, 768), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 512, 768), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 384, 768)
        assert outputs[1].shape == (1, 512, 768)
