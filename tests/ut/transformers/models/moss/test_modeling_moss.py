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
# pylint: disable=C0103
# pylint: disable=C0415
"""
Test Moss functions
"""
import unittest
import numpy as np
import pytest
import mindspore

from mindspore import Tensor

from mindnlp.transformers.models.moss import MossConfig, MossAttention, MossMLP, MossBlock, MossModel, MossForCausalLM
from .....common import MindNLPTestCase

@pytest.mark.gpu_only
class TestModelingMoss(MindNLPTestCase):
    r"""
    Test Moss
    """

    def setUp(self):
        """
        Set up.
        """
        self.MossAttention = MossAttention
        self.MossMLP = MossMLP
        self.MossBlock = MossBlock
        self.MossModel = MossModel
        self.MossForCausalLM = MossForCausalLM
        self.config = MossConfig(vocab_size=1000,
                                 n_positions=512,
                                 n_layer=2,
                                 n_head=8,
                                 n_embd=512)

    def test_moss_attention(self):
        r"""
        Test MossAttention
        """
        model = self.MossAttention(self.config)

        hidden_states = Tensor(np.random.randint(1, 16, (2, 512, self.config.n_embd)), dtype=mindspore.float32)

        position_ids = Tensor(np.random.randint(0, 1, (2, self.config.n_embd)), dtype=mindspore.int64)

        attn_output, _ = model(position_ids=position_ids, hidden_states=hidden_states, output_attentions=False)
        assert attn_output.shape == (2, 512, self.config.n_embd)

    def test_moss_mlp(self):
        r"""
        Test MossMLP
        """
        intermediate_size = 4096

        model = self.MossMLP(intermediate_size, self.config)

        hidden_states = Tensor(np.random.randint(1, 16, (2, 512, self.config.n_embd)), dtype=mindspore.float32)

        mlp_out = model(hidden_states=hidden_states)

        assert mlp_out.shape == (2, 512, self.config.n_embd)

    def test_moss_block(self):
        r"""
        Test MossBlock
        """
        model = self.MossBlock(self.config)

        hidden_states = Tensor(np.random.randint(1, 16, (2, 512, self.config.n_embd)), dtype=mindspore.float32)
        position_ids = Tensor(np.random.randint(0, 1, (2, self.config.n_embd)), dtype=mindspore.int64)

        block_out = model(position_ids=position_ids, hidden_states=hidden_states)[0]

        assert block_out.shape == (2, 512, self.config.n_embd)

    def test_moss_model(self):
        r"""
        Test MossModel
        """
        model = self.MossModel(self.config)

        ms_input = Tensor(np.random.randint(0, 512, (2, 512)))

        model_out = model(input_ids=ms_input)[0]

        assert model_out.shape == (2, 512, 512)

    def test_moss_for_causal_lm(self):
        r"""
        Test MossForCausalLM
        """
        model = self.MossForCausalLM(self.config)

        ms_input = Tensor(np.random.randint(0, 100, 2), dtype=mindspore.int64)

        model_out = model(input_ids=ms_input)

        assert model_out[0].shape == (2, self.config.vocab_size)
        assert model_out[1][0][0].shape == (1, 8, 2, 64)
