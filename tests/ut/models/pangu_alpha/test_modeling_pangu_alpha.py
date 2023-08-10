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
# pylint:disable=W0212
"""Test PanGu_Alpha"""
import gc
import os
import unittest
import pytest
import numpy as np

import mindspore
from mindspore import Tensor
from mindnlp.models.pangu_alpha import PanGuAlphaConfig
from mindnlp.models.pangu_alpha import PanGuAlphaAttention, PanGuAlphaMLP, PanGuAlphaBlock, \
                                        PanGuAlphaModel, PanGuAlphaForCausalLM


class TestModelingPanGuAlpha(unittest.TestCase):
    """
    Test PanGu-Alpha
    """

    def setUp(self):
        """
        Set up.
        """
        self.config = PanGuAlphaConfig()

    def test_pangu_alpha_attention(self):
        """
        Test PanGu-Alpha Attention
        """
        model = PanGuAlphaAttention(self.config)

        hidden_states = Tensor(np.random.randint(0, self.config.vocab_size, (2, 512, self.config.hidden_size)), mindspore.float32)

        attn_output, _ = model(hidden_states)
        assert attn_output.shape == (2, 512, self.config.hidden_size)

    def test_pangu_alpha_mlp(self):
        """
        Test PanGu-Alpha MLP
        """
        model = PanGuAlphaMLP(self.config.hidden_size, self.config)

        hidden_states = Tensor(np.random.randint(0, self.config.vocab_size, (2, 512, self.config.hidden_size)), mindspore.float32)

        hidden_states = model(hidden_states)
        assert hidden_states.shape == (2, 512, self.config.hidden_size)

    def test_pangu_alpha_block(self):
        """
        Test PanGu-Alpha Block
        """
        model = PanGuAlphaBlock(self.config)

        hidden_states = Tensor(np.random.randint(0, self.config.vocab_size, (2, 512, self.config.hidden_size)), mindspore.float32)

        outputs = model(hidden_states)
        assert outputs[0].shape == (2, 512, self.config.hidden_size)

    def test_pangu_alpha_model(self):
        """
        Test PanGu-Alpha Model
        """
        model = PanGuAlphaModel(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (2, 512)))

        output_dict = model(input_ids, return_dict=True)
        assert output_dict['last_hidden_state'].shape == (2, 512, self.config.hidden_size)

    def test_pangu_alpha_for_causallm(self):
        """
        Test PanGu-Alpha for CausalLM
        """
        model = PanGuAlphaForCausalLM(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (2, 512)))

        output = model(input_ids, return_dict=True, output_hidden_states=True)
        assert output['logits'].shape == (2, 512, self.config.vocab_size)
        assert output['hidden_states'][0].shape == (2, 512, self.config.hidden_size)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = PanGuAlphaModel.from_pretrained('pangu-350M')

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
