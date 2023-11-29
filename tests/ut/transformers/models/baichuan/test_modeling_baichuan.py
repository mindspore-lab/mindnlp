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
# pylint:disable=R0904
# pylint:disable=W0611
"""Test BaiChuan"""
import gc
import os
import unittest

import mindspore
import numpy as np
from mindspore import Tensor

from mindnlp.transformers.models.baichuan import baichuan, baichuan_config
from .....common import MindNLPTestCase

class TestModelingBaiChuan(MindNLPTestCase):
    r"""
    Test BaiChuan
    """

    def setUp(self):
        """
        Set up.
        """
        self.config = baichuan_config.BaiChuanConfig(vocab_size=1000, hidden_size=128, num_hidden_layers=2)

    def test_attention(self):
        r"""
        Test Attention
        """
        model = baichuan.Attention(self.config)
        hidden_states = Tensor(np.random.randint(0, self.config.vocab_size, (1, 128, 128)), mindspore.float32)
        outputs = model(hidden_states)
        assert outputs[0].shape == (1, 128, 128)

    def test_decoder_layer(self):
        r"""
        Test DecoderLayer
        """
        model = baichuan.DecoderLayer(self.config)
        hidden_states = Tensor(np.random.randint(0, self.config.vocab_size, (1, 128, 128)), mindspore.int32)
        outputs = model(hidden_states)
        assert outputs[0].shape == (1, 128, 128)

    def test_model(self):
        r"""
        Test Model
        """
        model = baichuan.Model(self.config)
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 128)

    def test_baichuan_for_causal_lm(self):
        r"""
        Test BaiChuanForCausalLM
        """
        model = baichuan.BaiChuanForCausalLM(self.config)
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 1000)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
