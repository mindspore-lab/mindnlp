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
from mindnlp.transformers.models.baichuan import BaiChuanConfig, BaiChuan7bModel, BaiChuan13bModel, \
    BaiChuanForCausalLM
from mindnlp.utils.testing_utils import slow
from .....common import MindNLPTestCase

class TestModelingBaiChuan(MindNLPTestCase):
    r"""
    Test BaiChuan
    """

    def setUp(self):
        """
        Set up.
        """
        self.config_7b = BaiChuanConfig(vocab_size=1000, num_hidden_layers=2)
        self.config_13b = BaiChuanConfig(vocab_size=1000, hidden_size=5120, num_hidden_layers=2)

    @slow
    def test_7b_model(self):
        r"""
        Test Model
        """
        model = BaiChuan7bModel(self.config_7b)
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 4096)

    @slow
    def test_13b_model(self):
        r"""
        Test Model
        """
        model = BaiChuan13bModel(self.config_13b)
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 5120)

    @slow
    def test_baichuan_for_causal_lm_7b(self):
        r"""
        Test BaiChuanForCausalLM
        """
        model = BaiChuanForCausalLM(self.config_7b, size='7b')
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 1000)

    @slow
    def test_baichuan_for_causal_lm_13b(self):
        r"""
        Test BaiChuanForCausalLM
        """
        model = BaiChuanForCausalLM(self.config_13b, size='13b')
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 1000)
