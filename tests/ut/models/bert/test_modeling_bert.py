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
"""Test Bert"""
import gc
import os
import unittest
import pytest
import numpy as np
from ddt import ddt, data

import mindspore
from mindspore import Tensor
from mindnlp import ms_jit
from mindnlp.models import BertConfig, BertModel


@ddt
class TestModelingBert(unittest.TestCase):
    r"""
    Test model bert
    """
    def setUp(self) -> None:
        self.config = BertConfig(vocab_size=1000,
                                 hidden_size=128,
                                 num_hidden_layers=2,
                                 num_attention_heads=8,
                                 intermediate_size=256)
    @data(True, False)
    def test_modeling_bert(self, jit):
        r"""
        Test model bert
        """
        model = BertModel(self.config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        def forward(input_ids):
            outputs, pooled = model(input_ids)
            return outputs, pooled

        if jit:
            forward = ms_jit(forward)

        outputs, pooled = forward(input_ids)

        assert outputs.shape == (1, 512, self.config.hidden_size)
        assert pooled.shape == (1, self.config.hidden_size)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = BertModel.from_pretrained('bert-base-uncased')

    @pytest.mark.download
    def test_from_pretrained_path(self):
        """test from pretrained"""
        _ = BertModel.from_pretrained('.mindnlp/models/bert-base-uncased')

    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = BertModel.from_pretrained('bert-base-uncased', from_pt=True)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
