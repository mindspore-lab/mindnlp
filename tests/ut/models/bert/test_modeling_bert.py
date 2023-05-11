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
    @data(True, False)
    def test_modeling_bert(self, jit):
        r"""
        Test model bert
        """

        config = BertConfig(num_hidden_layers=2)
        model = BertModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        def forward(input_ids):
            outputs, pooled = model(input_ids)
            return outputs, pooled

        if jit:
            forward = ms_jit(forward)

        outputs, pooled = forward(input_ids)

        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = BertModel.from_pretrained('bert-base-uncased')

    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = BertModel.from_pretrained('bert-base-uncased', from_pt=True)
