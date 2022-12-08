# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore

from mindspore import Tensor
from mindspore import context

from mindnlp.models.bert import BertConfig, BertModel
class TestModelingBert(unittest.TestCase):
    r"""
    Test model bert
    """
    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_modeling_bert_pynative(self):
        r"""
        Test model bert with pynative mode
        """

        context.set_context(mode=context.PYNATIVE_MODE)
        config = BertConfig()
        model = BertModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)

    def test_modeling_bert_graph(self):
        r"""
        Test model bert with graph mode
        """

        context.set_context(mode=context.GRAPH_MODE)
        config = BertConfig()
        model = BertModel(config)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)

        outputs, pooled = model(input_ids)
        assert outputs.shape == (1, 512, 768)
        assert pooled.shape == (1, 768)
