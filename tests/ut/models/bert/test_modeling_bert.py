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
import pytest
import numpy as np
from ddt import ddt, data

import mindspore
from mindspore import Tensor

import mindnlp
from mindnlp.models import BertConfig, BertModel
from ..model_test import ModelTest

@ddt
class TestModelingBert(ModelTest):
    r"""
    Test model bert
    """
    def setUp(self) -> None:
        super().setUp()
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
        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randn(1, 512), mindspore.int32)


        outputs, pooled = self.modeling(model, input_ids, jit)

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
