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
"""Test Roberta"""
import pytest
import numpy as np
from ddt import ddt, data

import mindspore
import unittest
from mindspore import Tensor

import mindnlp
from mindnlp.transformers import RobertaConfig, RobertaModel

@ddt
class TestModelingRoberta(unittest.TestCase):
    r"""
    Test model bert
    """
    def setUp(self) -> None:
        super().setUp()
        self.config = RobertaConfig(vocab_size=1000,
                                 hidden_size=128,
                                 num_hidden_layers=2,
                                 num_attention_heads=8,
                                 intermediate_size=256,
                                 pad_token_id=0)

    def test_modeling_roberta(self):
        r"""
        Test model bert
        """
        model = RobertaModel(self.config)

        input_ids = Tensor(np.random.randint(1, self.config.vocab_size, (1, 256)), mindspore.int64)

        outputs = model(input_ids)

        assert outputs.last_hidden_state.shape == (1, 256, self.config.hidden_size)
        assert outputs.pooler_output.shape == (1, self.config.hidden_size)


    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = RobertaModel.from_pretrained('roberta-base', from_pt=True)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = RobertaModel.from_pretrained('roberta-base')
