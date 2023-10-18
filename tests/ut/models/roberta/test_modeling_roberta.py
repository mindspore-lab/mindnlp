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

from mindspore import Tensor

import mindnlp
from mindnlp.transformers import RobertaConfig, RobertaModel
from ..model_test import ModelTest

@ddt
class TestModelingRoberta(ModelTest):
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

    @data(False)
    def test_modeling_roberta(self, jit):
        r"""
        Test model bert
        """
        model = RobertaModel(self.config)
        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randint(1, self.config.vocab_size, (1, 256)), mindspore.int64)

        outputs, pooled = self.modeling(model, input_ids, jit)

        assert outputs.shape == (1, 256, self.config.hidden_size)
        assert pooled.shape == (1, self.config.hidden_size)


    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = RobertaModel.from_pretrained('roberta-base', from_pt=True)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = RobertaModel.from_pretrained('roberta-base')
