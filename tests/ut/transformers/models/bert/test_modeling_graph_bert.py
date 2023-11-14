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
from packaging import version
import pytest
import numpy as np
from ddt import ddt, data

import mindspore
from mindspore import Tensor

import mindnlp
from mindnlp.utils.compatibility import MS_VERSION
from mindnlp.transformers import BertConfig, MSBertModel
from .....common import MindNLPTestCase

@ddt
class TestModelingBert(MindNLPTestCase):
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

    @pytest.mark.skipif(version.parse(MS_VERSION) < version.parse("2.2.0"), reason='older version skip')
    def test_modeling_bert(self):
        r"""
        Test model bert
        """
        model = MSBertModel(self.config)


        input_ids = Tensor(np.random.randint(1, self.config.vocab_size, (1, 512)), mindspore.int32)

        outputs, pooled = model(input_ids)

        assert outputs.shape == (1, 512, self.config.hidden_size)
        assert pooled.shape == (1, self.config.hidden_size)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = MSBertModel.from_pretrained('bert-base-uncased')
