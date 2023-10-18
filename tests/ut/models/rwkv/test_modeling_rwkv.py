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
"""Test RWKV"""
import gc
import os
import unittest
import pytest
import numpy as np

import mindspore
from mindspore import Tensor
from mindnlp.transformers.models.rwkv.rwkv_config import RwkvConfig
from mindnlp.transformers.models.rwkv.rwkv import RwkvModel


class TestModelingRWKV(unittest.TestCase):
    r"""
    Test model bert
    """
    def setUp(self) -> None:
        self.config = RwkvConfig(vocab_size=1000,
                                 hidden_size=128,
                                 num_hidden_layers=2,
                                 use_cache=False)

    @pytest.mark.gpu_only
    def test_modeling_rwkv(self):
        r"""
        Test model RWKV
        """
        model = RwkvModel(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (1, self.config.context_length)), mindspore.int32)

        def forward(input_ids):
            outputs = model(input_ids)
            return outputs

        outputs = forward(input_ids)

        assert outputs[0].shape == (1, self.config.context_length, self.config.hidden_size)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
