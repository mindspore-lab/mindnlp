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
"""
Test XLM
"""
import numpy as np
import unittest
import mindspore
from mindnlp.models.xlm import xlm_config
from mindnlp.models import xlm

class TestXlm(unittest.TestCase):
    """
    Test XLM Models
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up config
        """
        cls.xlm_config = xlm_config.XLMConfig(n_words=22)

    def test_xlm_predlayer(self):
        """
        Test xlm_XLMPredLayer
        """
        xlm_XLMPredLayer = xlm.XLMPredLayer(self.xlm_config)

        input_ids = mindspore.Tensor(np.random.randint(0, 1000, (2, 2048)),mindspore.float32)

        output = xlm_XLMPredLayer(input_ids)

