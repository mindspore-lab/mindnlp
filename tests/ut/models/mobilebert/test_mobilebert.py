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
Test MobileBert
"""
import unittest

import mindspore
from mindspore import Tensor
import numpy as np

from mindnlp.models.mobilebert import MobileBertSelfAttention,MobileBertConfig

class TestMobileBert(unittest.TestCase):
    """
    Test TinyBert Models
    """

    def setUp(self):
        """
        Set up config
        """

        self.input=None

    def test_mobilebert_selfattention(self):
        """
        Test MobileBertEmbeddings
        """
        mobilebert_selfattention=MobileBertSelfAttention(MobileBertConfig())
        query_tensor=Tensor(np.random.randint(0, 1000, (2,8, 128)),dtype=mindspore.float32)
        key_tensor=Tensor(np.random.randint(0, 1000, (2,8, 128)),dtype=mindspore.float32)
        value_tensor=Tensor(np.random.randint(0, 1000, (2,8, 512)),dtype=mindspore.float32)
        output = mobilebert_selfattention(query_tensor,key_tensor,value_tensor)
        assert output[0].shape == (2,8,128)
