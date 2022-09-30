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
"""Test Metrics"""

import unittest
import math
from functools import reduce
from scipy import stats

from mindspore.common.initializer import initializer
from mindnlp.common.initializer import XavierNormal

class TestXavierNormal(unittest.TestCase):
    r"""
    Test XavierNormal
    """

    def setUp(self):
        self.input = None

    def test_xaviernormal(self):
        """
        Test XavierNormal
        """
        gain = 1.2
        tensor1 = initializer(XavierNormal(gain=gain), [20, 22])
        tensor2 = initializer(XavierNormal(), [20, 22])
        tensor3 = initializer(XavierNormal(gain=gain), [20, 22, 5, 5])
        tensor4 = initializer(XavierNormal(), [20, 22, 5, 5])
        tensor_dict = {tensor1: gain, tensor2: None, tensor3: gain, tensor4: None}

        for tensor, gain_value in tensor_dict.items():
            if gain_value is None:
                gain_value = 1
            shape = tensor.asnumpy().shape
            if len(shape) > 2:
                s_num = reduce(lambda x, y: x * y, shape[2:])
            else:
                s_num = 1

            fan_in = shape[1] * s_num
            fan_out = shape[0] * s_num
            std = gain_value * math.sqrt(2 / (fan_in + fan_out))
            samples = tensor.asnumpy().reshape((-1))
            _, prob = stats.kstest(samples, 'norm', (0, std))
            assert prob > 0.0001
