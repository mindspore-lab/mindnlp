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
# pylint: disable=ungrouped-imports
"""Test CMRC2018Loss"""

import numpy as np
import mindspore
from ddt import ddt, data
from mindspore import Tensor
from mindnlp.modules import CMRC2018Loss
from mindnlp import ms_jit
from ....common import MindNLPTestCase

@ddt
class TestCMRC2018Loss(MindNLPTestCase):
    r"""
    Test CMRC2018Loss
    """

    @data(True, False)
    def test_loss(self, jit):
        r"""
        Test CMRC2018Loss loss
        """
        tensor_a = Tensor(np.array([1, 2, 1]), mindspore.int32)
        tensor_b = Tensor(np.array([2, 1, 2]), mindspore.int32)
        my_context_len = Tensor(np.array([2., 1., 2.]), mindspore.float32)
        tensor_c = Tensor(np.array([
            [0.1, 0.2, 0.1],
            [0.1, 0.2, 0.1],
            [0.1, 0.2, 0.1]
        ]), mindspore.float32)
        tensor_d = Tensor(np.array([
            [0.2, 0.1, 0.2],
            [0.2, 0.1, 0.2],
            [0.2, 0.1, 0.2]
        ]), mindspore.float32)

        cmrc_loss = CMRC2018Loss()

        def forward(tensor_a, tensor_b, my_context_len, tensor_c, tensor_d):
            loss = cmrc_loss(tensor_a, tensor_b, my_context_len, tensor_c, tensor_d)
            return loss

        @ms_jit
        def forward_jit(tensor_a, tensor_b, my_context_len, tensor_c, tensor_d):
            loss = cmrc_loss(tensor_a, tensor_b, my_context_len, tensor_c, tensor_d)
            return loss

        if jit:
            loss = forward_jit(tensor_a, tensor_b, my_context_len, tensor_c, tensor_d)
        else:
            loss = forward(tensor_a, tensor_b, my_context_len, tensor_c, tensor_d)
        assert loss.shape == ()
