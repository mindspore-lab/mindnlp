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
"""Test RDropLoss"""

import pytest
import numpy as np
import mindspore
from ddt import ddt, data
from mindspore import Tensor
from mindnlp.modules import RDropLoss
from mindnlp import ms_jit
from ....common import MindNLPTestCase

@ddt
class TestRDropLoss(MindNLPTestCase):
    r"""
    Test RDropLoss
    """

    def test_loss_inputs_shape_error(self):
        r"""
        Test RDropLoss input shape error
        """

        r_drop_loss = RDropLoss()
        with pytest.raises(ValueError):
            r_drop_loss(Tensor(np.array(1), mindspore.float32),
                        Tensor(np.array(2), mindspore.float32))

    def test_loss_shape(self):
        r"""
        Test RDropLoss loss shape
        """

        r_drop_loss = RDropLoss()
        temp_p = Tensor(np.array([1., 0., 1.]), mindspore.float32)
        temp_q = Tensor(np.array([0.2, 0.3, 1.1]), mindspore.float32)
        loss = r_drop_loss(temp_p, temp_q)
        assert loss.shape == ()

    @data(True, False)
    def test_loss(self, jit):
        r"""
        Test RDropLoss loss
        """
        r_drop_loss = RDropLoss()

        @ms_jit
        def forward_jit(temp_p, temp_q):
            loss = r_drop_loss(temp_p, temp_q)
            return loss

        def forward(temp_p, temp_q):
            loss = r_drop_loss(temp_p, temp_q)
            return loss

        temp_p = Tensor(np.array([1., 0., 1.]), mindspore.float32)
        temp_q = Tensor(np.array([0.2, 0.3, 1.1]), mindspore.float32)


        if jit:
            loss = forward_jit(temp_p, temp_q)
        else:
            loss = forward(temp_p, temp_q)

        assert np.allclose(loss.asnumpy(), np.array([0.10013707]), 1e-5, 1e-5)
