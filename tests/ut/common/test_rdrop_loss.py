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

"""Test RDropLoss"""
# pylint: disable=C0415

import unittest
import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.common.loss import RDropLoss


class TestRDropLoss(unittest.TestCase):
    r"""
    Test RDropLoss
    """

    def setUp(self):
        self.input = None

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

    @pytest.mark.skip(reason="this ut has already tested")
    def test_loss(self):
        r"""
        Test RDropLoss loss
        """

        import paddle
        import paddlenlp
        from mindspore import grad

        # mindnlp
        r_drop_loss_mn = RDropLoss()
        temp_p_mn = Tensor(np.array([1., 0., 1.]), mindspore.float32)
        temp_q_mn = Tensor(np.array([0.2, 0.3, 1.1]), mindspore.float32)
        loss_mn = r_drop_loss_mn(temp_p_mn, temp_q_mn)

        # paddle
        r_drop_loss_pd = paddlenlp.losses.RDropLoss()
        temp_p_pd = paddle.to_tensor(
            data=np.array([1., 0., 1.]),
            dtype='float32',
            stop_gradient=False
        )
        temp_q_pd = paddle.to_tensor(
            data=np.array([0.2, 0.3, 1.1]),
            dtype='float32',
            stop_gradient=False
        )
        loss_pd = r_drop_loss_pd(temp_p_pd, temp_q_pd)

        assert np.allclose(loss_mn.asnumpy(), loss_pd.numpy())

        # mindspore
        gradient_mn = grad(r_drop_loss_mn, grad_position=(0, 1))(temp_p_mn, temp_q_mn)

        # paddle
        gradient_pd = paddle.grad(
            outputs = loss_pd,
            inputs = [temp_p_pd, temp_q_pd],
            retain_graph = True,
            create_graph = False,
            only_inputs = True
        )

        for i, j in zip(gradient_mn, gradient_pd):
            assert np.allclose(i.asnumpy(), j.numpy())
