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
# pylint: disable=C0103

"""test gru."""
import mindspore
import numpy as np
from mindspore import Tensor, nn
from mindnlp.modules import StaticGRU
from ....common import MindNLPTestCase

class TestGRU(MindNLPTestCase):
    """test gru"""
    def setUp(self):
        self.input_size, self.hidden_size = 16, 32
        self.x = np.random.randn(3, 10, self.input_size)

    def test_gru(self):
        """test simple gru"""
        rnn = StaticGRU(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 32)
        assert h.shape == (1, 3, 32)

    def test_gru_long(self):
        """test simple gru with long sequence"""
        self.x = np.random.randn(3, 10000, self.input_size)
        rnn = StaticGRU(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10000, self.hidden_size)
        assert h.shape == (1, 3, self.hidden_size)

    def test_gru_fp16(self):
        """test simple gru with fp16"""
        rnn = StaticGRU(self.input_size, self.hidden_size, batch_first=True).to_float(mindspore.float16)
        inputs = Tensor(self.x, mindspore.float16)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 32)
        assert h.shape == (1, 3, 32)
        assert output.dtype == mindspore.float16

    def test_gru_bidirection(self):
        """test bidirectional gru"""
        rnn = StaticGRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 32 * 2)
        assert h.shape == (2, 3, 32)

    def test_gru_multi_layer(self):
        """test multilayer gru"""
        rnn = StaticGRU(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 32)
        assert h.shape == (1 * 3, 3, 32)

    def test_gru_single_layer_precision(self):
        """test simple gru precision"""
        static_rnn = StaticGRU(self.input_size, self.hidden_size, num_layers=1, batch_first=True, has_bias=False)
        nn_rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=1, batch_first=True, has_bias=False)
        mindspore.load_param_into_net(static_rnn, nn_rnn.parameters_dict())
        inputs = Tensor(self.x, mindspore.float32)

        output0, h0 = static_rnn(inputs)
        output1, h1 = nn_rnn(inputs)

        assert np.allclose(output0.asnumpy(), output1.asnumpy(), 1e-3, 1e-3)
        assert np.allclose(h0.asnumpy(), h1.asnumpy(), 1e-3, 1e-3)

    def test_gru_multi_layer_precision(self):
        """test multilayer gru precision"""
        static_rnn = StaticGRU(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        nn_rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        mindspore.load_param_into_net(static_rnn, nn_rnn.parameters_dict())
        inputs = Tensor(self.x, mindspore.float32)
        static_rnn.set_train(False)
        nn_rnn.set_train(False)

        output0, h0 = static_rnn(inputs)
        output1, h1 = nn_rnn(inputs)
        assert np.allclose(output0.asnumpy(), output1.asnumpy(), 1e-3, 1e-3)
        assert np.allclose(h0.asnumpy(), h1.asnumpy(), 1e-3, 1e-3)

    def test_gru_bidirectional_precision(self):
        """test bidirectional gru precision"""
        static_rnn = StaticGRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        nn_rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        mindspore.load_param_into_net(static_rnn, nn_rnn.parameters_dict())
        static_rnn.set_train(False)
        nn_rnn.set_train(False)
        inputs = Tensor(self.x, mindspore.float32)
        output0, h0 = static_rnn(inputs)
        output1, h1 = nn_rnn(inputs)

        assert np.allclose(output0.asnumpy(), output1.asnumpy(), 1e-3, 1e-3)
        assert np.allclose(h0.asnumpy(), h1.asnumpy(), 1e-3, 1e-3)
