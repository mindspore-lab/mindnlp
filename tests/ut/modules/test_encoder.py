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
Test Encoder
"""

import unittest
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor

from mindnlp.modules import RNNEncoder, LSTMEncoder, GRUEncoder, CNNEncoder


class TestRNNEncoder(unittest.TestCase):
    r"""
    Test module RNN Encoder
    """

    def test_rnn_encoder_graph(self):
        """
        Test rnn encoder module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        rnn_encoder = RNNEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, hiddens_n, mask = rnn_encoder(src_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 16)
        assert hiddens_n.shape == (2, 8, 16)
        assert mask.shape == (8, 16)

    def test_rnn_encoder_pynative(self):
        """
        Test rnn encoder module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        rnn_encoder = RNNEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, hiddens_n, mask = rnn_encoder(src_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 16)
        assert hiddens_n.shape == (2, 8, 16)
        assert mask.shape == (8, 16)


class TestLSTMEncoder(unittest.TestCase):
    r"""
    Test module LSTM Encoder
    """

    def test_lstm_encoder_graph(self):
        """
        Test lstm encoder module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        lstm_encoder = LSTMEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, bidirectional=False)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, (hiddens_n, cells_n), mask = lstm_encoder(src_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 16)
        assert hiddens_n.shape == (2, 8, 16)
        assert cells_n.shape == (2, 8, 16)
        assert mask.shape == (8, 16)

    def test_lstm_encoder_pynative(self):
        """
        Test lstm encoder module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        lstm_encoder = LSTMEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, bidirectional=False)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, (hiddens_n, cells_n), mask = lstm_encoder(src_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 16)
        assert hiddens_n.shape == (2, 8, 16)
        assert cells_n.shape == (2, 8, 16)
        assert mask.shape == (8, 16)


class TestGRUEncoder(unittest.TestCase):
    r"""
    Test module GRU Encoder
    """

    def test_gru_encoder_graph(self):
        """
        Test gru encoder module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        gru_encoder = GRUEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, hiddens_n, mask = gru_encoder(src_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 16)
        assert hiddens_n.shape == (2, 8, 16)
        assert mask.shape == (8, 16)

    def test_gru_encoder_pynative(self):
        """
        Test gru encoder module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        gru_encoder = GRUEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, hiddens_n, mask = gru_encoder(src_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 16)
        assert hiddens_n.shape == (2, 8, 16)
        assert mask.shape == (8, 16)

class TestCNNEncoder(unittest.TestCase):
    r"""
    Test module CNN Encoder
    """

    def test_cnn_encoder_graph(self):
        """
        Test cnn encoder module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        cnn_encoder = CNNEncoder(emb_dim=128, num_filter=128, ngram_filter_sizes=(3,))

        input_dim = cnn_encoder.get_input_dim()
        output_dim = cnn_encoder.get_input_dim()

        assert input_dim == 128
        assert output_dim == 128

    def test_cnn_encoder_pynative(self):
        """
        Test cnn encoder module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        cnn_encoder = CNNEncoder(emb_dim=128, num_filter=128, ngram_filter_sizes=(3,))

        input_dim = cnn_encoder.get_input_dim()
        output_dim = cnn_encoder.get_input_dim()

        assert input_dim == 128
        assert output_dim == 128
