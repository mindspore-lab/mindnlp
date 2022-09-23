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
"""Test RNN"""

import unittest
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor

from mindnlp.modules import RNNEncoder, RNNDecoder, LSTMEncoder, LSTMDecoder, GRUEncoder, GRUDecoder
from mindnlp.models import RNN, LSTM, GRU


class TestRNN(unittest.TestCase):
    r"""
    Test module RNN
    """

    def test_rnn_graph(self):
        """
        Test rnn module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        rnn_encoder = RNNEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)
        rnn_decoder = RNNDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)
        rnn = RNN(rnn_encoder, rnn_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = rnn(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)

    def test_rnn_pynative(self):
        """
        Test rnn module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        rnn_encoder = RNNEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)
        rnn_decoder = RNNDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)
        rnn = RNN(rnn_encoder, rnn_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = rnn(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)


class TestLSTM(unittest.TestCase):
    r"""
    Test module LSTM
    """

    def test_lstm_graph(self):
        """
        Test lstm module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        lstm_encoder = LSTMEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, bidirectional=False)
        lstm_decoder = LSTMDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, attention=True, encoder_output_units=16)
        lstm = LSTM(lstm_encoder, lstm_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = lstm(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)

    def test_lstm_pynative(self):
        """
        Test lstm module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        lstm_encoder = LSTMEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, bidirectional=False)
        lstm_decoder = LSTMDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, attention=True, encoder_output_units=16)
        lstm = GRU(lstm_encoder, lstm_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = lstm(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)


class TestGRU(unittest.TestCase):
    r"""
    Test module GRU
    """

    def test_gru_graph(self):
        """
        Test gru module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        gru_encoder = GRUEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)
        gru_decoder = GRUDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)
        gru = RNN(gru_encoder, gru_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = gru(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)

    def test_gru_pynative(self):
        """
        Test gru module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        gru_encoder = GRUEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, bidirectional=False)
        gru_decoder = GRUDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)
        gru = RNN(gru_encoder, gru_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = gru(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)
