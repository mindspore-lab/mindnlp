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
Test Decoder
"""

import unittest
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor

from mindnlp.modules import RNNDecoder, LSTMDecoder, GRUDecoder


class TestRNNDecoder(unittest.TestCase):
    r"""
    Test module RNN Decoder
    """

    def test_rnn_decoder_graph(self):
        """
        Test rnn decoder module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        rnn_decoder = RNNDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)

        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        encoder_output = Tensor(np.ones([8, 16, 16]), mindspore.int32)
        hiddens_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = rnn_decoder(tgt_tokens, (encoder_output, hiddens_n, mask))

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)

    def test_rnn_decoder_pynative(self):
        """
        Test rnn decoder module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        rnn_decoder = RNNDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)

        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        encoder_output = Tensor(np.ones([8, 16, 16]), mindspore.int32)
        hiddens_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = rnn_decoder(tgt_tokens, (encoder_output, hiddens_n, mask))

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)


class TestLSTMDecoder(unittest.TestCase):
    r"""
    Test module LSTM Decoder
    """

    def test_lstm_decoder_graph(self):
        """
        Test lstm decoder module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        lstm_decoder = LSTMDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, attention=True, encoder_output_units=16)

        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        encoder_output = Tensor(np.ones([8, 16, 16]), mindspore.int32)
        hiddens_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        cells_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        hx_n = (hiddens_n, cells_n)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = lstm_decoder(tgt_tokens, (encoder_output, hx_n, mask))

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)

    def test_lstm_decoder_pynative(self):
        """
        Test lstm decoder module in pynative mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        lstm_decoder = LSTMDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                   dropout=0.1, attention=True, encoder_output_units=16)

        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        encoder_output = Tensor(np.ones([8, 16, 16]), mindspore.int32)
        hiddens_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        cells_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        hx_n = (hiddens_n, cells_n)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = lstm_decoder(tgt_tokens, (encoder_output, hx_n, mask))

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)


class TestGRUDecoder(unittest.TestCase):
    r"""
    Test module GRU Decoder
    """

    def test_gru_decoder_graph(self):
        """
        Test gru module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        gru_decoder = GRUDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)

        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        encoder_output = Tensor(np.ones([8, 16, 16]), mindspore.int32)
        hiddens_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = gru_decoder(tgt_tokens, (encoder_output, hiddens_n, mask))

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)

    def test_gru_decoder_pynative(self):
        """
        Test gru module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        gru_decoder = GRUDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                                 dropout=0.1, attention=True, encoder_output_units=16)

        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        encoder_output = Tensor(np.ones([8, 16, 16]), mindspore.int32)
        hiddens_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output, attn_scores = gru_decoder(tgt_tokens, (encoder_output, hiddens_n, mask))

        assert output.shape == (8, 16, 1000)
        assert attn_scores.shape == (8, 16, 16)
