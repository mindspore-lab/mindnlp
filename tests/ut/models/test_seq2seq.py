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
from mindspore import nn
from mindspore import context
from mindspore import Tensor

from mindnlp.modules import Seq2SeqEncoder, Seq2SeqDecoder
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

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        encoder_output_units = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        rnn_layer = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)

        rnn_encoder = Seq2SeqEncoder(embedding, rnn_layer)
        rnn_decoder = Seq2SeqDecoder(embedding, rnn_layer, dropout=dropout, attention=True,
                                     encoder_output_units=encoder_output_units)
        rnn = RNN(rnn_encoder, rnn_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output = rnn(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 1000, 16)

    def test_rnn_pynative(self):
        """
        Test rnn module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        encoder_output_units = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        rnn_layer = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)

        rnn_encoder = Seq2SeqEncoder(embedding, rnn_layer)
        rnn_decoder = Seq2SeqDecoder(embedding, rnn_layer, dropout=dropout, attention=True,
                                     encoder_output_units=encoder_output_units)
        rnn = RNN(rnn_encoder, rnn_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output = rnn(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 1000, 16)


class TestLSTM(unittest.TestCase):
    r"""
    Test module LSTM
    """

    def test_lstm_graph(self):
        """
        Test lstm module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        encoder_output_units = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        lstm_layer = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                             batch_first=True, dropout=dropout, bidirectional=bidirectional)

        lstm_encoder = Seq2SeqEncoder(embedding, lstm_layer)
        lstm_decoder = Seq2SeqDecoder(embedding, lstm_layer, dropout=dropout, attention=True,
                                      encoder_output_units=encoder_output_units)
        lstm = LSTM(lstm_encoder, lstm_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output = lstm(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 1000, 16)

    def test_lstm_pynative(self):
        """
        Test lstm module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        encoder_output_units = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        lstm_layer = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                             batch_first=True, dropout=dropout, bidirectional=bidirectional)

        lstm_encoder = Seq2SeqEncoder(embedding, lstm_layer)
        lstm_decoder = Seq2SeqDecoder(embedding, lstm_layer, dropout=dropout, attention=True,
                                      encoder_output_units=encoder_output_units)
        lstm = GRU(lstm_encoder, lstm_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output = lstm(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 1000, 16)


class TestGRU(unittest.TestCase):
    r"""
    Test module GRU
    """

    def test_gru_graph(self):
        """
        Test gru module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        encoder_output_units = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        gru_layer = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)

        gru_encoder = Seq2SeqEncoder(embedding, gru_layer)
        gru_decoder = Seq2SeqDecoder(embedding, gru_layer, dropout=dropout, attention=True,
                                     encoder_output_units=encoder_output_units)
        gru = GRU(gru_encoder, gru_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output = gru(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 1000, 16)

    def test_gru_pynative(self):
        """
        Test gru module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        encoder_output_units = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        gru_layer = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)

        gru_encoder = Seq2SeqEncoder(embedding, gru_layer)
        gru_decoder = Seq2SeqDecoder(embedding, gru_layer, dropout=dropout, attention=True,
                                     encoder_output_units=encoder_output_units)
        gru = GRU(gru_encoder, gru_decoder)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        src_length = Tensor(np.ones([8]), mindspore.int32)
        mask = Tensor(np.ones([8, 16]), mindspore.int32)

        output = gru(src_tokens, tgt_tokens, src_length, mask=mask)

        assert output.shape == (8, 1000, 16)
