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
from mindspore import nn
from mindspore import context
from mindspore import Tensor

from mindnlp.modules import Seq2SeqEncoder, CNNEncoder


class TestRNNEncoder(unittest.TestCase):
    r"""
    Test module RNN Encoder
    """

    def test_rnn_encoder_graph(self):
        """
        Test rnn encoder module in graph mode
        """
        context.set_context(mode=context.GRAPH_MODE)

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        embedding = nn.Embedding(vocab_size, embedding_size)
        rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                     batch_first=True, dropout=dropout, bidirectional=bidirectional)

        rnn_encoder = Seq2SeqEncoder(embedding, rnn)

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

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        embedding = nn.Embedding(vocab_size, embedding_size)
        rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                     batch_first=True, dropout=dropout, bidirectional=bidirectional)

        rnn_encoder = Seq2SeqEncoder(embedding, rnn)

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

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        embedding = nn.Embedding(vocab_size, embedding_size)
        lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                     batch_first=True, dropout=dropout, bidirectional=bidirectional)

        lstm_encoder = Seq2SeqEncoder(embedding, lstm)

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

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        embedding = nn.Embedding(vocab_size, embedding_size)
        lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                     batch_first=True, dropout=dropout, bidirectional=bidirectional)

        lstm_encoder = Seq2SeqEncoder(embedding, lstm)

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

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        embedding = nn.Embedding(vocab_size, embedding_size)
        gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                     batch_first=True, dropout=dropout, bidirectional=bidirectional)

        gru_encoder = Seq2SeqEncoder(embedding, gru)

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

        vocab_size = 1000
        embedding_size = 32
        hidden_size = 16
        num_layers = 2
        has_bias = True
        dropout = 0.1
        bidirectional = False
        embedding = nn.Embedding(vocab_size, embedding_size)
        gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                     batch_first=True, dropout=dropout, bidirectional=bidirectional)

        gru_encoder = Seq2SeqEncoder(embedding, gru)

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

        vocab_size = 1000
        embedding_size = 32
        num_filter = 128
        ngram_filter_sizes = (2, 3, 4, 5)
        output_dim = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        convs = [
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(i, embedding_size),
                      pad_mode="pad") for i in ngram_filter_sizes
        ]

        cnn_encoder = CNNEncoder(embedding, convs, output_dim=output_dim)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)

        result = cnn_encoder(src_tokens)

        assert result.shape == (8, 16)

    def test_cnn_encoder_pynative(self):
        """
        Test cnn encoder module in pynative mode
        """
        context.set_context(mode=context.PYNATIVE_MODE)

        vocab_size = 1000
        embedding_size = 32
        num_filter = 128
        ngram_filter_sizes = (2, 3, 4, 5)
        output_dim = 16
        embedding = nn.Embedding(vocab_size, embedding_size)
        convs = [
            nn.Conv2d(in_channels=1,
                      out_channels=num_filter,
                      kernel_size=(i, embedding_size),
                      pad_mode="pad") for i in ngram_filter_sizes
        ]

        cnn_encoder = CNNEncoder(embedding, convs, output_dim=output_dim)

        src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)

        result = cnn_encoder(src_tokens)

        assert result.shape == (8, 16)
