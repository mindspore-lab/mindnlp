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
"""RNN encoder modules"""
# pylint: disable=abstract-method

from mindspore import nn
from mindnlp.abc import EncoderBase

class RNNEncoder(EncoderBase):
    r"""
    RNN Encoder.

     Apply RNN layer with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to the input.

    For each element in the input sequence, each layer computes the following function:

    .. math::
        h_t = activation(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked LSTM . Default: 1.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
          LSTM layer except the last layer. Default 0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional LSTM,
          num_directions=2 if bidirectional=True otherwise 1. Default: False.

    Inputs:
        - **src_token** (Tensor) - Tokens in the source language with shape [batch, max_len].
        - **src_length** (Tensor) - Lengths of each sentence with shape [batch].
        - **mask** (Tensor) - Its elements identify whether the corresponding input token is padding or not.
          If the value is 1, not padding token. If the value is 0, padding token. Defaults to None.

    Outputs:
        Tuple, a tuple contains (`output`, `hiddens_n`, `mask`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hiddens_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **mask** (Tensor) - Mask Tensor used in decoder.

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from text.modules import RNNEncoder
        >>> rnn_encoder = RNNEncoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                          dropout=0.1, bidirectional=False)
        >>> src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> src_length = Tensor(np.ones([8]), mindspore.int32)
        >>> mask = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> output, hiddens_n, mask = rnn_encoder(src_tokens, src_length, mask=mask)
        >>> print(output.shape)
        >>> print(hiddens_n.shape)
        >>> print(mask.shape)
        (8, 16, 16)
        (2, 8, 16)
        (8, 16)
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1,
                 has_bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                          batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def construct(self, src_token, src_length=None, mask=None):
        if mask is None:
            src_token = src_token * mask
        embed = self.embedding(src_token)

        output, hiddens_n = self.rnn(embed, seq_length=src_length)
        return output, hiddens_n, mask

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to `new_order`."""
        encoder_output = encoder_out[0]
        encoder_hiddens = encoder_out[1]
        encoder_padding_mask = encoder_out[2]

        new_output = encoder_output.gather(new_order, 1)
        new_hiddens = encoder_hiddens.gather(new_order, 1)
        new_padding_mask = encoder_padding_mask.gather(new_order, 0)

        return new_output, new_hiddens, new_padding_mask
