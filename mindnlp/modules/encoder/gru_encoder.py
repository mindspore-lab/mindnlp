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
"""GRU encoder modules"""
# pylint: disable=abstract-method

from mindspore import nn
from mindnlp.abc import EncoderBase

class GRUEncoder(EncoderBase):
    r"""
    GRU Encoder.

    Apply GRU layer to the input.

    There are two gates in a GRU model; one is update gate and the other is reset gate.
    Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, a hidden state :math:`h_{t-1}`, the update and reset gate at
    time :math:`t` is computed using a gating mechanism. Update gate :math:`z_t` is designed to protect the cell
    from perturbation by irrelevant inputs and past hidden state. Reset gate :math:`r_t` determines how much
    information should be reset from old hidden state. New memory state :math:`{n}_t` is
    calculated with the current input, on which the reset gate will be applied. Finally, current hidden state
    :math:`h_{t}` is computed with the calculated update grate and new memory state. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

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
        >>> from text.modules import GRUEncoder
        >>> gru_encoder = GRUEncoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                          dropout=0.1, bidirectional=False)
        >>> src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> src_length = Tensor(np.ones([8]), mindspore.int32)
        >>> mask = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> output, hiddens_n, mask = gru_encoder(src_tokens, src_length, mask=mask)
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
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
                          batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def construct(self, src_token, src_length=None, mask=None):
        if mask is None:
            src_token = src_token * mask
        embed = self.embedding(src_token)

        output, hiddens_n = self.gru(embed, seq_length=src_length)
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
