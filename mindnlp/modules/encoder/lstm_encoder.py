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
"""LSTM encoder modules"""
# pylint: disable=abstract-method

from mindspore import nn
from mindnlp.abc import EncoderBase

class LSTMEncoder(EncoderBase):
    r"""
    LSTM (Long Short-Term Memory) Encoder.

    There are two pipelines connecting two consecutive cells in a LSTM model; one is cell state pipeline
    and the other is hidden state pipeline. Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}` and an cell
    state :math:`c_{t-1}` of the layer at time :math:`{t-1}`, the cell state and hidden state at
    time :math:`t` is computed using an gating mechanism. Input gate :math:`i_t` is designed to protect the cell
    from perturbation by irrelevant inputs. Forget gate :math:`f_t` affords protection of the cell by forgetting
    some information in the past, which is stored in :math:`h_{t-1}`. Output gate :math:`o_t` protects other
    units from perturbation by currently irrelevant memory contents. Candidate cell state :math:`\tilde{c}_t` is
    calculated with the current input, on which the input gate will be applied. Finally, current cell state
    :math:`c_{t}` and hidden state :math:`h_{t}` are computed with the calculated gates and cell states. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ix}, b_{ix}` are the weight and bias used to transform from input :math:`x` to :math:`i`.
    Details can be found in paper `LONG SHORT-TERM MEMORY
    <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ and
    `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
    <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_.

    LSTM hides the cycle of the whole cyclic neural network on the time step of the sequence,
    and input the sequence and initial state to obtain the matrix spliced by
    the hidden state of each time step and the hidden state of the last time step.
    We use the hidden state of the last time step as the coding feature of the input sentence and
    output it to the next layer.

    .. math::
        h_{0:n},(h_{n}, c_{n}) = LSTM(x_{0:n},(h_{0},c_{0}))

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
        Tuple, a tuple contains (`output`, (`hiddens_n`, `cells_n`), `mask`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hx_n** (Tensor) -  A tuple of two Tensor (hiddens_n, cells_n)
          both of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **mask** (Tensor) - Mask Tensor used in decoder.

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from text.modules import LSTMEncoder
        >>> lstm_encoder = LSTMEncoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                            dropout=0.1, bidirectional=False)
        >>> src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> src_length = Tensor(np.ones([8]), mindspore.int32)
        >>> mask = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> output, (hiddens_n, cells_n), mask = lstm_encoder(src_tokens, src_length, mask=mask)
        >>> print(output.shape)
        >>> print(hiddens_n.shape)
        >>> print(cells_n.shape)
        >>> print(mask.shape)
        (8, 16, 16)
        (2, 8, 16)
        (2, 8, 16)
        (8, 16)
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1,
                 has_bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers,
                            has_bias=has_bias, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

    def construct(self, src_token, src_length=None, mask=None):
        if mask is not None:
            src_token = src_token * mask
        embed = self.embedding(src_token)

        output, (hiddens_n, cells_n) = self.lstm(embed, seq_length=src_length)
        return output, (hiddens_n, cells_n), mask

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to `new_order`."""
        encoder_output = encoder_out[0]
        encoder_hiddens, encoder_cells = encoder_out[1]
        encoder_padding_mask = encoder_out[2]

        new_output = encoder_output.gather(new_order, 1)
        new_hiddens = encoder_hiddens.gather(new_order, 1)
        new_cells = encoder_cells.gather(new_order, 1)
        new_padding_mask = encoder_padding_mask.gather(new_order, 0)

        return new_output, (new_hiddens, new_cells), new_padding_mask
