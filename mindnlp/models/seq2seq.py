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
RNN modules
"""
# pylint: disable=abstract-method

from mindnlp.abc import Seq2seqModel


class RNN(Seq2seqModel):
    r"""
    Stacked Elman RNN layers.

    Apply RNN layer with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to the input.

    For each element in the input sequence, each layer computes the following function:

    .. math::
        h_t = activation(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
      encoder (RNNEncoder): The RNN encoder.
      decoder (RNNDecoder):  The RNN decoder.

    Inputs:
        - **src_tokens** (Tensor) - Tokens of source sentences with shape [batch, src_len].
        - **tgt_tokens** (Tensor) - Tokens of targets with shape [batch, src_len].
        - **src_length** (Tensor) - Lengths of each source sentence with shape [batch].
        - **mask** (Tensor) - Its elements identify whether the corresponding input token is padding or not.
            If True, not padding token. If False, padding token. Defaults to None.

    Outputs:
        Tuple, a tuple contains (`output`, `attn_scores`).

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from text.modules import RNNEncoder, RNNDecoder
        >>> from text.models import RNN
        >>> rnn_encoder = RNNEncoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                          dropout=0.1, bidirectional=False)
        >>> rnn_decoder = RNNDecoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                          dropout=0.1, attention=True, encoder_output_units=16)
        >>> rnn = RNN(rnn_encoder, rnn_decoder)
        >>> src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> src_length = Tensor(np.ones([8]), mindspore.int32)
        >>> mask = Tensor(np.ones([8, 16], dtype=bool), mindspore.bool_)
        >>> output, attn_scores = rnn(src_tokens, tgt_tokens, src_length, mask=mask)
        >>> print(output.shape)
        >>> print(attn_scores.shape)
        (8, 16, 1000)
        (8, 16, 16)
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, src_tokens, tgt_tokens, src_length, mask=None):
        if mask is None:
            mask = self._gen_mask(src_tokens)

        encoder_out = self.encoder(src_tokens, src_length=src_length, mask=mask)
        decoder_out = self.decoder(tgt_tokens, encoder_out=encoder_out)
        return decoder_out


class LSTM(Seq2seqModel):
    r"""
    Stacked LSTM (Long Short-Term Memory) layers.

    Apply LSTM layer to the input.

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
      encoder (LSTMEncoder): The LSTM encoder.
      decoder (LSTMDecoder):  The LSTM decoder.

    Inputs:
        - **src_tokens** (Tensor) - Tokens of source sentences with shape [batch, src_len].
        - **tgt_tokens** (Tensor) - Tokens of targets with shape [batch, src_len].
        - **src_length** (Tensor) - Lengths of each source sentence with shape [batch].
        - **mask** (Tensor) - Its elements identify whether the corresponding input token is padding or not.
            If True, not padding token. If False, padding token. Defaults to None.

    Outputs:
        Tuple, a tuple contains (`output`, `attn_scores`).

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from text.modules import LSTMEncoder, LSTMDecoder
        >>> from text.models import LSTM
        >>> lstm_encoder = LSTMEncoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                            dropout=0.1, bidirectional=False)
        >>> lstm_decoder = LSTMDecoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                            dropout=0.1, attention=True, encoder_output_units=16)
        >>> lstm = LSTM(lstm_encoder, lstm_decoder)
        >>> src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> src_length = Tensor(np.ones([8]), mindspore.int32)
        >>> mask = Tensor(np.ones([8, 16], dtype=bool), mindspore.bool_)
        >>> output, attn_scores = lstm(src_tokens, tgt_tokens, src_length, mask=mask)
        >>> print(output.shape)
        >>> print(attn_scores.shape)
        (8, 16, 1000)
        (8, 16, 16)
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, src_tokens, tgt_tokens, src_length, mask=None):
        if mask is None:
            mask = self._gen_mask(src_tokens)

        encoder_out = self.encoder(src_tokens, src_length=src_length, mask=mask)
        decoder_out = self.decoder(tgt_tokens, encoder_out=encoder_out)
        return decoder_out


class GRU(Seq2seqModel):
    r"""
    Stacked GRU (Gated Recurrent Unit) layers.

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
      encoder (GRUEncoder): The GRU encoder.
      decoder (GRUDecoder):  The GRU decoder.

    Inputs:
        - **src_tokens** (Tensor) - Tokens of source sentences with shape [batch, src_len].
        - **tgt_tokens** (Tensor) - Tokens of targets with shape [batch, src_len].
        - **src_length** (Tensor) - Lengths of each source sentence with shape [batch].
        - **mask** (Tensor) - Its elements identify whether the corresponding input token is padding or not.
            If True, not padding token. If False, padding token. Defaults to None.

    Outputs:
        Tuple, a tuple contains (`output`, `attn_scores`).

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from text.modules import GRUEncoder, GRUDecoder
        >>> from text.models import GRU
        >>> gru_encoder = GRUEncoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                          dropout=0.1, bidirectional=False)
        >>> gru_decoder = GRUDecoder(1000, 32, 16, num_layers=2, has_bias=True,
        ...                          dropout=0.1, attention=True, encoder_output_units=16)
        >>> gru = RNN(gru_encoder, gru_decoder)
        >>> src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> src_length = Tensor(np.ones([8]), mindspore.int32)
        >>> mask = Tensor(np.ones([8, 16], dtype=bool), mindspore.bool_)
        >>> output, attn_scores = gru(src_tokens, tgt_tokens, src_length, mask=mask)
        >>> print(output.shape)
        >>> print(attn_scores.shape)
        (8, 16, 1000)
        (8, 16, 16)
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, src_tokens, tgt_tokens, src_length, mask=None):
        if mask is None:
            mask = self._gen_mask(src_tokens)

        encoder_out = self.encoder(src_tokens, src_length=src_length, mask=mask)
        decoder_out = self.decoder(tgt_tokens, encoder_out=encoder_out)
        return decoder_out
