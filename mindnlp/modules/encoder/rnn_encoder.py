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

from mindnlp.abc import EncoderBase

class RNNEncoder(EncoderBase):
    r"""
    Stacked Elman RNN Encoder.

    Args:
        embedding (Cell): The embedding layer.
        rnn (Cell): The RNN Layer.

    Examples:
        >>> vocab_size = 1000
        >>> embedding_size = 32
        >>> hidden_size = 16
        >>> num_layers = 2
        >>> has_bias = True
        >>> dropout = 0.1
        >>> bidirectional = False
        >>> embedding = nn.Embedding(vocab_size, embedding_size)
        >>> rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, has_bias=has_bias,
        ...              batch_first=True, dropout=dropout, bidirectional=bidirectional)
        >>> rnn_encoder = RNNEncoder(embedding, rnn)
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

    def __init__(self, embedding, rnn):
        super().__init__(embedding)
        self.rnn = rnn

    def construct(self, src_token, src_length=None, mask=None):
        """
        Construct method.

        Args:
            src_token (Tensor): Tokens in the source language with shape [batch, max_len].
            src_length (Tensor): Lengths of each sentence with shape [batch].
            mask (Tensor): Its elements identify whether the corresponding input token is padding or not.
                If the value is 1, not padding token. If the value is 0, padding token. Defaults to None.

        Returns:
            Tuple, a tuple contains (`output`, `hiddens_n`, `mask`).

            - output (Tensor): Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
            - hiddens_n (Tensor): Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
            - mask (Tensor): Mask Tensor used in decoder.
        """
        if mask is None:
            mask = self._gen_mask(src_token)
        src_token = src_token * mask
        embed = self.embedding(src_token)

        output, hiddens_n = self.rnn(embed, seq_length=src_length)
        return output, hiddens_n, mask

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out (Union[Tensor, tuple]): The encoder's output.
            new_order (Tensor): Desired order.

        Returns:
            Tuple, encoder_out rearranged according to new_order.
        """
        encoder_output = encoder_out[0]
        encoder_hiddens = encoder_out[1]
        encoder_padding_mask = encoder_out[2]

        new_output = encoder_output.gather(new_order, 1)
        new_hiddens = encoder_hiddens.gather(new_order, 1)
        new_padding_mask = encoder_padding_mask.gather(new_order, 0)

        return new_output, new_hiddens, new_padding_mask
