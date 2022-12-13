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
RNN Decoder modules
"""
# pylint: disable=abstract-method

from mindspore import nn
from mindspore import ops
import mindspore.numpy as mnp
from mindnlp.abc import DecoderBase

class RNNDecoder(DecoderBase):
    r"""
    Stacked Elman RNN Decoder.

    Args:
        embedding (Cell): The embedding layer.
        rnns (list): The list of RNN cells.
        dropout_in (Union[float, int]): If not 0, append `Dropout` layer on the inputs of each
            RNN layer. Default 0. The range of dropout is [0.0, 1.0).
        dropout_out (Union[float, int]): If not 0, append `Dropout` layer on the outputs of each
            RNN layer except the last layer. Default 0. The range of dropout is [0.0, 1.0).
        attention (bool): Whether to use attention. Default: True.
        encoder_output_units (int): Number of features of encoder output. Default: 512.

    Examples:
        >>> vocab_size = 1000
        >>> embedding_size = 32
        >>> hidden_size = 16
        >>> num_layers = 2
        >>> dropout_in = 0.1
        >>> dropout_out = 0.1
        >>> encoder_output_units = 16
        >>> embedding = nn.Embedding(vocab_size, embedding_size)
        >>> input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        >>> rnns = [
        ...     nn.RNNCell(
        ...         input_size=embedding_size + input_feed_size
        ...         if layer == 0
        ...             else hidden_size,
        ...         hidden_size=hidden_size
        ...         )
        ...         for layer in range(num_layers)
        ... ]
        >>> rnn_decoder = RNNDecoder(embedding, rnns, dropout_in=dropout_in, dropout_out=dropout_out,
        ...                          attention=True, encoder_output_units=encoder_output_units, mode="RNN")
        >>> tgt_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> encoder_output = Tensor(np.ones([8, 16, 16]), mindspore.float32)
        >>> hiddens_n = Tensor(np.ones([2, 8, 16]), mindspore.float32)
        >>> mask = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> output, attn_scores = rnn_decoder(tgt_tokens, (encoder_output, hiddens_n, mask))
        >>> print(output.shape)
        >>> print(attn_scores.shape)
        (8, 16, 1000)
        (8, 16, 16)
    """

    def __init__(self, embedding, rnns, dropout_in=0, dropout_out=0, attention=True,
                 encoder_output_units=512, mode="RNN"):
        super().__init__(embedding)
        self.dropout_in_module = nn.Dropout(1 - dropout_in)
        self.dropout_out_module = nn.Dropout(1 - dropout_out)
        self.layers = nn.CellList(rnns)
        self.num_layers = len(rnns)
        self.hidden_size = rnns[0].hidden_size
        self.vocab_size = self.embedding.vocab_size
        self.is_lstm = mode == "LSTM"

        self.attention = attention
        if attention:
            self.input_proj = nn.Dense(self.hidden_size, encoder_output_units, has_bias=False)
            self.output_proj = nn.Dense(self.hidden_size + encoder_output_units, self.hidden_size, has_bias=False)
            self.softmax = nn.Softmax(axis=1)
            self.tanh = nn.Tanh()

        self.fc_out = nn.Dense(self.hidden_size, self.vocab_size)

    def construct(self, prev_output_tokens, encoder_out=None):
        """
        Construct method.

        Args:
            prev_output_tokens (Tensor): Output tokens for teacher forcing with shape [batch, tgt_len].
            encoder_out (Tensor): Output of encoder. Default: None.

        Returns:
            Tuple, a tuple contains (`output`, `attn_scores`).

            - output (Tensor): Tensor of shape (batch, `tgt_len`, `vocab_size`).
            - attn_scores (Tensor): Tensor of shape (batch, `tgt_len`, `src_len`)
              if attention=True otherwise None.
        """
        output, attn_scores = self.extract_features(prev_output_tokens, encoder_out)
        output = self.output_layer(output)
        return output, attn_scores

    def _attention_layer(self, hidden, encoder_output, mask):
        """
        Attention method.
        """
        # hidden: [batch, hidden_size]
        # encoder_output: [batch, src_len, encoder_output_units]
        # mask: [batch, src_len]
        query = self.input_proj(hidden)  # [batch, encoder_output_units]

        # compute attention
        attn_scores = (query * encoder_output.transpose((1, 0, 2))).sum(axis=2)  # [src_len, batch]
        attn_scores = attn_scores.transpose((1, 0))  # [batch, src_len]

        # don't attend over padding
        if mask is not None:
            attn_scores = ops.masked_fill(attn_scores, mask == 0, float("-inf"))

        attn_scores = self.softmax(attn_scores)  # [batch, src_len]

        # sum weighted sources
        output = (attn_scores.expand_dims(axis=2) * encoder_output).sum(axis=1)  # [batch, encoder_output_units]
        output = self.tanh(self.output_proj(ops.concat((output, hidden), axis=1)))  # [batch, hidden_size]

        return output, attn_scores

    def extract_features(self, prev_output_tokens, encoder_out=None):
        """
        Extract features of encoder output.

        Args:
            prev_output_tokens (Tensor): Output tokens for teacher forcing with shape [batch, tgt_len].
            encoder_out (Tensor): Output of encoder. Default: None.

        Returns:
            Tuple, a tuple contains (`output`, `attn_scores`).

            - output (Tensor): The extracted feature Tensor of shape (batch, `tgt_len`, `hidden_size`).
            - attn_scores (Tensor): Tensor of shape (batch, `tgt_len`, `src_len`)
              if attention=True otherwise None.
        """
        batch_size, tgt_len = prev_output_tokens.shape

        # embed the target tokens
        embed_token = self.embedding(prev_output_tokens)  # [batch, tgt_len, embedding_size]
        embed_token = self.dropout_in_module(embed_token)

        # get output from encoder
        if encoder_out is not None:
            encoder_output = encoder_out[0]  # [batch_size, src_len, num_directions * hidden_size]
            encoder_hiddens = encoder_out[1]  # [num_directions * num_layers, batch_size, hidden_size]
            encoder_padding_mask = encoder_out[2]  # [batch, src_len]

            if self.is_lstm:
                prev_hiddens = [encoder_hiddens[0][i] for i in range(self.num_layers)]
                prev_cells = [encoder_hiddens[1][i] for i in range(self.num_layers)]
            else:
                prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
                prev_cells = None
            input_feed = ops.zeros((batch_size, self.hidden_size), embed_token.dtype)
        else:
            encoder_output = mnp.empty(0)
            encoder_hiddens = mnp.empty(0)
            encoder_padding_mask = mnp.empty(0)

            zero_state = ops.zeros((batch_size, self.hidden_size), embed_token.dtype)
            prev_hiddens = [zero_state for _ in range(self.num_layers)]
            prev_cells = [zero_state for _ in range(self.num_layers)]
            input_feed = None

        outs = []
        attns = []
        for j in range(tgt_len):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                # [batch, embedding_size + hidden_size]
                input_rnn = ops.concat((embed_token[:, j, :], input_feed), axis=1)
            else:
                input_rnn = embed_token[:, j, :]  # [batch, embedding_size]

            hidden = None
            cell = None
            for i , rnn in enumerate(self.layers):
                # recurrent cell
                if self.is_lstm:
                    hidden, cell = rnn(input_rnn, (prev_hiddens[i], prev_cells[i]))  # [batch, hidden_size]

                    # hidden state becomes the input to the next layer
                    input_rnn = self.dropout_out_module(hidden)

                    # save state for next time step
                    prev_hiddens[i] = hidden
                    prev_cells[i] = cell
                else:
                    hidden = rnn(input_rnn, prev_hiddens[i])

                    # hidden state becomes the input to the next layer
                    input_rnn = self.dropout_out_module(hidden)

                    # save state for next time step
                    prev_hiddens[i] = hidden

            # apply attention using the last layer's hidden state
            if self.attention:
                out, attn = self._attention_layer(hidden, encoder_output, encoder_padding_mask)
            else:
                out = hidden
                attn = None
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)
            attns.append(attn)

        output = ops.stack(outs, 1)
        if self.attention:
            attn_scores = ops.stack(attns, 1)
        else:
            attn_scores = None

        return output, attn_scores

    def output_layer(self, features):
        """
        Project features to the vocabulary size.

        Args:
            features (Tensor): The extracted feature Tensor.

        Returns:
            Tensor, the output of decoder.
        """
        output = self.fc_out(features)  # [batch, tgt_len, vocab_size]
        return output
