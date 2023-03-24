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
"""CNN encoder modules"""
# pylint: disable=abstract-method

from mindspore import nn
from mindspore import ops
from mindnlp.abc import EncoderBase

class CNNEncoder(EncoderBase):
    r"""
    CNN Encoder.

    Convolutional encoder consisting of `len(convolutions)` layers.

    Args:
        embedding (Cell): The embedding layer.
        convs (list[Cell]): The list of Conv Cell.
        conv_layer_activation (Module): Activation to use after the convolution layers.
        output_dim (int): The output vector of collected features after doing convolutions and pooling.
            If this value is `None`, return the result of the max pooling, an output of shape.

    Raises:
        TypeError: if `embedding` is not a Cell.
        TypeErrpr: if `convs` is not a list[Cell].
        TypeError: if `conv_layer_activation` is not a Module.
        TypeError: if `output_dim` is not a int.
        RuntimeError: If `embedding` is None.
        RuntimeError: If `convs` is None.

    Examples:
        >>> vocab_size = 1000
        >>> embedding_size = 32
        >>> num_filter = 128
        >>> ngram_filter_sizes = (2, 3, 4, 5)
        >>> output_dim = 16
        >>> embedding = nn.Embedding(vocab_size, embedding_size)
        >>> convs = [
        ...     nn.Conv2d(in_channels=1,
        ...               out_channels=num_filter,
        ...               kernel_size=(i, embedding_size),
        ...               pad_mode="pad") for i in ngram_filter_sizes
        ... ]
        >>> cnn_encoder = CNNEncoder(embedding, convs, output_dim=output_dim)
        >>> src_tokens = Tensor(np.ones([8, 16]), mindspore.int32)
        >>> result = cnn_encoder(src_tokens)
        >>> print(result.shape)
        (8, 16)

    """

    def __init__(self, embedding, convs, conv_layer_activation=nn.Tanh(), output_dim=None):
        super().__init__(embedding)
        self.emb_axis = self.embedding.embedding_size
        self.act = conv_layer_activation
        self.output_axis = output_dim
        self.num_filter = convs[0].out_channels
        self.ngram_filter_sizes = len(convs)

        self.convs = nn.CellList(convs)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        maxpool_output_axis = self.num_filter * self.ngram_filter_sizes

        if self.output_axis:
            self.projection_layer = nn.Dense(maxpool_output_axis, self.output_axis)
        else:
            self.projection_layer = None
            self.output_axis = maxpool_output_axis

    def get_input_dim(self):
        """Returns the dimension of input vector"""
        return self.emb_axis

    def get_output_dim(self):
        """Returns the dimension of the output vector"""
        return self.output_axis

    def construct(self, src_token, src_length=None, mask=None):
        """Construct function of CNNEncoder.

        Args:
            src_token (Tensor): Tokens in the source language with shape [batch, max_len].
            mask (Tensor): Its elements identify whether the corresponding input token is padding or not.
                If the value is 1, not padding token. If the value is 0, padding token. Defaults to None.

        Returns:
            Tensor, If output_dim is None, the result shape is of `(batch_size, len(convs) * num_filter)`
            and dtype is `float`; If not, the result shape is of `(batch_size, output_dim)`.
        """
        if mask is None:
            mask = self._gen_mask(src_token)
        src_token = src_token * mask
        embed = self.embedding(src_token)

        embed = ops.expand_dims(embed, 1)
        convs_out = ()
        for conv in self.convs:
            convs_out += (self.act(conv(embed)).squeeze(3),)

        maxpool_out = ()
        for out in convs_out:
            maxpool_out += (self.pool(out).squeeze(axis=2),)
        result = ops.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)
        return result
