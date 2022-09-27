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

    Details can be found in paper
    `Relation classification via convolutional deep neural network
    <https://aclanthology.org/C14-1220.pdf>`

    Args:
        emb_dim (int): The dimension of each vector in the input sequence.
        num_filter (int): The output dim for each convolutional layer.
        ngram_filter_sizes (Tuple[int]): Max length of sentence.
        conv_layer_activation (Module): Activation to use after the convolution layers.
        output_dim (int): The output vector of collected features after doing convolutions and pooling.
            If this value is `None`, return the result of the max pooling, an output of shape.

    Inputs:
        - **src_token** (Tensor) - Tokens in the source language with shape [batch, max_len].
        - **mask** (Tensor) - Its elements identify whether the corresponding input token is padding or not.
          If the value is 1, not padding token. If the value is 0, padding token. Defaults to None.

    Outputs:
        Tuple, a tuple contains (`output`, `hiddens_n`, `mask`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hiddens_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **mask** (Tensor) - Mask Tensor used in decoder.

    Examples:
        >>> import mindspore
        >>> from mindtext.modules import CNNEncoder
        >>> encoder = CNNEncoder(emb_dim = 128, num_filter = 128, ngram_filter_sizes = (3,))
        >>> print(encoder.get_input_axis())
        >>> print(encoder.get_output_axis())
        128 128
    """

    def __init__(self,
                 emb_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.Tanh(),
                 output_dim=None):
        """"Init"""
        super().__init__()
        self.emb_axis = emb_dim
        self.num_filter = num_filter
        self.ngram_filter_sizes = ngram_filter_sizes
        self.act = conv_layer_activation
        self.output_axis = output_dim

        self.convs = [
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_filter,
                      kernel_size=(i, self.emb_axis)) for i in self.ngram_filter_sizes
        ]
        self.pool = nn.AdaptiveMaxPool1d(output_size=3)

        maxpool_output_axis = self.num_filter * len(self.ngram_filter_sizes)

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
        """Construct"""
        if mask:
            src_token = src_token * mask

        src_token = ops.expand_dims(src_token, 1)
        convs_out = [self.act(conv(src_token)).squeeze(3) for conv in self.convs]

        maxpool_out = [
            (self.pool(t)).squeeze(axis=2) for t in convs_out
        ]
        result = ops.concat(maxpool_out, axis=1)

        if self.projection_layer:
            result = self.projection_layer(result)
        return result
