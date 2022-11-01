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
"""Encoder basic model"""

from mindspore import nn
from mindspore import ops


class EncoderBase(nn.Cell):
    r"""
    Basic class for encoders

    Args:
        embedding (Cell): The embedding layer.
    """

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def construct(self, src_token, src_length=None):
        """
        Construct method.

        Args:
            src_token (Tensor): Tokens in the source language with shape [batch, max_len].
            src_length (Tensor): Lengths of each sentence with shape [batch].
            mask (Tensor): Its elements identify whether the corresponding input token is padding or not.
                If True, not padding token. If False, padding token. Defaults to None.
        """
        raise NotImplementedError("Model must implement the construct method")

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out (Union[Tensor, tuple]): The encoder's output.
            new_order (Tensor): Desired order.
        """
        raise NotImplementedError

    def reset_parameters(self, mask=None):
        """
        Reset model's parameters

        Args:
            mask (Tensor): Its elements identify whether the corresponding input token is padding or not.
                If True, not padding token. If False, padding token. Defaults to None.
        """
        raise NotImplementedError

    def _gen_mask(self, inputs):
        """Generate mask tensor"""
        return ops.ones_like(inputs)
