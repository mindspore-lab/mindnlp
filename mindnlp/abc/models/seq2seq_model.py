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
"""Sequence-to-sequence basic model"""
# pylint: disable=abstract-method
# pylint: disable=arguments-differ
from .base_model import BaseModel


class Seq2seqModel(BaseModel):
    r"""
    Basic class for seq2seq models

    Args:
        encoder (EncoderBase): The encoder.
        decoder (DecoderBase): The decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, src_tokens, tgt_tokens, src_length, mask=None):
        """
        Construct method.

        Args:
            src_tokens (Tensor): Tokens of source sentences with shape [batch, src_len].
            tgt_tokens (Tensor): Tokens of targets with shape [batch, src_len].
            src_length (Tensor): Lengths of each source sentence with shape [batch].
            mask (Tensor): Its elements identify whether the corresponding input token is padding or not.
                If True, not padding token. If False, padding token. Defaults to None.

        Returns:
            Tensor, The result vector of seq2seq model with shape [batch, max_len, vocab_size].
        """
        encoder_out = self.encoder(src_tokens, src_length=src_length, mask=mask)

        decoder_out = self.decoder(tgt_tokens, encoder_out=encoder_out)
        return decoder_out

    def get_context(self, src_tokens, mask=None):
        """
        Get Context from encoder.

        Args:
            src_tokens (Tensor): Tokens of source sentences with shape [batch, src_len].
            mask (Tensor): Its elements identify whether the corresponding input token is padding or not.
                If True, not padding token. If False, padding token. Defaults to None.

        Returns:
            Union[Tensor, tuple], the output of encoder.
        """
        return self.encoder(src_tokens, mask=mask)

    def extract_features(self, src_tokens, tgt_tokens, src_length):
        """
        Extract features of encoder output.

        Args:
            src_tokens (Tensor): Tokens of source sentences with shape [batch, src_len].
            tgt_tokens (Tensor): Tokens of targets with shape [batch, src_len].
            src_length (Tensor): Lengths of each source sentence with shape [batch].

        Returns:
            Tensor, the extracted features.
        """
        encoder_out = self.encoder(src_tokens, src_length=src_length)
        features = self.decoder.extract_features(tgt_tokens, encoder_out=encoder_out)
        return features

    def output_layer(self, features):
        """
        Project features to the default output size.

        Args:
            features (Tensor): The extracted features.

        Returns:
            Tensor, the output of decoder.
        """
        return self.decoder.output_layer(features)
