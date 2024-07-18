# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Tokenization classes for ChatYuan."""
from typing import List

import numpy as np
import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {
    "vocab_file": "spiece.model"
}

PRETRAINED_VOCAB_MAP = {
    "ChatYuan-large-v2": "https://openi.pcl.ac.cn/mindnlp/ChatYuan-large-v2/raw/branch/master/spiece.model"
}

class ChatYuanTokenizer(PreTrainedTokenizer):
    """Tokenizer for ChatYuan"""
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, **kwargs):
        """
        __init__
        
        Initializes a new instance of the ChatYuanTokenizer class.
        
        Args:
            vocab_file (str): The file path to the vocabulary file used for tokenization.
            **kwargs: Additional keyword arguments.
                return_token (bool, optional): A flag indicating whether to return the token. Defaults to False.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None.
        """
        super().__init__()

        return_token = kwargs.pop('return_token', False)
        self.return_token = return_token

        self.vocab_file = vocab_file
        self._tokenizer = self.get_spm_processor()

    def get_spm_processor(self):
        """Get SentencePieceProcessor Tokenizer."""
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(self.vocab_file)
        return tokenizer

    def execute_py(self, text_input):
        """Execute method."""
        return self.tokenize(text_input)

    def _execute_py(self, text_input):
        """Execute method."""
        return self._tokenize(text_input)

    def tokenize(self, text_input) -> List[str]:
        """
        This method tokenizes the input text using the ChatYuanTokenizer.
        
        Args:
            self (ChatYuanTokenizer): An instance of the ChatYuanTokenizer class.
            text_input (str): The input text to be tokenized.
        
        Returns:
            List[str]: A list of strings representing the tokens extracted from the input text.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        return self._execute_py(text_input)

    def _tokenize(self, text_input):
        """
        Returns a tokenized string.
        """
        text_input = self._convert_to_unicode(text_input)

        tokens = self._tokenizer.encode(text_input, out_type=str)
        if self.return_token:
            return tokens
        # return ids
        return np.array(self.convert_tokens_to_ids(tokens))

    def _convert_to_unicode(self, text_input):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text_input, str):
            return text_input
        if isinstance(text_input, bytes):
            return text_input.decode("utf-8", "ignore")
        if isinstance(text_input, np.ndarray):
            if text_input.dtype.type is np.bytes_:
                text_input = np.char.decode(text_input, "utf-8")
            return str(text_input)
        raise ValueError(f"Unsupported string type: {type(text_input)}, {text_input.dtype}")

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._tokenizer.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self._tokenizer.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self._tokenizer.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self._tokenizer.decode(current_sub_tokens)
        return out_string

__all__ = ['ChatYuanTokenizer']
