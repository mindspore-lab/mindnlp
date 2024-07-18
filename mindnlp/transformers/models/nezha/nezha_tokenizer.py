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
"""
NezhaTokenizer
"""

import numpy as np
from mindspore.dataset.text.transforms import Implementation
from tokenizers.implementations import BertWordPieceTokenizer
from ...tokenization_utils import PreTrainedTokenizer


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "nezha-cn-base": 512,
    "nezha-cn-large": 512,
    "nezha-base-wwm": 512,
    "nezha-large-wwm": 512
}

class NezhaTokenizer(PreTrainedTokenizer):
    """
    Tokenizer used for Nezha text process.

    Args:
        vocab (Vocab): Vocabulary used to look up words.
        return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.
    """
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab: str, **kwargs):
        """
        Initialize the NezhaTokenizer class with the provided vocabulary.
        
        Args:
            self: The instance of the NezhaTokenizer class.
            vocab (str): The path to the vocabulary file or the vocabulary as a string.
                If a string is provided, the method initializes the tokenizer with the vocabulary.
                Otherwise, it raises a ValueError.
            **kwargs: Additional keyword arguments.
                return_token (bool): Optional. If True, the method will return the token. Default is False.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the 'vocab' parameter is not a string.
        """
        super().__init__()
        return_token = kwargs.pop('return_token', False)

        if isinstance(vocab, str):
            self._tokenizer = BertWordPieceTokenizer.from_file(vocab)
        else:
            raise ValueError(f'only support string, but got {vocab}')
        self.return_token = return_token
        self.implementation = Implementation.PY

    def execute_py(self, text_input):
        """
        Execute method.
        """
        return self._execute_py(text_input)

    def _execute_py(self, text_input):
        """
        Execute method.
        """
        text = self._convert_to_unicode(text_input)
        output = self._tokenizer.encode(text)
        if self.return_token is True:
            return np.array(output.tokens)
        return np.array(output.ids)

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
        """
        Converts a given token to its corresponding ID using the NezhaTokenizer.
        
        Args:
            self (NezhaTokenizer): The instance of the NezhaTokenizer class.
            token (str): The token to be converted into its corresponding ID.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the token provided is not a string.
            ValueError: If the token does not exist in the tokenizer's vocabulary.
        """
        return self._tokenizer.token_to_id(token)

__all__ = ['NezhaTokenizer']
