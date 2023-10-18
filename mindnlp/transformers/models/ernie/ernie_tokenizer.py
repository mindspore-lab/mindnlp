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
Ernie Tokenizer
"""

import re
import numpy as np
from mindspore.dataset.text.transforms import Implementation
from tokenizers import Tokenizer
from mindnlp.configs import MINDNLP_TOKENIZER_CONFIG_URL_BASE
from .ernie_config import ERNIE_SUPPORT_LIST
from ...tokenization_utils import PreTrainedTokenizer

PRETRAINED_VOCAB_MAP = {
    model: MINDNLP_TOKENIZER_CONFIG_URL_BASE.format(
        re.search(r"^[^-]*", model).group(), model
    )
    for model in ERNIE_SUPPORT_LIST
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ernie-3.0-base-zh": 2048,
    "ernie-3.0-xbase-zh": 2048,
    "ernie-3.0-medium-zh": 2048,
    "ernie-3.0-mini-zh": 2048,
    "ernie-3.0-micro-zh": 2048,
    "ernie-3.0-nano-zh": 2048,
    "ernie-3.0-tiny-base-v1-zh": 2048,
    "ernie-3.0-tiny-medium-v1-zh": 2048,
    "ernie-3.0-tiny-mini-v1-zh": 2048,
    "ernie-3.0-tiny-micro-v1-zh": 2048,
    "ernie-3.0-tiny-nano-v1-zh": 2048,
    "uie-base": 2048,
    "uie-medium": 2048,
    "uie-mini": 2048,
    "uie-micro": 2048,
    "uie-nano": 2048,
    "uie-base-en": 512,
    "uie-senta-base": 2048,
    "uie-senta-medium": 2048,
    "uie-senta-mini": 2048,
    "uie-senta-micro": 2048,
    "uie-senta-nano": 2048,
    "uie-base-answer-extractor": 2048,
    "uie-base-qa-filter": 2048,
}


class ErnieTokenizer(PreTrainedTokenizer):
    """
    Tokenizer used for Ernie text process.

    Args:
        vocab (Vocab): Vocabulary used to look up words.
        return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.

    """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP

    def __init__(self, vocab: str, **kwargs):
        return_token = kwargs.pop("return_token", False)

        if isinstance(vocab, str):
            self._tokenizer = Tokenizer.from_file(vocab)
        else:
            raise ValueError(f"only support string, but got {vocab}")
        self.return_token = return_token
        self.implementation = Implementation.PY

        super().__init__(**kwargs)

    def __call__(self, text_input):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if isinstance(text_input, str):
            text_input = np.array(text_input)
        elif not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}."
            )
        return super().__call__(text_input)

    def execute_py(self, text_input):
        """
        Execute method.
        """
        return self._execute_py(text_input)

    def _execute_py(self, text_input):
        """
        Execute method.
        """
        text_input = self._convert_to_unicode(text_input)
        tokens = self._tokenizer.encode(text_input)
        if self.return_token is True:
            return np.array(tokens.tokens)
        return np.array(tokens.ids)

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
        raise ValueError(
            f"Unsupported string type: {type(text_input)}, {text_input.dtype}"
        )

    def _convert_token_to_id(self, token):
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

__all__ = ['ErnieTokenizer']
