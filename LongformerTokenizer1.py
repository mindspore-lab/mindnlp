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
T5Tokenizer
"""

import os
import numpy as np
from tokenizers import Tokenizer, models
from mindspore.dataset.transforms.transforms import PyTensorOperation
from mindspore.dataset.text.transforms import Implementation
from mindnlp.utils.download import cache_file
from mindnlp.configs import DEFAULT_ROOT

URL = {
        "allenai/longformer-base-4096": (
            "https://huggingface.co/allenai/longformer-base-4096/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-large-4096": (
            "https://huggingface.co/allenai/longformer-large-4096/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-large-4096-finetuned-triviaqa": (
            "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-base-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-large-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/tokenizer.json"
        ),
}

class LongformerTokenizer(PyTensorOperation):
    """
    Tokenizer used for Longformer text process.
    Args:
        tokenizer_file (Str): The path of the tokenizer.json 
    Examples:
        >>> from mindspore.dataset import text
        >>> from mindnlp.transforms import LongformerTokenizer
        >>> text = "Believing that faith can triumph over everything is in itself the greatest belief"
        >>> tokenizer = LongformerTokenizer.from_pretrained('Longformer-base')
        >>> tokens = tokenizer.encode(text)
    """
    def __init__(
        self,
        tokenizer_file=None,
    ):
        super().__init__()
        if tokenizer_file is not None:
            self._tokenizer = Tokenizer.from_file(tokenizer_file)
        self.implementation = Implementation.PY

    def __call__(self, text_input):
        if isinstance(text_input, str):
            text_input = np.array(text_input)
        elif not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}.")
        return super().__call__(text_input)

    @classmethod
    def from_pretrained(cls, size:str):
        """load T5Tokenizer from pretrained tokenizer.json"""
        cache_dir = os.path.join(DEFAULT_ROOT, "tokenizers", size)
        path, _ = cache_file(None, url=URL[size], cache_dir=cache_dir)
        tokenizer = cls(tokenizer_file=str(path))
        return tokenizer

    def encode(self, text_input):
        """encode function"""
        tokens = self._tokenizer.encode(text_input)
        return tokens

    def decode(self, ids: list):
        """decode function"""
        return self._tokenizer.decode(ids)

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
        return tokens.ids

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
