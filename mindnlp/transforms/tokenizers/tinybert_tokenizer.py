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
TinyBert Tokenizer
"""
import os
from mindspore.dataset.text import Vocab as msVocab
from mindspore.dataset.text.transforms import Implementation
import numpy as np
from tokenizers.implementations import BertWordPieceTokenizer

from mindnlp.abc import PreTrainedTokenizer
from mindnlp.vocab import Vocab

PRETRAINED_VOCAB_MAP = {
    "tinybert_4L_zh": "https://download.mindspore.cn/toolkits/mindnlp/models/tinybert/tinybert_4L_zh/vocab.txt",
    "tinybert_6L_zh": "https://download.mindspore.cn/toolkits/mindnlp/models/tinybert/tinybert_6L_zh/vocab.txt"
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tinybert_4L_zh": 312,
    "tinybert_6L_zh": 768
}

class TinyBertTokenizer(PreTrainedTokenizer):
    """
    Tokenizer used for TinyBert text process.
    """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP

    def __init__(
        self,
        vocab=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        if isinstance(vocab, msVocab):
            vocab_dict = vocab.vocab()
        elif isinstance(vocab, Vocab):
            vocab_dict = vocab.vocab
        elif isinstance(vocab, str):
            if not os.path.isfile(vocab):
                raise ValueError(f"{vocab} is not a file.")
            if not vocab.endswith('.txt'):
                raise ValueError(f"{vocab} is not a txt file.")
            vocab_dict = msVocab.from_file(vocab).vocab()
        else:
            raise ValueError(
                f'only support Vocab class from mindspore or mindnlp, \
                    and a vocab.txt, but got {vocab}'
            )

        return_token = kwargs.pop('return_token', False)

        self._tokenizer = BertWordPieceTokenizer(vocab=vocab_dict, lowercase=do_lower_case)

        self.return_token = return_token
        self.implementation = Implementation.PY

    def save(self, save_path: str):
        """
        save tokenizer
        """
        # check save_path
        if not save_path.endswith('.txt'):
            raise ValueError(f"{save_path} is not a txt file.")

        vocab_dict = self._tokenizer.get_vocab()
        sorted_array = sorted(vocab_dict.items(), key=lambda item: item[1])

        with open(save_path, 'w', encoding='utf-8') as file:
            for token, _ in sorted_array:
                file.write(token+"\n")

    def __call__(self, text_input):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if isinstance(text_input, str):
            text_input = np.array(text_input)
        elif not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}.")
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
        return self._tokenizer.token_to_id(token)
