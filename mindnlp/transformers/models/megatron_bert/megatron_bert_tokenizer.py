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
MegatronBertTokenizer
"""

import os
import numpy as np
from mindspore.dataset.text.transforms import Implementation
from mindspore.dataset.text import Vocab as msVocab
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from mindnlp.vocab import Vocab
from ...tokenization_utils import PreTrainedTokenizer


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "nvidia/Erlangshen-MegatronBert-1.3B-NLI": 512,
    "KBLab/megatron-bert-large-swedish-cased-165-zero-shot": 512,
    "nvidia/megatron-bert-uncased-345m": 512,
    "nvidia/megatron-bert-cased-345m": 512,
}


class MegatronBertTokenizer(PreTrainedTokenizer):
    """
        Tokenizer used for MegatronBert text process.
        Args:
            vocab (Vocab): Vocabulary used to look up words.
            return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.
        Examples:
            >>> from mindspore.dataset import text
            >>> from mindnlp.transforms import MegatronBertTokenizer
            >>> text = "Believing that faith can triumph over everything is in itself the greatest belief"
            >>> tokenizer = MegatronBertTokenizer.from_pretrained('nvidia/megatron-bert-cased-345m')
            >>> tokens = tokenizer.encode(text)
    """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

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
            self.tokenizer = Tokenizer.from_file(vocab)
        else:
            raise ValueError(f'only support Vocab class from mindspore or mindnlp, but got {vocab}')

        return_token = kwargs.pop('return_token', False)

        if isinstance(vocab, str):
            self._tokenizer = Tokenizer.from_file(vocab)
        else:
            self._tokenizer = BertWordPieceTokenizer(vocab=vocab_dict, lowercase=do_lower_case)

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
        text_input = self._convert_to_unicode(text_input)
        tokens = self.tokenizer.encode(text_input)
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
        raise ValueError(f"Unsupported string type: {type(text_input)}, {text_input.dtype}")

    def _convert_token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        return self._tokenizer.id_to_token(index)


__all__ = ['MegatronBertTokenizer']
