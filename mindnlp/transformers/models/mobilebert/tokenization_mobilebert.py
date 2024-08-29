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
"""
MobileBertTokenizer
"""
import os
import numpy as np
from mindspore.dataset.text.transforms import Implementation
from mindspore.dataset.text import Vocab as msVocab
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from mindnlp.vocab import Vocab
from ...tokenization_utils import PreTrainedTokenizer


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mobilebert-uncased": 512}


class MobileBertTokenizer(PreTrainedTokenizer):
    """
    Tokenizer used for MobileBert text process.

    Args:
        vocab (Vocab): Vocabulary used to look up words.
        lower_case (bool, optional): Whether to perform lowercase processing on the text. If True, will fold the
            text to lower case. Default: True.
        return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.

    Raises:
        TypeError: If `lower_case` is not of type bool.
        TypeError: If `py_transform` is not of type bool.
        RuntimeError: If dtype of input Tensor is not str.

    Example:
        ```python
        >>> from mindspore.dataset import text
        >>> from mindnlp.transforms import MobileBertTokenizer
        >>> vocab_list = ["床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低",
              "思", "故", "乡","繁", "體", "字", "嘿", "哈", "大", "笑", "嘻", "i", "am", "mak",
              "make", "small", "mistake", "##s", "during", "work", "##ing", "hour", "😀", "😃",
              "😄", "😁", "+", "/", "-", "=", "12", "28", "40", "16", " ", "I", "[CLS]", "[SEP]",
              "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"]
        >>> vocab = text.Vocab.from_list(vocab_list)
        >>> tokenizer_op = MobileBertTokenizer(vocab=vocab, lower_case=True)
        >>> text = "i make a small mistake when i\'m working! 床前明月光😀"
        >>> test_dataset = ['A small mistake was made when I was working.']
        >>> dataset = GeneratorDataset(test_dataset, 'text')
        >>> tokenized_text = tokenizer_op(text)
        >>> tokenized_dataset = dataset.map(operations=tokenizer_op)
        >>> #encode method will return a Encoding class with many useful attributes
        >>> tokens = tokenizer_op.encode(text)
        >>> tokens_offset = tokens.offsets
        ```
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
        """
        Initialize the MobileBertTokenizer class.

        Args:
            self: The instance of the class.
            vocab (msVocab, Vocab, str): The vocabulary used for tokenization. If vocab is an instance of msVocab,
                vocab_dict is obtained from vocab.vocab(). If vocab is an instance of Vocab, vocab_dict is
                obtained directly from vocab.vocab. If vocab is a path to a file, the file is loaded as the vocabulary.
                Default is None.
            tokenizer_file (str): Path to the tokenizer file. Default is None.
            do_lower_case (bool): Whether to convert tokens to lowercase. Default is True.
            unk_token (str): The token representing unknown words. Default is '[UNK]'.
            sep_token (str): The token representing separator. Default is '[SEP]'.
            pad_token (str): The token representing padding. Default is '[PAD]'.
            cls_token (str): The token representing classification. Default is '[CLS]'.
            mask_token (str): The token representing masking. Default is '[MASK]'.
            tokenize_chinese_chars (bool): Whether to tokenize Chinese characters. Default is True.
            strip_accents (None): Not used in this implementation.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            ValueError: If the vocab parameter is not of type msVocab or Vocab, or if it is a string that
                does not represent a file.
        """
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
        Converts a token to its corresponding token ID using the MobileBERT Tokenizer.
        
        Args:
            self (MobileBertTokenizer): An instance of the MobileBertTokenizer class.
                It represents the tokenizer object used for tokenization.
            token (str): The input token to be converted to its corresponding token ID.
                This token should be a valid string representing a token in the tokenizer's vocabulary.
        
        Returns:
            None: This method does not return a value directly but instead passes the token to the tokenizer to
                retrieve its ID.
        
        Raises:
            TypeError: If the token is not a string.
            ValueError: If the token provided is not found in the tokenizer's vocabulary.
        """
        return self._tokenizer.token_to_id(token)

__all__ = ['MobileBertTokenizer']
