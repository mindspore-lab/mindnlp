# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tokenization classes for CPMAnt."""
import collections
import os
from typing import List, Optional, Tuple

from mindnlp.utils import is_jieba_available, requires_backends
from mindnlp.utils import logging
from ...tokenization_utils import PreTrainedTokenizer


if is_jieba_available():
    import jieba


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openbmb/cpm-ant-10b": "https://hf-mirror.com/openbmb/cpm-ant-10b/blob/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openbmb/cpm-ant-10b": 1024,
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class WordpieceTokenizer:

    """
    The WordpieceTokenizer class represents a tokenizer that tokenizes input text into subword tokens using the WordPiece algorithm.
    
    Attributes:
        vocab (dict): A dictionary containing the vocabulary of subword tokens.
        unk_token (str): The token to be used for out-of-vocabulary or unknown words.
        max_input_chars_per_word (int): The maximum number of input characters per word for tokenization.
    
    Methods:
        tokenize(token):
            Tokenizes the input token into subword tokens using the WordPiece algorithm and the specified vocabulary.
    
    Example:
        ```python
        >>> vocab = {'hello': 'he', 'world': 'wo', 'hello,': 'hello'}
        >>> tokenizer = WordpieceTokenizer(vocab, '<unk>', 200)
        >>> tokenized_text = tokenizer.tokenize('helloworld')
        ```
    """
    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        """
        Initializes a new instance of the WordpieceTokenizer class.

        Args:
            self (WordpieceTokenizer): The current instance of the WordpieceTokenizer class.
            vocab (list): A list of strings representing the vocabulary for the tokenizer.
            unk_token (str, optional): The token to use for unknown words. Defaults to '<unk>'.
            max_input_chars_per_word (int, optional): The maximum number of characters allowed per word. Defaults to 200.

        Returns:
            None

        Raises:
            None.

        This method initializes the WordpieceTokenizer object with the provided vocabulary, unknown token, and maximum input characters per word.
        The vocabulary is a list of strings that represents the set of tokens used by the tokenizer.
        The unk_token parameter allows customization of the token used to represent unknown words. If not provided, it defaults to '<unk>'.
        The max_input_chars_per_word parameter limits the number of characters allowed per word.
        If a word exceeds this limit, it will be split into subwords.

        Example:
            ```python
            >>> tokenizer = WordpieceTokenizer(vocab=['hello', 'world'], unk_token='<unk>', max_input_chars_per_word=200)
            ```
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        """
        This method tokenizes a given input token into sub-tokens based on the vocabulary of the WordpieceTokenizer class.

        Args:
            self (WordpieceTokenizer): The instance of the WordpieceTokenizer class.
                It is used to access the vocabulary and maximum input characters per word.
            token (str): The input token to be tokenized.
                It represents the word to be broken down into sub-tokens.
                Must be a string.

        Returns:
            list: A list of sub-tokens generated from the input token based on the vocabulary.
                If the length of the input token exceeds the maximum allowed characters per word,
                it returns a list containing the unknown token (unk_token).
                Otherwise, it returns a list of sub-tokens that are part of the vocabulary or the unknown token.

        Raises:
            None
        """
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
            else:
                sub_tokens.append(cur_substr)
                start = end

        return sub_tokens


class CpmAntTokenizer(PreTrainedTokenizer):
    """
    Construct a CPMAnt tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bod_token (`str`, *optional*, defaults to `"<d>"`):
            The beginning of document token.
        eod_token (`str`, *optional*, defaults to `"</d>"`):
            The end of document token.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        line_token (`str`, *optional*, defaults to `"</n>"`):
            The line token.
        space_token (`str`, *optional*, defaults to `"</_>"`):
            The space token.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    add_prefix_space = False

    def __init__(
        self,
        vocab_file,
        bod_token="<d>",
        eod_token="</d>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        line_token="</n>",
        space_token="</_>",
        padding_side="left",
        **kwargs,
    ):
        """
        Initialize a CpmAntTokenizer object with the provided parameters.

        Args:
            vocab_file (str): The path to the vocabulary file to load.
            bod_token (str, optional): Beginning of document token (default is '<d>').
            eod_token (str, optional): End of document token (default is '</d>').
            bos_token (str, optional): Beginning of sentence token (default is '<s>').
            eos_token (str, optional): End of sentence token (default is '</s>').
            pad_token (str, optional): Padding token (default is '<pad>').
            unk_token (str, optional): Token for unknown words (default is '<unk>').
            line_token (str, optional): Line break token (default is '</n>').
            space_token (str, optional): Space token (default is '</_>').
            padding_side (str, optional): Side for padding (default is 'left').

        Returns:
            None.

        Raises:
            MissingBackendError: If required backend 'jieba' is not available.
            FileNotFoundError: If the specified 'vocab_file' does not exist.
            KeyError: If 'space_token' or 'line_token' are missing in the loaded vocabulary.
            Exception: Any other unforeseen error that may occur during initialization.
        """
        requires_backends(self, ["jieba"])
        self.bod_token = bod_token
        self.eod_token = eod_token
        self.encoder = load_vocab(vocab_file)
        self.encoder[" "] = self.encoder[space_token]
        self.encoder["\n"] = self.encoder[line_token]

        del self.encoder[space_token]
        del self.encoder[line_token]

        self.encoder = collections.OrderedDict(sorted(self.encoder.items(), key=lambda x: x[1]))
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder, unk_token=unk_token)

        super().__init__(
            bod_token=bod_token,
            eod_token=eod_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            line_token=line_token,
            space_token=space_token,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def bod_token_id(self):
        """
        This method, 'bod_token_id', is a property method defined in the 'CpmAntTokenizer' class.
        It takes no external parameters and returns the token ID associated with the 'bod_token'.

        Args:
            self (CpmAntTokenizer): The instance of the CpmAntTokenizer class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.encoder[self.bod_token]

    @property
    def eod_token_id(self):
        """
        This method 'eod_token_id' in the class 'CpmAntTokenizer' retrieves the token ID of the end-of-document token.

        Args:
            self: An instance of the class CpmAntTokenizer.
                It is required as this method is part of the class and needs access to its attributes and methods.

        Returns:
            None: This method returns a value of type None.
                It retrieves the token ID of the end-of-document token from the encoder attribute of the class instance.

        Raises:
            None.
        """
        return self.encoder[self.eod_token]

    @property
    def newline_id(self):
        r"""
        This method, newline_id, in the class CpmAntTokenizer, returns the value associated with the newline character in the encoder.

        Args:
            self (CpmAntTokenizer): The instance of the CpmAntTokenizer class.

        Returns:
            None.

        Raises:
            KeyError: If the newline character `'\n'` is not found in the encoder dictionary, a KeyError is raised.
        """
        return self.encoder["\n"]

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary used by the CpmAntTokenizer instance.

        Args:
            self: The CpmAntTokenizer instance itself.

        Returns:
            int: The number of unique tokens in the vocabulary.

        Raises:
            None.
        """
        return len(self.encoder)

    def get_vocab(self):
        """
        Retrieves the vocabulary of the CpmAntTokenizer instance.

        Args:
            self (CpmAntTokenizer): The instance of CpmAntTokenizer.

        Returns:
            dict: The vocabulary of the tokenizer, which is a dictionary mapping tokens to their corresponding IDs.

        Raises:
            None.

        Example:
            ```python
            >>> tokenizer = CpmAntTokenizer()
            >>> vocab = tokenizer.get_vocab()
            >>> vocab
            {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, ...}
            ```
        """
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text):
        """Tokenize a string."""
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens

    def _decode(self, token_ids, **kwargs):
        """Decode ids into a string."""
        token_ids = [i for i in token_ids if i >= 0]
        token_ids = [
            x for x in token_ids if x not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)
        ]
        return super()._decode(token_ids, **kwargs)

    def check(self, token):
        """
        Check if a token is present in the encoder of the CpmAntTokenizer.

        Args:
            self (CpmAntTokenizer): An instance of the CpmAntTokenizer class.
            token (Any): The token to be checked.

        Returns:
            None.

        Raises:
            None.
        """
        return token in self.encoder

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens into a string representation.

        Args:
            self (CpmAntTokenizer): An instance of the CpmAntTokenizer class.
            tokens (List[str]): A list of tokens to be converted into a string representation.

        Returns:
            str: A string representation of the tokens.

        Raises:
            None.

        Note:
            - The tokens should be provided as a list of strings.
            - The method will join the tokens together using an empty string as a separator.

        Example:
            ```python
            >>> tokenizer = CpmAntTokenizer()
            >>> tokens = ['Hello', 'world', '!']
            >>> tokenizer.convert_tokens_to_string(tokens)
            'Hello world!'
            ```
        """
        return "".join(tokens)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a file with the specified directory and filename prefix.

        Args:
            self: Instance of the CpmAntTokenizer class.
            save_directory (str): The directory where the vocabulary file will be saved.
            filename_prefix (Optional[str]): A string to be prefixed to the filename. Defaults to None.

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.

        Raises:
            None.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        index = 0
        if " " in self.encoder:
            self.encoder["</_>"] = self.encoder[" "]
            del self.encoder[" "]
        if "\n" in self.encoder:
            self.encoder["</n>"] = self.encoder["\n"]
            del self.encoder["\n"]
        self.encoder = collections.OrderedDict(sorted(self.encoder.items(), key=lambda x: x[1]))
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in self.encoder.items():
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A CPMAnt sequence has the following format:

        - single sequence: `[BOS] Sequence`.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence that special tokens will be added.
            token_ids_1 (`List[int]`): The optional second tokenized sequence that special tokens will be added.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0
        return [self.bos_token_id] + token_ids_0 + [self.bos_token_id] + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        return [1] + ([0] * len(token_ids_0))

__all__ = ['CpmAntTokenizer']
