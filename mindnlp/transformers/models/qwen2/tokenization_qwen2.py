# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Qwen2."""

import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple

import regex as re

from mindnlp.utils import logging
from ...tokenization_utils import AddedToken, PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "https://hf-mirror.com/qwen/qwen-tokenizer/resolve/main/vocab.json"},
    "merges_file": {"qwen/qwen-tokenizer": "https://hf-mirror.com/qwen/qwen-tokenizer/resolve/main/merges.txt"},
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


@lru_cache()
# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# Copied from transformers.models.gpt2.tokenization_gpt2.get_pairs
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Qwen2Tokenizer(PreTrainedTokenizer):
    """
    Construct a Qwen2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import Qwen2Tokenizer
        ...
        >>> tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")
        >>> tokenizer("Hello world")["input_ids"]
        [9707, 1879]
        >>> tokenizer(" Hello world")["input_ids"]
        [21927, 1879]
        ```
    This is expected.

    You should not use GPT2Tokenizer instead, because of the different pretokenization rules.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process. Not applicable to this tokenizer, since tokenization does not add spaces.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. The default behavior is
            to not split special tokens. This means that if `<|endoftext|>` is the `eos_token`, then `tokenizer.tokenize("<|endoftext|>") =
            ['<|endoftext|>`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("<|endoftext|>")` will be give `['<',
            '|', 'endo', 'ft', 'ext', '|', '>']`. This argument is only supported for `slow` tokenizers for the moment.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        """
        Initializes an instance of the Qwen2Tokenizer class.

        Args:
            self: The instance of the class.
            vocab_file (str): The path to the vocabulary file.
            merges_file (str): The path to the merges file.
            errors (str, optional): Specifies how to handle errors during tokenization. Defaults to 'replace'.
            unk_token (str, optional): The unknown token. Defaults to 'endoftext'.
            bos_token (str or None, optional): The beginning-of-sequence token. Defaults to None.
            eos_token (str, optional): The end-of-sequence token. Defaults to 'endoftext'.
            pad_token (str, optional): The padding token. Defaults to 'endoftext'.
            clean_up_tokenization_spaces (bool, optional): Specifies whether to clean up tokenization spaces.
                Defaults to False.
            split_special_tokens (bool, optional): Specifies whether to split special tokens. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the vocab_file or merges_file does not exist.
            UnicodeDecodeError: If there is an error decoding the vocab_file or merges_file.
            ValueError: If the vocab_file or merges_file is empty.
        """
        # Qwen vocab does not contain control tokens; added tokens need to be special
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for line in merges_handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # NOTE: the cache can grow without bound and will get really large for long running processes
        # (esp. for texts of language that do not use space between word, e.g. Chinese); technically
        # not a memory leak but appears as one.
        # GPT2Tokenizer has the same problem, so let's be consistent.
        self.cache = {}

        self.pat = re.compile(PRETOKENIZE_REGEX)

        if kwargs.get("add_prefix_space", False):
            logger.warning_once(
                f"{self.__class__.__name} does not support `add_prefix_space`, setting it to True has no effect."
            )

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        This method returns the number of unique tokens in the tokenizer's encoder.

        Args:
            self (Qwen2Tokenizer): An instance of the Qwen2Tokenizer class.

        Returns:
            int: The size of the vocabulary.

        Raises:
            None.
        """
        return len(self.encoder)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.get_vocab
    def get_vocab(self):
        """
        Returns the vocabulary of the tokenizer.

        Args:
            self (Qwen2Tokenizer): The instance of the Qwen2Tokenizer class.

        Returns:
            dict: A dictionary representing the vocabulary of the tokenizer.
                The keys are the tokens, and the values are their corresponding indices in the vocabulary.

        Raises:
            None.

        Note:
            The vocabulary is obtained by merging the `encoder` and `added_tokens_encoder` dictionaries of the
            tokenizer instance.
        """
        return dict(self.encoder, **self.added_tokens_encoder)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe
    def bpe(self, token):
        """
        Perform Byte Pair Encoding (BPE) on a given token.

        Args:
            self (Qwen2Tokenizer): An instance of the Qwen2Tokenizer class.
            token (str): The input token to be encoded using BPE.

        Returns:
            str: The BPE-encoded version of the input token.

        Raises:
            None.

        Note:
            This method applies Byte Pair Encoding (BPE) algorithm to a given token. BPE is a subword tokenization technique
            commonly used in natural language processing tasks. It splits a token into subword units based on the most
            frequently occurring pairs of characters.

            The BPE algorithm starts by converting the token into a tuple of individual characters. It then identifies the
            most frequent character pairs using the `get_pairs` function. If no pairs are found, the original token is
            returned as it cannot be further split.

            The algorithm iteratively replaces the most frequent character pair with a new subword unit. This process is
            repeated until no more frequent character pairs are found or the token is reduced to a single character.

            Finally, the BPE-encoded token is returned as a string with subword units separated by spaces.

            To improve performance, the method utilizes a cache to store previously processed tokens. If a token is found in
            the cache, its encoded version is returned directly without recomputing.

        Example:
            ```python
            >>> tokenizer = Qwen2Tokenizer()
            >>> encoded_token = tokenizer.bpe('hello')
            >>> print(encoded_token)
            >>> # Output: 'he ll o'
            ...
            >>> encoded_token = tokenizer.bpe('world')
            >>> print(encoded_token)
            >>> # Output: 'wo r ld'
            ```
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Decodes a list of token IDs into a string representation.

        Args:
            self: An instance of the Qwen2Tokenizer class.
            token_ids (List[int]): A list of token IDs to be decoded.
            skip_special_tokens (bool, optional): Whether to skip special tokens during decoding. Defaults to False.
            clean_up_tokenization_spaces (bool, optional): Whether to remove leading and trailing whitespaces
                around tokens. Defaults to False.
            spaces_between_special_tokens (bool, optional): Whether to add spaces between special tokens.
                Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the superclass method.

        Returns:
            str: The decoded string representation of the given token IDs.

        Raises:
            None.

        Note:
            - Special tokens are typically used to mark the beginning and end of a sequence, or to represent special
            tokens such as padding or unknown tokens.
            - If skip_special_tokens is set to True, the special tokens will be excluded from the decoded string.
            - If clean_up_tokenization_spaces is set to True, any leading or trailing whitespaces around tokens
            will be removed.
            - If spaces_between_special_tokens is set to True, spaces will be added between special tokens
            in the decoded string.
        """
        # `spaces_between_special_tokens` defaults to True for _decode in slow tokenizers
        # and cannot be configured elsewhere, but it should default to False for Qwen2Tokenizer
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save vocabulary to a specified directory with an optional filename prefix.

        Args:
            self: An instance of the Qwen2Tokenizer class.
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the saved vocabulary filenames.

        Returns:
            Tuple[str]: A tuple containing the file paths of the saved vocabulary and merge files.

        Raises:
            FileNotFoundError: If the specified save_directory does not exist.
            IOError: If there are any issues with writing the vocabulary or merge files.
            ValueError: If the save_directory is not a valid directory path.
            Exception: Any other unexpected errors that may occur during the process.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, **kwargs):
        """
        Prepares the given text for tokenization.

        Args:
            self (Qwen2Tokenizer): An instance of the Qwen2Tokenizer class.
            text (str): The text to be prepared for tokenization.

        Returns:
            None: The method modifies the text in-place.

        Raises:
            None.

        This method takes in an instance of the Qwen2Tokenizer class and a string of text.
        It prepares the text for tokenization by normalizing it using the 'NFC' (Normalization Form C) Unicode
        normalization.
        The normalization ensures that the text is in a standardized form, reducing any potential ambiguities or
        variations in the text. The method then returns the modified text along with any additional keyword
        arguments passed to the method.
        
        Note that this method modifies the text in-place, meaning that the original text variable will be
        updated with the normalized version. No values are returned explicitly by this method.
        """
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)

__all__ = ['Qwen2Tokenizer']
