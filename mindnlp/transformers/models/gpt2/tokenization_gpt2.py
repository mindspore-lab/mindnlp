# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""


import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re

from mindnlp.utils import logging
from ...tokenization_utils import AddedToken, PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "gpt2": "https://hf-mirror.com/gpt2/resolve/main/vocab.json",
        "gpt2-medium": "https://hf-mirror.com/gpt2-medium/resolve/main/vocab.json",
        "gpt2-large": "https://hf-mirror.com/gpt2-large/resolve/main/vocab.json",
        "gpt2-xl": "https://hf-mirror.com/gpt2-xl/resolve/main/vocab.json",
        "distilgpt2": "https://hf-mirror.com/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "gpt2": "https://hf-mirror.com/gpt2/resolve/main/merges.txt",
        "gpt2-medium": "https://hf-mirror.com/gpt2-medium/resolve/main/merges.txt",
        "gpt2-large": "https://hf-mirror.com/gpt2-large/resolve/main/merges.txt",
        "gpt2-xl": "https://hf-mirror.com/gpt2-xl/resolve/main/merges.txt",
        "distilgpt2": "https://hf-mirror.com/distilgpt2/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}


@lru_cache()
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


class GPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import GPT2Tokenizer
        ...
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> tokenizer("Hello world")["input_ids"]
        [15496, 995]
        >>> tokenizer(" Hello world")["input_ids"]
        [18435, 995]
        ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

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
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        """Initializes a GPT2Tokenizer object.

        Args:
            self: The instance of the GPT2Tokenizer class.
            vocab_file (str): The path to the vocabulary file.
            merges_file (str): The path to the merges file.
            errors (str, optional): Specifies how to handle errors during tokenization. Defaults to 'replace'.
            unk_token (str, optional): The unknown token to be used during tokenization. Defaults to 'endoftext'.
            bos_token (str, optional): The beginning of sentence token. Defaults to 'endoftext'.
            eos_token (str, optional): The end of sentence token. Defaults to 'endoftext'.
            pad_token (str, optional): The padding token. Defaults to None.
            add_prefix_space (bool, optional): Specifies whether to add a prefix space to the input. Defaults to False.
            add_bos_token (bool, optional): Specifies whether to add the beginning of sentence token to the input. Defaults to False.

        Returns:
            None

        Raises:
            FileNotFoundError: If the vocab_file or merges_file is not found.
            UnicodeDecodeError: If there is an error decoding the vocab_file or merges_file.
        """
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """
        This method retrieves the vocabulary size of the GPT2Tokenizer.

        Args:
            self (GPT2Tokenizer): The instance of the GPT2Tokenizer class.

        Returns:
            int: The number of unique tokens in the tokenizer's vocabulary.

        Raises:
            None.
        """
        return len(self.encoder)

    def get_vocab(self):
        """
        Method to retrieve the vocabulary of the GPT2Tokenizer.

        Args:
            self: GPT2Tokenizer object. The instance of the GPT2Tokenizer class.

        Returns:
            dict or None: A merged dictionary containing the encoder and added tokens encoder.

        Raises:
            None.
        """
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        """
        This method 'bpe' in the class 'GPT2Tokenizer' implements byte pair encoding (BPE) algorithm for tokenization.

        Args:
            self (object): The instance of the GPT2Tokenizer class.
            token (str): The input token to be processed by the BPE algorithm. It should be a string representing a single token.

        Returns:
            str: The processed token after applying the BPE algorithm, which may involve merging characters based on predefined pairs.

        Raises:
            ValueError: If the input token 'token' is not a valid string or is empty.
            KeyError: If an error occurs while accessing or updating the cache dictionary within the method.
            IndexError: If an index error occurs during the processing of the token.
            Exception: Any other unforeseen exceptions that may occur during the execution of the BPE algorithm.
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

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Method to build inputs with special tokens in the GPT2Tokenizer class.

        Args:
            self: The instance of the GPT2Tokenizer class.
            token_ids_0 (list): List of token IDs for the first input.
            token_ids_1 (list, optional): List of token IDs for the second input. Default is None.

        Returns:
            None: This method does not return a value, but it modifies the input lists by adding special tokens.

        Raises:
            None.
        """
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to the specified directory.

        Args:
            self (GPT2Tokenizer): The instance of the GPT2Tokenizer class.
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (Optional[str], optional): The prefix to be added to the filename of the vocabulary files.
                Defaults to None.

        Returns:
            Tuple[str]: A tuple containing the paths of the saved vocabulary files.

        Raises:
            OSError: If the save_directory is not a valid directory.

        This method saves the vocabulary of the GPT2Tokenizer instance to the specified save_directory.
        The vocabulary is saved in two files: a vocabulary file and a merge file. The vocabulary file contains
        the encoder dictionary in JSON format, and the merge file contains the BPE merge indices.

        If the save_directory does not exist or is not a directory, an OSError is raised. The filename_prefix parameter
        is optional and can be used to add a prefix to the filename of the saved vocabulary files.
        If filename_prefix is not provided, no prefix will be added to the filenames.

        The method returns a tuple containing the paths of the saved vocabulary files, i.e., (vocab_file, merge_file).
        The vocab_file path points to the saved vocabulary file, and the merge_file path points to the saved merge file.
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

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Prepare for tokenization method in the GPT2Tokenizer class.
        
        Args:
            self (GPT2Tokenizer): The instance of the GPT2Tokenizer class.
            text (str): The input text to be prepared for tokenization.
            is_split_into_words (bool): A flag indicating whether the input text is already split into words.
                If True, the text will not be modified.
        
        Returns:
            tuple: A tuple containing the prepared text and any remaining keyword arguments after processing.
        
        Raises:
            None.
        """
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    @property
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://hf-mirror.com/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"

__all__ = ['GPT2Tokenizer']
