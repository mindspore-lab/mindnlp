# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for BlenderbotSmall."""

import json
import os
from typing import Dict, List, Optional, Tuple

import regex as re

from ...tokenization_utils import PreTrainedTokenizer
from ....utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}


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

    pairs = set(pairs)
    return pairs


class BlenderbotSmallTokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    the superclass for more information regarding methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        merges_file (`str`):
            Path to the merges file.
        bos_token (`str`, *optional*, defaults to `"__start__"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"__end__"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"__unk__"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"__null__"`):
            The token used for padding, for example when batching sequences of different lengths.
        kwargs (*optional*):
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="__start__",
        eos_token="__end__",
        unk_token="__unk__",
        pad_token="__null__",
        **kwargs,
    ):
        """
        Initializes a BlenderbotSmallTokenizer instance with the provided parameters.
        
        Args:
            self (BlenderbotSmallTokenizer): The instance of the BlenderbotSmallTokenizer class.
            vocab_file (str): The file path to the vocabulary file containing encoding information.
            merges_file (str): The file path to the merges file containing BPE merges information.
            bos_token (str, optional): The beginning of sentence token. Defaults to '__start__'.
            eos_token (str, optional): The end of sentence token. Defaults to '__end__'.
            unk_token (str, optional): The unknown token. Defaults to '__unk__'.
            pad_token (str, optional): The padding token. Defaults to '__null__'.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None: This method initializes the BlenderbotSmallTokenizer instance with the provided parameters.
        
        Raises:
            FileNotFoundError: If either vocab_file or merges_file is not found.
            JSONDecodeError: If there is an issue decoding the vocabulary file.
            IndexError: If there is an issue accessing elements during initialization.
        """
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary used by the BlenderbotSmallTokenizer instance.
        
        Args:
            self: The instance of the BlenderbotSmallTokenizer class.
        
        Returns:
            An integer representing the size of the vocabulary.
        
        Raises:
            None.
        """
        return len(self.encoder)

    def get_vocab(self) -> Dict:
        """
        Retrieve the vocabulary of the BlenderbotSmallTokenizer.
        
        Args:
            self (BlenderbotSmallTokenizer): The instance of the BlenderbotSmallTokenizer class.
        
        Returns:
            Dict: A dictionary representing the vocabulary of the tokenizer, containing the encoder and added tokens encoder.
        
        Raises:
            None.
        """
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token: str) -> str:
        """
        The 'bpe' method in the 'BlenderbotSmallTokenizer' class performs Byte Pair Encoding (BPE) on a given token.
        
        Args:
            self (BlenderbotSmallTokenizer): An instance of the BlenderbotSmallTokenizer class.
            token (str): The input token to be processed with BPE.
        
        Returns:
            str: The token after BPE processing.
        
        Raises:
            None.
        
        This method applies the following steps to perform BPE:

        1. Checks if the token exists in the cache. If yes, returns the cached value.
        2. Applies regular expression substitution to separate certain punctuation marks from the token.
        3. Replaces single quotes with spaces around them.
        4. Reduces consecutive whitespace characters to a single space.
        5. If the token contains a newline character, replaces it with '__newln__'.
        6. Splits the token into a list of individual words.
        7. Processes each word in the list:

            - Converts the word to lowercase.
            - Converts the word into a tuple.
            - Appends '</w>' to the last character of the tuple.
            - Retrieves the pairs of characters in the word.
            - If no pairs are found, appends the original word to the final list and continues to the next word.
            - Continues to find and merge the most frequent pair of characters in the word until no more relevant pairs are found.
            - Joins the merged characters with '@@ ' and removes the '</w>' suffix.
            - Caches the processed word for future use.
            - Appends the processed word to the final list.

        8. Joins all the words in the final list with a space separator and returns the result.
        """
        if token in self.cache:
            return self.cache[token]
        token = re.sub("([.,!?()])", r" \1", token)
        token = re.sub("(')", r" \1 ", token)
        token = re.sub(r"\s{2,}", " ", token)
        if "\n" in token:
            token = token.replace("\n", " __newln__")

        tokens = token.split(" ")
        words = []
        for token in tokens:
            if not len(token): # pylint: disable=use-implicit-booleaness-not-len
                continue

            token = token.lower()
            word = tuple(token)
            word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
            pairs = get_pairs(word)

            if not pairs:
                words.append(token)
                continue

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
                        new_word.extend(word[i:j])
                        i = j
                    except ValueError:
                        new_word.extend(word[i:])
                        break

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
                else:
                    pairs = get_pairs(word)
            word = "@@ ".join(word)
            word = word[:-4]

            self.cache[token] = word
            words.append(word)
        return " ".join(words)

    def _tokenize(self, text: str) -> List[str]:
        """Split a string into tokens using BPE."""
        split_tokens = []

        words = re.findall(r"\S+\n?", text)

        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to an id using the vocab."""
        token = token.lower()
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string."""
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and merge files for the BlenderbotSmallTokenizer.
        
        Args:
            self (BlenderbotSmallTokenizer): An instance of the BlenderbotSmallTokenizer class.
            save_directory (str): The directory where the vocabulary and merge files will be saved.
            filename_prefix (Optional[str], optional): A prefix to be added to the filename. Defaults to None.
        
        Returns:
            Tuple[str]: A tuple containing the paths of the saved vocabulary and merge files.
        
        Raises:
            FileNotFoundError: If the specified save_directory does not exist.
            TypeError: If the save_directory is not of type str.
            ValueError: If the save_directory is not a valid directory path.
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

    @property
    # Copied from transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )

__all__ = ['BlenderbotSmallTokenizer']
