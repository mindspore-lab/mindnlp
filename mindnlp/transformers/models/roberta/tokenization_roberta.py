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
"""Tokenization classes for RoBERTa."""

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
        "roberta-base": "https://hf-mirror.com/roberta-base/resolve/main/vocab.json",
        "roberta-large": "https://hf-mirror.com/roberta-large/resolve/main/vocab.json",
        "roberta-large-mnli": "https://hf-mirror.com/roberta-large-mnli/resolve/main/vocab.json",
        "distilroberta-base": "https://hf-mirror.com/distilroberta-base/resolve/main/vocab.json",
        "roberta-base-openai-detector": "https://hf-mirror.com/roberta-base-openai-detector/resolve/main/vocab.json",
        "roberta-large-openai-detector": (
            "https://hf-mirror.com/roberta-large-openai-detector/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "roberta-base": "https://hf-mirror.com/roberta-base/resolve/main/merges.txt",
        "roberta-large": "https://hf-mirror.com/roberta-large/resolve/main/merges.txt",
        "roberta-large-mnli": "https://hf-mirror.com/roberta-large-mnli/resolve/main/merges.txt",
        "distilroberta-base": "https://hf-mirror.com/distilroberta-base/resolve/main/merges.txt",
        "roberta-base-openai-detector": "https://hf-mirror.com/roberta-base-openai-detector/resolve/main/merges.txt",
        "roberta-large-openai-detector": (
            "https://hf-mirror.com/roberta-large-openai-detector/resolve/main/merges.txt"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
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


class RobertaTokenizer(PreTrainedTokenizer):
    """
    Constructs a RoBERTa tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import RobertaTokenizer
        ...
        >>> tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        >>> tokenizer("Hello world")["input_ids"]
        [0, 31414, 232, 2]
        >>> tokenizer(" Hello world")["input_ids"]
        [0, 20920, 232, 2]
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
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
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
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs,
    ):
        """
        This method initializes an instance of the RobertaTokenizer class.

        Args:
            self: The instance of the class.
            vocab_file (str): The path to the vocabulary file.
            merges_file (str): The path to the merges file.
            errors (str, optional): The error handling scheme for encoding and decoding. Defaults to 'replace'.
            bos_token (str, optional): The beginning of sequence token. Defaults to '<s>'.
            eos_token (str, optional): The end of sequence token. Defaults to '</s>'.
            sep_token (str, optional): The separator token. Defaults to '</s>'.
            cls_token (str, optional): The classification token. Defaults to '<s>'.
            unk_token (str, optional): The unknown token. Defaults to '<unk>'.
            pad_token (str, optional): The padding token. Defaults to '<pad>'.
            mask_token (str, optional): The mask token. Defaults to '<mask>'.
            add_prefix_space (bool, optional): Whether to add prefix space. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            FileNotFoundError: If vocab_file or merges_file is not found.
            ValueError: If an invalid argument is provided for any token parameter.
            IOError: If an I/O error occurs when handling the files.
        """
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # these special tokens are not part of the vocab.json, let's add them in the correct order

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
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary used by the RobertaTokenizer instance.

        Args:
            self (RobertaTokenizer): The instance of the RobertaTokenizer class.

        Returns:
            int: The number of unique tokens in the vocabulary of the tokenizer.

        Raises:
            None.
        """
        return len(self.encoder)

    def get_vocab(self):
        """
        Returns a vocabulary dictionary containing both the base encoder and any additional tokens added to the tokenizer.

        Args:
            self (RobertaTokenizer): An instance of the RobertaTokenizer class.

        Returns:
            dict or None:
                The vocabulary dictionary containing the base encoder and any additional tokens added to the tokenizer.
                If the tokenizer has not been initialized with a base encoder or any additional tokens, None is returned.

        Raises:
            None.

        Note:
            The vocabulary dictionary is created by copying the base encoder dictionary and updating it with the
            added_tokens_encoder dictionary. The base encoder dictionary contains the original encoding for the tokenizer,
            while the added_tokens_encoder dictionary contains any additional tokens that have been added to the tokenizer.

        Example:
            ```python
            >>> tokenizer = RobertaTokenizer()
            >>> vocab = tokenizer.get_vocab()
            >>> print(vocab)
            {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 4}
            ```
        """
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def bpe(self, token):
        """
        This method is a part of the RobertaTokenizer class and implements Byte-Pair Encoding (BPE) for tokenizing input tokens.

        Args:
            self (RobertaTokenizer): The instance of the RobertaTokenizer class.
            token (str): The input token to be tokenized using BPE. It is a string representing a single token to
                be processed. Must not be None.

        Returns:
            str: The token after applying the Byte-Pair Encoding process. Returns the token as a string after processing
                it through the BPE algorithm.

        Raises:
            ValueError: If the input token is None or empty.
            TypeError: If the input token is not a string.
            KeyError: If the input token is not found in the cache attribute of the RobertaTokenizer instance.
            IndexError: If an index is out of range while processing the token.
            Exception: Any other unexpected exceptions may be raised during the BPE process.
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
        Saves the vocabulary and merge files of the RobertaTokenizer.

        Args:
            self (RobertaTokenizer): An instance of the RobertaTokenizer class.
            save_directory (str): The directory where the vocabulary and merge files will be saved.
            filename_prefix (Optional[str], optional): The prefix to be added to the filenames. Defaults to None.

        Returns:
            Tuple[str]: A tuple containing the paths of the saved vocabulary and merge files.

        Raises:
            OSError: If the save_directory is not a valid directory.

        This method saves the vocabulary file and merge file used by the RobertaTokenizer.
        The vocabulary file contains the encoder dictionary in JSON format, while the merge file contains the BPE merge
        indices and tokens. The files are saved in the specified save_directory with optional filename_prefix added to
        the filenames.

        Note:
            If the save_directory does not exist, the method will raise an OSError.

        Example:
            ```python
            >>> tokenizer = RobertaTokenizer()
            >>> tokenizer.save_vocabulary('/path/to/save')
            ('/path/to/save/vocab.txt', '/path/to/save/merges.txt')
            ```
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

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

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

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Prepares the given text for tokenization by adding a prefix space if necessary.

        Args:
            self (RobertaTokenizer): The instance of the RobertaTokenizer class.
            text (str): The input text to be prepared for tokenization.
            is_split_into_words (bool, optional): A flag indicating whether the text is already split into words.
                Defaults to False.
            **kwargs: Additional keyword arguments.
                add_prefix_space (bool, optional):

                A flag indicating whether a prefix space should be added to the text.
                If not provided, the value from the self.add_prefix_space attribute will be used.

        Returns:
            str: The prepared text after adding a prefix space if required.

        Raises:
            None

        Note:
            - If is_split_into_words is True or add_prefix_space is True, and the text is not empty and does not start
            with a space, a prefix space will be added to the text.
            - The original kwargs dictionary is modified by removing the 'add_prefix_space' key using the pop() method.

        Example:
            ```python
            >>> tokenizer = RobertaTokenizer()
            >>> prepared_text = tokenizer.prepare_for_tokenization("Hello world!", is_split_into_words=True)
            >>> print(prepared_text)
            >>> # Output: " Hello world!"
            ```
        """
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

__all__ = ['RobertaTokenizer']
