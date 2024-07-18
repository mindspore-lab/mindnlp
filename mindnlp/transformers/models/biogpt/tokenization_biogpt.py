# coding=utf-8
# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science. All rights reserved.
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
"""Tokenization classes for BioGPT."""
import json
import os
from typing import List, Optional, Tuple

from mindnlp.utils import logging
from ...tokenization_utils import PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/biogpt": "https://hf-mirror.com/microsoft/biogpt/resolve/main/vocab.json",
    },
    "merges_file": {"microsoft/biogpt": "https://hf-mirror.com/microsoft/biogpt/resolve/main/merges.txt"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/biogpt": 1024,
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BioGptTokenizer(PreTrainedTokenizer):
    """
    Construct an FAIRSEQ Transformer tokenizer. Moses tokenization followed by Byte-Pair Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
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
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        """
        Initializes a new instance of the BioGptTokenizer class.
        
        Args:
            self: The instance of the class.
            vocab_file (str): The path to the vocabulary file.
            merges_file (str): The path to the merges file.
            unk_token (str, optional): The token to represent unknown words. Defaults to '<unk>'.
            bos_token (str, optional): The token to represent the beginning of a sentence. Defaults to '<s>'.
            eos_token (str, optional): The token to represent the end of a sentence. Defaults to '</s>'.
            sep_token (str, optional): The token to represent sentence separation. Defaults to '</s>'.
            pad_token (str, optional): The token to represent padding. Defaults to '<pad>'.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None
        
        Raises:
            ImportError: If sacremoses library is not installed.
            IOError: If the vocabulary or merges file cannot be read.
        """
        try:
            import sacremoses
        except ImportError as e:
            raise ImportError(
                "You need to install sacremoses to use BioGptTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            ) from e

        self.lang = "en"
        self.sm = sacremoses
        # cache of sm.MosesTokenizer instance
        self.cache_moses_tokenizer = {}
        self.cache_moses_detokenizer = {}

        """ Initialisation"""
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return len(self.encoder)

    def get_vocab(self):
        """
        Method to retrieve the vocabulary dictionary consisting of tokens and their corresponding encodings.
        
        Args:
            self (BioGptTokenizer): The instance of the BioGptTokenizer class.
                It represents the tokenizer object.
                
        Returns:
            None: The method returns a vocabulary dictionary that contains tokens and their respective encodings.
        
        Raises:
            None.
        """
        return dict(self.encoder, **self.added_tokens_encoder)

    def moses_tokenize(self, text, lang):
        """
        Perform Moses tokenization on the given text.
        
        Args:
            self (BioGptTokenizer): An instance of the BioGptTokenizer class.
            text (str): The text to be tokenized.
            lang (str): The language code for tokenization.
            
        Returns:
            None
            
        Raises:
            KeyError: If the language code is not found in the cache_moses_tokenizer dictionary.
            ValueError: If the language code is invalid or unsupported.
            Exception: If any other error occurs during tokenization.
        
        This method utilizes the MosesTokenizer from the nltk.translate.moses package to tokenize the input text.
        It first checks if the MosesTokenizer for the specified language is already cached.
        If not, it creates a new MosesTokenizer instance for the language and adds it to the cache.
        The tokenization is then performed using the cached MosesTokenizer object.

        The 'aggressive_dash_splits', 'return_str', and 'escape' parameters are passed to the tokenize method of
        the MosesTokenizer.
        'aggressive_dash_splits' determines whether to perform aggressive dash splitting,
        'return_str' specifies whether to return a string or a list of tokens,
        and 'escape' determines whether to escape XML/HTML characters in the text before tokenization.

        Note:
            This method assumes that the BioGptTokenizer instance has been properly initialized with the necessary
            resources for tokenization.
        """
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        return self.cache_moses_tokenizer[lang].tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=True
        )

    def moses_detokenize(self, tokens, lang):
        """
        Performs Moses detokenization on a list of tokens for a specified language.

        Args:
            self (BioGptTokenizer): An instance of the BioGptTokenizer class.
            tokens (list): A list of tokens to be detokenized.
            lang (str): The language of the tokens. Must be a supported language.

        Returns:
            None: The method modifies the cache_moses_detokenizer attribute of the BioGptTokenizer instance.

        Raises:
            KeyError: If the specified language is not supported.
            TypeError: If the tokens parameter is not a list.

        Note:
            This method utilizes a cache to store MosesDetokenizer objects for each language,
            ensuring efficient detokenization by reusing previously created instances.
        """
        if lang not in self.cache_moses_detokenizer:
            moses_detokenizer = self.sm.MosesDetokenizer(lang=lang)
            self.cache_moses_detokenizer[lang] = moses_detokenizer
        return self.cache_moses_detokenizer[lang].detokenize(tokens)

    def bpe(self, token):
        """
        Performs Byte Pair Encoding (BPE) on a given token.

        Args:
            self: An instance of the BioGptTokenizer class.
            token (str): The token to be encoded using BPE.

        Returns:
            str: The BPE-encoded representation of the token.

        Raises:
            None.

        Description:
            This method takes a token and applies Byte Pair Encoding (BPE) to it. BPE is a subword tokenization
            technique that breaks down a token into a sequence of subword units.
            The BPE algorithm iteratively  merges the most frequent pairs of subword units to create a vocabulary
            of subword units.

            The token parameter is the input token to be encoded using BPE. The token is expected to be a string.

            The method returns the BPE-encoded representation of the token as a string.
            The encoded representation is obtained by iteratively merging the most frequent pairs of subword units
            until no more merges can be made.
            The resulting subword units are then joined together to form the encoded token.

            Note that the method may use a cache to store previously encoded tokens for efficiency.

        Example:
            ```python
            >>> tokenizer = BioGptTokenizer()
            >>> encoded_token = tokenizer.bpe('sequence')
            >>> print(encoded_token)
            >>> # Output: 'seq uence'</w>'
            ```
        """
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

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
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def _tokenize(self, text, bypass_tokenizer=False):
        """Returns a tokenized string."""
        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.moses_tokenize(text, self.lang)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # remove BPE
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        tokens = "".join(tokens).split()
        # detokenize
        text = self.moses_detokenize(tokens, self.lang)
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BioGPT sequence has the following format:

        - single sequence: `</s> X `
        - pair of sequences: `</s> A </s> B `

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.sep_token_id] + token_ids_0
        sep = [self.sep_token_id]
        return sep + token_ids_0 + sep + token_ids_1

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
        # no bos used in fairseq
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        return [1] + ([0] * len(token_ids_0))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A FAIRSEQ
        Transformer sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]

        # no bos used in fairseq
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to the specified directory with the given filename prefix.
        
        Args:
            self: Instance of the BioGptTokenizer class.
            save_directory (str): The directory path where the vocabulary files will be saved.
                It should already exist, and the method will raise an error if the directory does not exist.
            filename_prefix (Optional[str]): An optional prefix to be added to the filenames of the vocabulary files.
                If provided, the filenames will be prefixed with this value. Default is None.
        
        Returns:
            Tuple[str]: A tuple containing the paths to the saved vocabulary file and merge file.
        
        Raises:
            OSError: If the specified save_directory is not a valid directory.
            IOError: If there is an issue writing the vocabulary files to the disk.
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

    def __getstate__(self):
        """
        The '__getstate__' method in the 'BioGptTokenizer' class is used to retrieve the state of the object for pickling.
        
        Args:
            self: An instance of the 'BioGptTokenizer' class.
        
        Returns:
            None: This method does not explicitly return a value, but modifies the state of the object.
        
        Raises:
            None.
        """
        state = self.__dict__.copy()
        state["sm"] = None
        return state

    def __setstate__(self, d):
        """
        Sets the state of the BioGptTokenizer object.
        
        Args:
            self (BioGptTokenizer): The instance of the BioGptTokenizer class.
            d (dict): The dictionary containing the state information to be set. 
        
        Returns:
            None.
        
        Raises:
            ImportError: If the sacremoses module is not installed, an ImportError is raised. 
                The error message specifies that sacremoses needs to be installed and provides a link to the installation page.
        """
        self.__dict__ = d

        try:
            import sacremoses
        except ImportError as e:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            ) from e

        self.sm = sacremoses

__all__ = ['BioGptTokenizer']
