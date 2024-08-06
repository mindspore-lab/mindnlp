# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for BigBird."""
import os
import re
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from mindnlp.utils import logging
from ...tokenization_utils import AddedToken, PreTrainedTokenizer

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/bigbird-roberta-base": "https://hf-mirror.com/google/bigbird-roberta-base/resolve/main/spiece.model",
        "google/bigbird-roberta-large": (
            "https://hf-mirror.com/google/bigbird-roberta-large/resolve/main/spiece.model"
        ),
        "google/bigbird-base-trivia-itc": (
            "https://hf-mirror.com/google/bigbird-base-trivia-itc/resolve/main/spiece.model"
        ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}


class BigBirdTokenizer(PreTrainedTokenizer):
    """
    Construct a BigBird tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The begin of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

                - `nbest_size = {0,1}`: No sampling is performed.
                - `nbest_size > 1`: samples from the nbest_size results.
                - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
            BPE-dropout.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sep_token="[SEP]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes an instance of the BigBirdTokenizer class.

        Args:
            self: The instance of the BigBirdTokenizer class.
            vocab_file (str): Path to the vocabulary file.
            unk_token (str, optional): The token representing unknown words. Defaults to '<unk>'.
            bos_token (str, optional): The token representing the beginning of a sentence. Defaults to '<s>'.
            eos_token (str, optional): The token representing the end of a sentence. Defaults to '</s>'.
            pad_token (str, optional): The token representing padding. Defaults to '<pad>'.
            sep_token (str, optional): The token representing sentence separation. Defaults to '[SEP]'.
            mask_token (str, optional): The token representing masked words. Defaults to '[MASK]'.
            cls_token (str, optional): The token representing classification. Defaults to '[CLS]'.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for the SentencePieceProcessor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            None.
        """
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False)
            if isinstance(pad_token, str)
            else pad_token
        )
        cls_token = (
            AddedToken(cls_token, lstrip=False, rstrip=False)
            if isinstance(cls_token, str)
            else cls_token
        )
        sep_token = (
            AddedToken(sep_token, lstrip=False, rstrip=False)
            if isinstance(sep_token, str)
            else sep_token
        )

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """
        Method to retrieve the vocabulary size of the BigBirdTokenizer.

        Args:
            self (BigBirdTokenizer): The instance of the BigBirdTokenizer class.
                This parameter is required to access the tokenizer's properties.

        Returns:
            None: The method returns the vocabulary size as an integer value.

        Raises:
            None.
        """
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """
        This method returns the vocabulary for the BigBirdTokenizer.

        Args:
            self (BigBirdTokenizer): The instance of the BigBirdTokenizer class.

        Returns:
            dict: A dictionary containing the vocabulary, where keys are tokens and values are their corresponding ids.

        Raises:
            None
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        """
        The '__getstate__' method in the 'BigBirdTokenizer' class is used to retrieve the current state of the object
        for serialization. This method takes one parameter, 'self', which refers to the instance of
        the 'BigBirdTokenizer' class.

        Args:
            self (BigBirdTokenizer): The instance of the 'BigBirdTokenizer' class.

        Returns:
            None.

        Raises:
            None.
        """
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        """
        Sets the state of the BigBirdTokenizer object based on the provided dictionary.

        Args:
            self (BigBirdTokenizer): The instance of the BigBirdTokenizer class.
            d (dict): The dictionary containing the state information.

        Returns:
            None

        Raises:
            None

        This method sets the state of the BigBirdTokenizer object by assigning the dictionary 'd' to the '__dict__' attribute of the instance.
        If the instance does not have the 'sp_model_kwargs' attribute, it is initialized as an empty dictionary.
        The SentencePieceProcessor object 'sp_model' is then created and assigned to the 'sp_model' attribute of the instance.
        The 'sp_model_kwargs' dictionary is used to pass any additional keyword arguments to the SentencePieceProcessor initialization.
        Finally, the vocabulary file is loaded using the 'Load' method of the 'sp_model' object.
        """
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        """
        Decode the token IDs into a human-readable string.

        Args:
            self: The BigBirdTokenizer instance.
            token_ids (List[int]): A list of token IDs to be decoded into a string.
            skip_special_tokens (bool, optional): Whether to skip special tokens during decoding. Defaults to False.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces in the decoded text.
                Defaults to None.
            spaces_between_special_tokens (bool, optional):
                Whether to include spaces between special tokens in the decoded text. Defaults to True.

        Returns:
            str: The decoded string representation of the input token IDs.

        Raises:
            None.
            """
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # Mimic the behavior of the Rust tokenizer:
        # No space before [MASK] and [SEP]
        if spaces_between_special_tokens:
            text = re.sub(r" (\[(MASK|SEP)\])", r"\1", " ".join(sub_texts))
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        return text

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        '''
        Save the vocabulary to a specified directory with an optional filename prefix.

        Args:
            self (BigBirdTokenizer): The instance of the BigBirdTokenizer class.
            save_directory (str): The directory where the vocabulary will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the filename of the vocabulary. Defaults to None.

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.

        Raises:
            OSError: If the save_directory is not a valid directory.
            IOError: If the vocabulary file cannot be copied or written to the specified location.
        '''
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Big Bird sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

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
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
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
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ```0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second sequence |```

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
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]


__all__ = ["BigBirdTokenizer"]
