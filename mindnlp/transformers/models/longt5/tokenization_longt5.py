# coding=utf-8
# Copyright 2018 T5 Authors and HuggingFace Inc. team.
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Tokenization class for model LongT5.
"""

import os
import re
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from mindnlp.utils import logging
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken


if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/long-t5-local-base": "https://hf-mirror.com/google/long-t5-local-base/blob/main/config.json",
        "google/long-t5-local-large": "https://hf-mirror.com/google/long-t5-local-large/blob/main/config.json",
        "google/long-t5-tglobal-base": "https://hf-mirror.com/google/long-t5-tglobal-base/blob/main/config.json",
        "google/long-t5-tglobal-large": "https://hf-mirror.com/google/long-t5-tglobal-large/blob/main/config.json",
   }
}


# TODO(PVP) - this should be removed in Transformers v5
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/long-t5-local-base": 512,
    "google/long-t5-local-large": 512,
    "google/long-t5-tglobal-base": 512,
    "google/long-t5-tglobal-large":512,
}

SPIECE_UNDERLINE = "▁"


class LongT5Tokenizer(PreTrainedTokenizer):
    """
    Copied from T5Tokenizer
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        legacy=None,
        **kwargs,
    ) -> None:
        """
        Initializes a LongT5Tokenizer object.
        
        Args:
            self (object): The instance of the class.
            vocab_file (str): Path to the vocabulary file.
            eos_token (str, optional): End-of-sequence token. Default is '</s>'.
            unk_token (str, optional): Token for unknown words. Default is '<unk>'.
            pad_token (str, optional): Token for padding. Default is '<pad>'.
            extra_ids (int): Number of additional special tokens.
            additional_special_tokens (List[str], optional): List of additional special tokens.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Optional arguments for the SentencePiece model.
            legacy (bool, optional): Flag to indicate whether to use legacy behavior.

        Returns:
            None.

        Raises:
            ValueError: If both extra_ids and additional_special_tokens are provided, and they are not consistent.
            Exception: If an unexpected error occurs during the execution of the method.
        """
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens

        # for legacy purpose, we keep this. Will be removed and tests updated. (when `added_tokens_decoder` is not passed as kwargs)
        self._added_tokens_decoder = {}
        for i in range(len(extra_tokens)):
            self._added_tokens_decoder[len(self.sp_model) - 1 + extra_ids - i] = AddedToken(
                f"<extra_id_{i}>", single_word=True, lstrip=True, rstrip=True, special=True
            )

        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thouroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/27144"
            )
            legacy = True

        self.legacy = legacy
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            legacy=legacy,
            **kwargs,
        )

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        """
        This method retrieves a SentencePieceProcessor object for tokenization.

        Args:
            self: An instance of the LongT5Tokenizer class.
            from_slow (bool): A flag indicating whether to load the tokenizer from a slow source. Defaults to False.

        Returns:
            None: This method does not return any value directly. It loads the tokenizer object for further processing.

        Raises:
            None.
        """
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    @staticmethod
    def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
        """
        This method '_eventually_correct_t5_max_length' is defined in the 'LongT5Tokenizer' class and is used to
        handle the correction of the maximum model length for T5 tokenizer.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model.
                This parameter specifies the model for which the maximum length correction is to be applied.
            max_model_length (int): The maximum model length to be used.
                This parameter represents the maximum allowed input sequence length for the model.
            init_max_model_length (int or None): The initial maximum model length.
                This parameter defines the initial maximum length that may need correction.

        Returns:
            None: This method does not return any value; it modifies the 'max_model_length' parameter in-place.

        Raises:
            FutureWarning: This method may raise a FutureWarning if the tokenizer was incorrectly instantiated with a
                model max length that needs correction. The warning provides guidance on how to avoid the warning and
                properly handle the model max length.
            Warning: This method may raise a generic Warning if the 'init_max_model_length' is not None and does not
                match the 'max_model_length', indicating a potential issue with the maximum model length.

        """
        if pretrained_model_name_or_path in LongT5Tokenizer.max_model_input_sizes:
            deprecated_max_model_length = LongT5Tokenizer.max_model_input_sizes[pretrained_model_name_or_path]
            if init_max_model_length is not None and init_max_model_length != max_model_length:
                return init_max_model_length
            if init_max_model_length is None:
                warnings.warn(
                    "This tokenizer was incorrectly instantiated with a model max length of"
                    f" {deprecated_max_model_length} which will be corrected in Transformers v5.\nFor now, this"
                    " behavior is kept to avoid breaking backwards compatibility when padding/encoding with"
                    " `truncation is True`.\n- Be aware that you SHOULD NOT rely on"
                    f" {pretrained_model_name_or_path} automatically truncating your input to"
                    f" {deprecated_max_model_length} when padding/encoding.\n- If you want to encode/pad to sequences"
                    f" longer than {deprecated_max_model_length} you can either instantiate this tokenizer with"
                    " `model_max_length` or pass `max_length` when encoding/padding.\n- To avoid this warning, please"
                    " instantiate this tokenizer with `model_max_length` set to your preferred value.",
                    FutureWarning,
                )

        return max_model_length

    @property
    def vocab_size(self):
        """
        Method to retrieve the vocabulary size of the LongT5Tokenizer.

        Args:
            self (LongT5Tokenizer): An instance of the LongT5Tokenizer class.
                Represents the tokenizer object.

        Returns:
            int: The vocabulary size of the tokenizer retrieved from the sp_model.

        Raises:
            None.
        """
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """
        Retrieves the vocabulary dictionary used by the LongT5Tokenizer.

        Args:
            self (LongT5Tokenizer): An instance of the LongT5Tokenizer class.

        Returns:
            dict: The vocabulary dictionary containing token-to-index mappings.
                The keys are tokens (str) and the values are their respective indices (int).

        Raises:
            None.

        Note:
            The method combines the default vocabulary dictionary generated from the `vocab_size` parameter and
            any additional tokens that have been added using the `add_tokens` method. The additional tokens
            are included in the vocabulary dictionary with their respective indices.

        Example:
            ```python
            >>> tokenizer = LongT5Tokenizer()
            >>> vocab = tokenizer.get_vocab()
            >>> vocab
            {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, '<extra_id_0>': 4, '<extra_id_1>': 5, ...}
            ```
            In this example, the vocabulary dictionary contains the default tokens as well as any additional tokens
            that have been added to the tokenizer.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

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

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def get_sentinel_tokens(self):
        """
            Retrieves sentinel tokens from the additional special tokens of the LongT5Tokenizer class.

            Args:
                self: An instance of the LongT5Tokenizer class.

            Returns:
                list: a list of sentinel tokens found in the additional special tokens of the tokenizer.

            Raises:
                None.

            Example:
                ```python
                >>> tokenizer = LongT5Tokenizer()
                >>> tokens = tokenizer.get_sentinel_tokens()
                ```
        """
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)) is not None, self.additional_special_tokens))
        )

    def get_sentinel_token_ids(self):
        """
        Returns a list of token IDs corresponding to the sentinel tokens in the input sequence.

        Args:
            self (LongT5Tokenizer): An instance of the LongT5Tokenizer class.

        Returns:
            list: A list of integer values representing the token IDs of the sentinel tokens.

        Raises:
            None

        This method retrieves the sentinel tokens from the input sequence using the 'get_sentinel_tokens' method
        and converts each token into its corresponding token ID using the 'convert_tokens_to_ids' method.
        The resulting token IDs are then returned as a list.

        Note:
            - The 'get_sentinel_tokens' method should be implemented in the 'LongT5Tokenizer' class.
            - The 'convert_tokens_to_ids' method should be implemented in the same class or inherited from a parent class.
        """
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        token_ids_1 = self._add_eos_if_not_present(token_ids_1)
        return token_ids_0 + token_ids_1

    def __getstate__(self):
        """
        __getstate__

        Method in the class 'LongT5Tokenizer' that returns a picklable representation of the object's state,
        excluding the 'sp_model' attribute.

        Args:
            self: An instance of the 'LongT5Tokenizer' class.

        Returns:
            None: The method does not explicitly return a value, but it modifies and returns the object's state.

        Raises:
            None.
        """
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        """
        This method '__setstate__' in the class 'LongT5Tokenizer' allows for setting the state of the tokenizer object.

        Args:
            self (object): The instance of the LongT5Tokenizer class.
            d (dict): A dictionary containing the state information to be set on the tokenizer object.
                It should include attributes that represent the state of the tokenizer.

        Returns:
            None.

        Raises:
            None: However, potential exceptions could be raised during the execution of the method if there are issues
            related to setting the state attributes or loading the vocab file using SentencePieceProcessor.
            It is recommended to handle exceptions related to attribute assignment or file loading gracefully
            in the surrounding code.
        """
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    @property
    def unk_token_length(self):
        """
        This method returns the length of the encoded unknown token in the LongT5Tokenizer.
        
        Args:
            self (object):
                An instance of the LongT5Tokenizer class.

                - Purpose: This parameter refers to the instance of the LongT5Tokenizer class,
                allowing access to its attributes and methods.
                - Restrictions: This parameter is mandatory for the method to operate correctly.
        
        Returns:
            int: The length of the encoded unknown token.
                Purpose: This method returns the length of the encoded unknown token.
                
        Raises:
            None.
        """
        return len(self.sp_model.encode(str(self.unk_token)))

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        tokens = self.sp_model.encode(text, out_type=str)
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        # since we manually add the prefix space, we have to remove it
        tokens[0] = tokens[0].lstrip(SPIECE_UNDERLINE)
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

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the vocabulary files to the specified directory with an optional filename prefix.
        
        Args:
            self (LongT5Tokenizer): The instance of the LongT5Tokenizer class.
            save_directory (str): The directory path where the vocabulary files will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the filename. Defaults to None.
        
        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.
        
        Raises:
            ValueError: If the save_directory is not a valid directory path.
            IOError: If an error occurs while copying or writing the vocabulary file.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

__all__ = ['LongT5Tokenizer']
