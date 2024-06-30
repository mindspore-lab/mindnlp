# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""tokenization baichuan"""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from mindnlp.utils import logging
from mindnlp.transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "tokenizer_file": {},
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


class BaiChuanTokenizer(PreTrainedTokenizer):
    """
    Construct a BaiChuan tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        """
        __init__
        
        Initializes a new instance of the BaiChuanTokenizer class.
        
        Args:
            self: The instance of the class.
            vocab_file (str): The path to the vocabulary file.
            unk_token (str, optional): The unknown token. Defaults to '<unk>'.
            bos_token (str, optional): The beginning of sentence token. Defaults to '<s>'.
            eos_token (str, optional): The end of sentence token. Defaults to '</s>'.
            pad_token (str, optional): The padding token. Defaults to None.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Optional arguments for SentencePieceProcessor. Defaults to None.
            add_bos_token (bool, optional): Whether to add the beginning of sentence token. Defaults to True.
            add_eos_token (bool, optional): Whether to add the end of sentence token. Defaults to False.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Defaults to False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None.
        
        Raises:
            TypeError: If vocab_file is not a string.
            TypeError: If unk_token is not a string.
            TypeError: If bos_token is not a string.
            TypeError: If eos_token is not a string.
            TypeError: If pad_token is not a string.
            TypeError: If sp_model_kwargs is not a dictionary or None.
            TypeError: If add_bos_token is not a boolean.
            TypeError: If add_eos_token is not a boolean.
            TypeError: If clean_up_tokenization_spaces is not a boolean.
        """
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def __getstate__(self):
        """Return a dictionary representing the state of the BaiChuanTokenizer instance.
        
        This method takes no additional parameters.
        
        Args:
            self: The instance of the BaiChuanTokenizer class.
        
        Returns:
            None: This method does not return any value. It modifies the state of the instance in-place.
        
        Raises:
            None.
        """
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        """
        __setstate__ method in the BaiChuanTokenizer class.
        
        Args:
            self (BaiChuanTokenizer): An instance of the BaiChuanTokenizer class.
                Represents the current object that the method is called on.
            d (dict): A dictionary containing the state information to be set.
                This dictionary is used to update the internal state of the object.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If there is an issue with the values passed as parameters.
            AttributeError: If there are issues related to attribute access or assignment.
            RuntimeError: If there is a runtime issue during the method execution.
        """
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

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
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
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

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Builds input sequences with special tokens for the BaiChuanTokenizer class.
        
        Args:
            self (BaiChuanTokenizer): An instance of the BaiChuanTokenizer class.
            token_ids_0 (List[int]): A list of token IDs representing the first input sequence.
            token_ids_1 (List[int], optional): A list of token IDs representing the second input sequence. Defaults to None.
        
        Returns:
            List[int]: A list of token IDs representing the input sequences with special tokens added.
        
        Raises:
            None.
        
        This method takes in two input sequences token_ids_0 and token_ids_1 (optional) and builds input sequences with special tokens.
        If the add_bos_token parameter is True, the method adds the beginning of sequence token (bos_token) at the beginning of the sequences.
        If the add_eos_token parameter is True, the method adds the end of sequence token (eos_token) at the end of the sequences.
        The method then concatenates the special tokens with the input sequences and returns the result.
        
        If a second input sequence (token_ids_1) is provided, the method also adds special tokens to it and concatenates it
        with the first input sequence.
        
        Note:
            The bos_token_id and eos_token_id are specific token IDs for the beginning and end of sequence tokens, respectively,
            as defined in the BaiChuanTokenizer class.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

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

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

__all__ = ['BaiChuanTokenizer']
