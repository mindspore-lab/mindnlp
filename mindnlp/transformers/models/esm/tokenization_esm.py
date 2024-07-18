# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ESM."""
import os
from typing import List, Optional

from mindnlp.utils import logging
from ...tokenization_utils import PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/esm2_t6_8M_UR50D": "https://hf-mirror.com/facebook/esm2_t6_8M_UR50D/resolve/main/vocab.txt",
        "facebook/esm2_t12_35M_UR50D": "https://hf-mirror.com/facebook/esm2_t12_35M_UR50D/resolve/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/esm2_t6_8M_UR50D": 1024,
    "facebook/esm2_t12_35M_UR50D": 1024,
}


def load_vocab_file(vocab_file):
    """
    Loads a vocabulary file and returns a list of stripped lines.
    
    Args:
        vocab_file (str): The path of the vocabulary file to be loaded.
    
    Returns:
        list: A list of strings representing each line in the vocabulary file,
            with leading and trailing whitespaces removed.
    
    Raises:
        FileNotFoundError: If the specified vocabulary file does not exist.
        PermissionError: If there is a permission issue with accessing the vocabulary file.
        OSError: If there is an error while reading the vocabulary file.
    """
    with open(vocab_file, "r", encoding='utf-8') as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]


class EsmTokenizer(PreTrainedTokenizer):
    """
    Constructs an ESM tokenizer.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        **kwargs,
    ):
        """
        Initializes an instance of the EsmTokenizer class.
        
        Args:
            self: The instance of the class itself.
            vocab_file (str): The path to the vocabulary file.
            unk_token (str, optional): The token to represent unknown words. Defaults to '<unk>'.
            cls_token (str, optional): The token to represent the start of a sequence. Defaults to '<cls>'.
            pad_token (str, optional): The token to represent padding. Defaults to '<pad>'.
            mask_token (str, optional): The token to represent masked values. Defaults to '<mask>'.
            eos_token (str, optional): The token to represent the end of a sequence. Defaults to '<eos>'.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

        # TODO, all the tokens are added? But they are also part of the vocab... bit strange.
        # none of them are special, but they all need special splitting.

        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an index to a token using the mapping stored in the EsmTokenizer instance.
        
        Args:
            self (EsmTokenizer): The instance of the EsmTokenizer class.
                This parameter is used to access the mapping between indices and tokens.
            index (int): The index of the token to be converted.
                This parameter specifies the index of the token for which the conversion is needed.
                It must be an integer representing the position of the token in the mapping.
        
        Returns:
            str: The token corresponding to the provided index.
                Returns the token associated with the provided index in the mapping.
                If the index is not found in the mapping, the method returns the unknown token (unk_token).
        
        Raises:
            None
        """
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token to its corresponding ID using the provided token string.
        
        Args:
            self (EsmTokenizer): An instance of the EsmTokenizer class.
            token (str): The token to be converted to its corresponding ID.
        
        Returns:
            int: The ID corresponding to the input token. If the token is not found in the token-to-ID mapping, 
                 the ID corresponding to the unknown token (unk_token) is returned as a fallback.
        
        Raises:
            None
        """
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        """
        Method _tokenize in the EsmTokenizer class tokenizes the input text.
        
        Args:
            self (EsmTokenizer): An instance of the EsmTokenizer class.
            text (str): The input text to be tokenized.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return text.split()

    def get_vocab(self):
        """
        Method to retrieve the vocabulary from the EsmTokenizer instance.
        
        Args:
            self (EsmTokenizer):
                The EsmTokenizer instance itself.

                - Type: EsmTokenizer object
                - Purpose: Represents the current instance of the EsmTokenizer class.

        Returns:
            dict:
                A dictionary containing the combined vocabulary.

                - Type: dict
                - Purpose: Represents the vocabulary with the base vocabulary and any added tokens.
        
        Raises:
            None.
        """
        base_vocab = self._token_to_id.copy()
        base_vocab.update(self.added_tokens_encoder)
        return base_vocab

    def token_to_id(self, token: str) -> int:
        """
        Method to retrieve the ID corresponding to a given token from the EsmTokenizer instance.
        
        Args:
            self (EsmTokenizer): The EsmTokenizer instance on which the method is called.
            token (str): The input token for which the corresponding ID needs to be retrieved. It should be a string.
        
        Returns:
            int: Returns the ID corresponding to the input token from the EsmTokenizer instance.
                If the token is not found in the internal token-to-ID mapping,
                the method returns the ID associated with the unknown token (unk_token) if defined.
        
        Raises:
            None
        """
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        """
        Retrieve the token associated with the provided index from the EsmTokenizer.
        
        Args:
            self (EsmTokenizer): The instance of the EsmTokenizer class.
            index (int): The index of the token to retrieve.
                Must be a non-negative integer corresponding to a valid token index.
        
        Returns:
            str: The token associated with the provided index.
                If the index is not found in the mapping, the unknown token (unk_token) is returned.
        
        Raises:
            None
        """
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        This method builds inputs with special tokens for the EsmTokenizer class.
        
        Args:
            self: The instance of the EsmTokenizer class.
            token_ids_0 (List[int]): List of token IDs for the first sequence.
            token_ids_1 (Optional[List[int]]): List of token IDs for the second sequence, if present. Defaults to None.
        
        Returns:
            List[int]: A list of token IDs representing the input sequences with special tokens added.
        
        Raises:
            ValueError: Raised if token_ids_1 is not None and self.eos_token_id is None,
                indicating that multiple sequences cannot be tokenized when the EOS token is not set.
        """
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]  # No sep token in ESM vocabulary
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            return cls + token_ids_0 + sep
        if self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + sep  # Multiple inputs always have an EOS token

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, filename_prefix):
        """
        Save the vocabulary to a text file.
        
        Args:
            self (EsmTokenizer): The instance of the EsmTokenizer class.
            save_directory (str): The directory path where the vocabulary file will be saved.
            filename_prefix (str): A prefix to be added to the vocabulary file name. If None, no prefix is added.
        
        Returns:
            None.
        
        Raises:
            FileNotFoundError: If the specified save_directory does not exist.
            PermissionError: If the method does not have permission to write to the save_directory.
            OSError: If an error occurs while opening or writing to the vocabulary file.
        """
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w", encoding='utf-8') as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        """
        This method, vocab_size, in the class EsmTokenizer calculates the size of the vocabulary based on the
        number of unique tokens present.
        
        Args:
            self (EsmTokenizer): The instance of the EsmTokenizer class.
                This parameter represents the current instance of the EsmTokenizer class.
        
        Returns:
            int: The method returns an integer value representing the size of the vocabulary, which is determined
                by the number of unique tokens present in the instance.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return len(self.all_tokens)

__all__ = ["EsmTokenizer"]
