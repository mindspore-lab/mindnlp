# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
""" Tokenization class for model BertGeneration."""


import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from mindnlp.utils import logging
from ...tokenization_utils import PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class BertGenerationTokenizer(PreTrainedTokenizer):
    """
    Construct a BertGeneration tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*):
            The begin of sequence token, defaults to `"<s>"`
        eos_token (`str`, *optional*):
            The end of sequence token, defaults to `"</s>"`
        unk_token (`str`, *optional*):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead, defaults to `"<unk>"`
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths, defaults to `"<pad>"`
        sep_token (`str`, *optional*):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens, defaults to `"<::::>"`
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
    prefix_tokens: List[int] = []
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token="<::::>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes a BertGenerationTokenizer object.

        Args:
            vocab_file (str): The path to the vocabulary file containing the token mappings.
            bos_token (str, optional): The Beginning of Sentence token. Default is '<s>'.
            eos_token (str, optional): The End of Sentence token. Default is '</s>'.
            unk_token (str, optional): The token representing unknown words. Default is '<unk>'.
            pad_token (str, optional): The token used for padding sequences. Default is '<pad>'.
            sep_token (str, optional): The token used for separating different segments. Default is '<::::>'.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for SentencePieceProcessor. Default is None.

        Returns:
            None.

        Raises:
            TypeError: If the vocab_file is not a valid string path or if sp_model_kwargs is not a valid dictionary.
            OSError: If the vocab_file cannot be loaded or accessed.
            ValueError: If any of the default tokens are not valid strings.
        """
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # Add extra_ids to the special token list
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary used by the BertGenerationTokenizer instance.

        Args:
            self (BertGenerationTokenizer): The current instance of the BertGenerationTokenizer class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """
        Returns the vocabulary of the BertGenerationTokenizer.

        Args:
            self (BertGenerationTokenizer): An instance of the BertGenerationTokenizer class.

        Returns:
            dict:
                A dictionary representing the vocabulary. The keys are tokens (words, subwords, or special tokens)
                and the values are their corresponding token IDs.

        Raises:
            None.

        Note:
            - The vocabulary includes both the original vocabulary from the pre-trained model and any additional tokens added
            using the 'add_tokens' method.
            - The token IDs range from 0 to vocab_size - 1, where vocab_size is the total number of tokens in the vocabulary.

        Example:
            ```python:
            >>> tokenizer = BertGenerationTokenizer()
            >>> vocab = tokenizer.get_vocab()
            >>> print(vocab)
            {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'hello': 5, 'world': 6}
            ```
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        """
        Method __getstate__ in the class BertGenerationTokenizer.
        
        This method returns the state of the object which is a dictionary containing the object's attributes. The 'sp_model' attribute is set to None before returning the state.
        
        Args:
            self: Instance of the BertGenerationTokenizer class.
        
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
        Set the state of the BertGenerationTokenizer object.
        
        Args:
            self (BertGenerationTokenizer): The instance of the BertGenerationTokenizer class.
            d (dict): A dictionary containing the state information to be set for the object.
        
        Returns:
            None
        
        Raises:
            None
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

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary files to the specified directory.
        
        Args:
            self: The instance of the BertGenerationTokenizer class.
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the vocabulary file name.
                Defaults to None.
        
        Returns:
            Tuple[str]: A tuple containing the path of the saved vocabulary file.
        
        Raises:
            OSError: If the save_directory does not exist or is not a directory.
            IOError: If there is an issue with file operations such as copying or writing the vocabulary file.
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

__all__ = ['BertGenerationTokenizer']
