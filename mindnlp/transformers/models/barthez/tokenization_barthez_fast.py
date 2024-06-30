# coding=utf-8
# Copyright 2020 Ecole Polytechnique and the HuggingFace Inc. team.
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
# limitations under the License
""" Tokenization classes for the BARThez model."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from mindnlp.utils import is_sentencepiece_available, logging
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast


if is_sentencepiece_available():
    from .tokenization_barthez import BarthezTokenizer
else:
    BarthezTokenizer = None

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "moussaKam/mbarthez": "https://hf-mirror.com/moussaKam/mbarthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez": "https://hf-mirror.com/moussaKam/barthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez-orangesum-title": (
            "https://hf-mirror.com/moussaKam/barthez-orangesum-title/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "moussaKam/mbarthez": "https://hf-mirror.com/moussaKam/mbarthez/resolve/main/tokenizer.json",
        "moussaKam/barthez": "https://hf-mirror.com/moussaKam/barthez/resolve/main/tokenizer.json",
        "moussaKam/barthez-orangesum-title": (
            "https://hf-mirror.com/moussaKam/barthez-orangesum-title/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "moussaKam/mbarthez": 1024,
    "moussaKam/barthez": 1024,
    "moussaKam/barthez-orangesum-title": 1024,
}

SPIECE_UNDERLINE = "▁"


class BarthezTokenizerFast(PreTrainedTokenizerFast):
    """
    Adapted from [`CamembertTokenizer`] and [`BartTokenizer`]. Construct a "fast" BARThez tokenizer. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
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

        sep_token (`str`, *optional*):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens. defaults to `"</s>"`
        cls_token (`str`, *optional*):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            defaults to `"<s>"`
        unk_token (`str`, *optional*):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. defaults to `"<unk>"`
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths. defaults to `"<pad>"`
        mask_token (`str`, *optional*):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. defaults to `"<mask>"`
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer. defaults to `["<s>NOTUSED", "</s>NOTUSED"]`
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = BarthezTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    ):
        """
        Initialize a BarthezTokenizerFast object.

        Args:
            vocab_file (str): Path to the vocabulary file. Default is None.
            tokenizer_file (str): Path to the tokenizer file. Default is None.
            bos_token (str): Beginning of sentence token. Default is '<s>'.
            eos_token (str): End of sentence token. Default is '</s>'.
            sep_token (str): Separator token. Default is '</s>'.
            cls_token (str): Classification token. Default is '<s>'.
            unk_token (str): Token for unknown words. Default is '<unk>'.
            pad_token (str): Padding token. Default is '<pad>'.
            mask_token (str): Mask token. Default is '<mask>'.

        Returns:
            None.

        Raises:
            TypeError: If mask_token is not a string.
        """
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        Method to check if the slow tokenizer can be saved.

        Args:
            self (BarthezTokenizerFast): An instance of the BarthezTokenizerFast class.
                Represents the current object to check whether the slow tokenizer can be saved.

        Returns:
            bool: Returns a boolean value indicating whether the slow tokenizer can be saved.
                True if the vocab file exists, False if the vocab file does not exist or is not provided.

        Raises:
            None.
        """
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BARThez sequence has the following format:

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

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

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

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary for a slow tokenizer.

        Args:
            self (BarthezTokenizerFast): The instance of the BarthezTokenizerFast class.
            save_directory (str): The directory where the vocabulary will be saved.
            filename_prefix (Optional[str], optional): The prefix to be added to the filename (default: None).

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.

        Raises:
            ValueError: If the fast tokenizer does not have the necessary information to save the vocabulary for a slow tokenizer.
            OSError: If the provided save_directory is not a valid directory.
            IOError: If there is an error while copying the vocabulary file.

        Note:
            - The fast tokenizer must have the necessary information to save the vocabulary for a slow tokenizer.
            - The save_directory should be a valid directory.
            - The vocabulary file will be copied to the save_directory with an optional filename_prefix.

        Example:
            ```python
            >>> tokenizer = BarthezTokenizerFast()
            >>> tokenizer.save_vocabulary('/path/to/save')
            ('/path/to/save/vocab.txt', )
            ```
        
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

__all__ = ['BarthezTokenizerFast']
