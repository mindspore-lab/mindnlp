# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for MBart."""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from mindnlp.utils import logging
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": (
            "https://hf-mirror.com/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model"
        ),
        "facebook/mbart-large-cc25": (
            "https://hf-mirror.com/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model"
        ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}

FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX",
    "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"
]  # fmt: skip


class MBartTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for MBart
    """
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        src_lang=None,
        tgt_lang=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        additional_special_tokens=None,
        **kwargs,
    ):
        """
        This method initializes an instance of the MBartTokenizer class.
        
        Args:
            self: The instance of the class.
            vocab_file (str): The path to the vocabulary file.
            bos_token (str, optional): The beginning of sentence token. Defaults to '<s>'.
            eos_token (str, optional): The end of sentence token. Defaults to '</s>'.
            sep_token (str, optional): The separator token. Defaults to '</s>'.
            cls_token (str, optional): The classification token. Defaults to '<s>'.
            unk_token (str, optional): The unknown token. Defaults to '<unk>'.
            pad_token (str, optional): The padding token. Defaults to '<pad>'.
            mask_token (str, optional): The mask token. Defaults to '<mask>'.
            src_lang (str, optional): The source language. Defaults to None.
            tgt_lang (str, optional): The target language. Defaults to None.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Additional SentencePiece model arguments.
                Defaults to None.
            additional_special_tokens (list, optional): Additional special tokens. Defaults to None.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, normalized=False) if isinstance(mask_token, str) else mask_token
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1

        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        _additional_special_tokens = list(self.lang_code_to_id.keys())

        if additional_special_tokens is not None:
            # Only add those special tokens if they are not already there.
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=_additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def __getstate__(self):
        """
        __getstate__
        
        This method returns the state of the MBartTokenizer object for serialization.
        
        Args:
            self: The instance of the MBartTokenizer class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, _d):
        """
        Sets the state of the MBartTokenizer object during unpickling.
        
        Args:
            self (MBartTokenizer): The instance of the MBartTokenizer class.
            _d (dict): The dictionary containing the state of the object.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.__dict__ = _d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """
        Returns the total vocabulary size of the MBartTokenizer.
        
        Args:
            self (MBartTokenizer): The instance of the MBartTokenizer class.
        
        Returns:
            int: The total vocabulary size of the MBartTokenizer.
            
        Raises:
            None.
        """
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # Plus 1 for the mask token

    @property
    def src_lang(self) -> str:
        """src_lang"""
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        """
        Sets the source language for the MBartTokenizer.
        
        Args:
            self: An instance of the MBartTokenizer class.
            new_src_lang (str): The new source language to be set. It should be a string representing the language code.
        
        Returns:
            None.
        
        Raises:
            None.
        
        Note:
            This method updates the source language attribute (_src_lang) of the MBartTokenizer instance. It also calls the 
            set_src_lang_special_tokens method to update the special tokens related to the new source language.
        """
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

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

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. mBART does not
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

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def get_vocab(self):
        """
        Retrieves the vocabulary of the MBartTokenizer instance.

        Args:
            self (MBartTokenizer): The instance of the MBartTokenizer class.

        Returns:
            dict: A dictionary containing the vocabulary of the tokenizer. The keys are the tokens in the vocabulary,
                and the values are their corresponding token IDs.

        Raises:
            None

        Note:
            The vocabulary includes both the tokens from the original MBart model and any additional tokens that
            have been added using the `add_tokens` method.

        Example:
            ```python
            >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
            >>> vocab = tokenizer.get_vocab()
            >>> print(vocab)
            {'<s>': 0, '</s>': 1, '<pad>': 2, '<unk>': 3, '<mask>': 4, '▁': 5, 'a': 6, 'b': 7, 'c': 8, ...}
            ```
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the given text into a list of strings using the mBART tokenizer.

        Args:
            self: An instance of the MBartTokenizer class.
            text (str): The text to be tokenized.

        Returns:
            List[str]: A list of strings representing the tokenized text.

        Raises:
            None.

        This method takes in a text and tokenizes it into a list of strings using the mBART tokenizer.
        The tokenizer uses the sp_model to encode the text into a string representation.
        The encoded text is then returned as a list of strings.
        """
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary to a file in the specified directory.

        Args:
            self (MBartTokenizer): An instance of the MBartTokenizer class.
            save_directory (str): The directory where the vocabulary file will be saved.
            filename_prefix (Optional[str], default=None): An optional prefix for the filename.

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.

        Raises:
            TypeError: If the save_directory parameter is not a string.
            NotADirectoryError: If the save_directory does not exist or is not a directory.
            FileExistsError: If the output vocabulary file already exists.
            FileNotFoundError: If the input vocabulary file does not exist.
            IOError: If there is an error while copying the input vocabulary file to the output file.
            IOError: If there is an error while writing the serialized model to the output file.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return None
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as f_i:
                content_spiece_model = self.sp_model.serialized_model_proto()
                f_i.write(content_spiece_model)

        return (out_vocab_file,)

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepare and encode a batch of sequences for sequence-to-sequence (seq2seq) model input.

        Args:
            self (MBartTokenizer): The instance of the MBartTokenizer class.
            src_texts (List[str]): A list of source language texts to be encoded.
            src_lang (str): The source language code. Defaults to 'en_XX'.
            tgt_texts (Optional[List[str]]): An optional list of target language texts to be encoded. Defaults to None.
            tgt_lang (str): The target language code. Defaults to 'ro_RO'.
            **kwargs: Additional keyword arguments to be passed to the superclass method.

        Returns:
            BatchEncoding: A batch encoding object containing the encoded source and target sequences.

        Raises:
            None: This method does not raise any exceptions.
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        """
        Switch to input mode by setting the source language special tokens.

        Args:
            self (MBartTokenizer): An instance of the MBartTokenizer class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        """
        Switches the tokenizer to the target mode by setting the target language special tokens.

        Args:
            self (MBartTokenizer): An instance of the MBartTokenizer class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method is used to switch the tokenizer to the target mode. It sets the target language special tokens
            using the `set_tgt_lang_special_tokens` method with the target language specified during initialization.

        The `self` parameter is the instance of the MBartTokenizer class that the method is called on.

        Example:
            ```python
            >>> tokenizer = MBartTokenizer()
            >>> tokenizer._switch_to_target_mode()
            ```
        """
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]


__all__ = ['MBartTokenizer']
