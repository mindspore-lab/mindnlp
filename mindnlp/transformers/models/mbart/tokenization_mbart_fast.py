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
"""Fast Tokenization classes for MBart."""

import os
from shutil import copyfile
from typing import List, Optional, Tuple

from tokenizers import processors

from mindnlp.utils import is_sentencepiece_available, logging
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast

if is_sentencepiece_available():
    from .tokenization_mbart import MBartTokenizer
else:
    MBartTokenizer = None

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": (
            "https://hf-mirror.com/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model"
        ),
        "facebook/mbart-large-cc25": (
            "https://hf-mirror.com/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "facebook/mbart-large-en-ro": "https://hf-mirror.com/facebook/mbart-large-en-ro/resolve/main/tokenizer.json",
        "facebook/mbart-large-cc25": "https://hf-mirror.com/facebook/mbart-large-cc25/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}

FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
                          "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
                          "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]  # fmt: skip


class MBartTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" MBART tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [BPE](https://hf-mirror.com/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Example:
        ```python
        >>> from transformers import MBartTokenizerFast
        ...
        >>> tokenizer = MBartTokenizerFast.from_pretrained(
        ...     "facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO"
        ... )
        >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
        >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="ms")
        ```
    """
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = MBartTokenizer

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

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
            src_lang=None,
            tgt_lang=None,
            additional_special_tokens=None,
            **kwargs,
    ):
        """
        This method initializes an instance of the MBartTokenizerFast class.

        Args:
            self: The instance of the class.
            vocab_file (str, optional): The vocabulary file. Defaults to None.
            tokenizer_file (str, optional): The tokenizer file. Defaults to None.
            bos_token (str, optional): The beginning of sentence token. Defaults to '<s>'.
            eos_token (str, optional): The end of sentence token. Defaults to '</s>'.
            sep_token (str, optional): The separator token. Defaults to '</s>'.
            cls_token (str, optional): The classification token. Defaults to '<s>'.
            unk_token (str, optional): The unknown token. Defaults to '<unk>'.
            pad_token (str, optional): The padding token. Defaults to '<pad>'.
            mask_token (str, optional): The mask token. Defaults to '<mask>'.
            src_lang (str, optional): The source language. Defaults to None.
            tgt_lang (str, optional): The target language. Defaults to None.
            additional_special_tokens (list, optional): Additional special tokens. Defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        _additional_special_tokens = FAIRSEQ_LANGUAGE_CODES.copy()

        if additional_special_tokens is not None:
            # Only add those special tokens if they are not already there.
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=_additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        Check if the slow tokenizer can be saved.

        Args:
            self: An instance of the MBartTokenizerFast class.

        Returns:
            A boolean value indicating whether the slow tokenizer can be saved or not.

        Raises:
            None.

        This method checks if the slow tokenizer can be saved by verifying the existence of the vocabulary file specified
        by the 'vocab_file' attribute of the class. If the 'vocab_file' attribute is set and it corresponds to an
        existing file, the method returns True. Otherwise, it returns False.
        """
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @property
    def src_lang(self) -> str:
        """src_lang"""
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        """
        Sets the source language for the MBartTokenizerFast instance.

        Args:
            self (MBartTokenizerFast): The instance of the MBartTokenizerFast class.
            new_src_lang (str): The new source language to be set for the tokenizer.
                It should be a string representing the language code.

        Returns:
            None.

        Raises:
            TypeError: If the new source language provided is not a string.
        """
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. The special tokens depend on calling set_lang.

        An MBART sequence has the following format, where `X` represents the sequence:

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
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
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

    def prepare_seq2seq_batch(
            self,
            src_texts: List[str],
            src_lang: str = "en_XX",
            tgt_texts: Optional[List[str]] = None,
            tgt_lang: str = "ro_RO",
            **kwargs,
    ) -> BatchEncoding:
        """
        Prepare a batch for sequence-to-sequence tokenization using the MBartTokenizerFast class.

        Args:
            self (MBartTokenizerFast): An instance of the MBartTokenizerFast class.
            src_texts (List[str]): A list of source texts to be tokenized.
            src_lang (str, optional): The language code of the source texts. Defaults to 'en_XX'.
            tgt_texts (List[str], optional): A list of target texts to be tokenized. Defaults to None.
            tgt_lang (str, optional): The language code of the target texts. Defaults to 'ro_RO'.
            **kwargs: Additional keyword arguments.

        Returns:
            BatchEncoding: A dictionary-like object that contains the tokenized inputs and their corresponding IDs.

        Raises:
            None: This method does not raise any exceptions.

        Note:
            This method internally calls the prepare_seq2seq_batch method of the base class, passing the necessary parameters.

        Example:
            ```python
            >>> tokenizer = MBartTokenizerFast.from_pretrained('facebook/mbart-large-50')
            >>> src_texts = ['Hello world!', 'How are you?']
            >>> tgt_texts = ['Bonjour le monde!', 'Comment ça va?']
            >>> batch_encodings = tokenizer.prepare_seq2seq_batch(src_texts, src_lang='en_XX', tgt_texts=tgt_texts, tgt_lang='fr_FR')
            ```
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        """
        Method to switch to input mode for the MBartTokenizerFast class.

        Args:
            self (MBartTokenizerFast): The instance of the MBartTokenizerFast class.
                This parameter is required to access the methods and attributes of the class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        """
        Switches the tokenizer to the target mode for the MBartTokenizerFast class.

        Args:
            self: An instance of the MBartTokenizerFast class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method switches the tokenizer to the target mode for the MBartTokenizerFast class. In target mode,
            the tokenizer is configured to tokenize text according to the target language specified during initialization.

        The method takes one parameter, 'self', which refers to an instance of the MBartTokenizerFast class.
        This parameter is required to access the tokenizer instance and perform the necessary operations to
        switch to the target mode.

        The method does not raise any exceptions.

        Example:
            ```python
            >>> tokenizer = MBartTokenizerFast()
            >>> tokenizer._switch_to_target_mode()
            ```
        """
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary for the MBartTokenizerFast class.
        
        Args:
            self: The instance of the MBartTokenizerFast class.
            save_directory (str): The directory where the vocabulary file will be saved.
            filename_prefix (Optional[str]): An optional prefix to include in the vocabulary file name.
            
        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.
        
        Raises:
            ValueError: If the fast tokenizer does not have the necessary information to save the vocabulary
                for a slow tokenizer.
            FileNotFoundError: If the specified save_directory does not exist.
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return None
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)


__all__ = ['MBartTokenizerFast']
