# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Fast Tokenization classes for Bert."""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

from mindnlp.utils import logging
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt",
        "bert-large-uncased": "https://hf-mirror.com/bert-large-uncased/resolve/main/vocab.txt",
        "bert-base-cased": "https://hf-mirror.com/bert-base-cased/resolve/main/vocab.txt",
        "bert-large-cased": "https://hf-mirror.com/bert-large-cased/resolve/main/vocab.txt",
        "bert-base-multilingual-uncased": (
            "https://hf-mirror.com/bert-base-multilingual-uncased/resolve/main/vocab.txt"
        ),
        "bert-base-multilingual-cased": "https://hf-mirror.com/bert-base-multilingual-cased/resolve/main/vocab.txt",
        "bert-base-chinese": "https://hf-mirror.com/bert-base-chinese/resolve/main/vocab.txt",
        "bert-base-german-cased": "https://hf-mirror.com/bert-base-german-cased/resolve/main/vocab.txt",
        "bert-large-uncased-whole-word-masking": (
            "https://hf-mirror.com/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"
        ),
        "bert-large-cased-whole-word-masking": (
            "https://hf-mirror.com/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"
        ),
        "bert-large-uncased-whole-word-masking-finetuned-squad": (
            "https://hf-mirror.com/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        "bert-large-cased-whole-word-masking-finetuned-squad": (
            "https://hf-mirror.com/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        "bert-base-cased-finetuned-mrpc": (
            "https://hf-mirror.com/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
        ),
        "bert-base-german-dbmdz-cased": "https://hf-mirror.com/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
        "bert-base-german-dbmdz-uncased": (
            "https://hf-mirror.com/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"
        ),
        "TurkuNLP/bert-base-finnish-cased-v1": (
            "https://hf-mirror.com/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"
        ),
        "TurkuNLP/bert-base-finnish-uncased-v1": (
            "https://hf-mirror.com/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"
        ),
        "wietsedv/bert-base-dutch-cased": (
            "https://hf-mirror.com/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "bert-base-uncased": "https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer.json",
        "bert-large-uncased": "https://hf-mirror.com/bert-large-uncased/resolve/main/tokenizer.json",
        "bert-base-cased": "https://hf-mirror.com/bert-base-cased/resolve/main/tokenizer.json",
        "bert-large-cased": "https://hf-mirror.com/bert-large-cased/resolve/main/tokenizer.json",
        "bert-base-multilingual-uncased": (
            "https://hf-mirror.com/bert-base-multilingual-uncased/resolve/main/tokenizer.json"
        ),
        "bert-base-multilingual-cased": (
            "https://hf-mirror.com/bert-base-multilingual-cased/resolve/main/tokenizer.json"
        ),
        "bert-base-chinese": "https://hf-mirror.com/bert-base-chinese/resolve/main/tokenizer.json",
        "bert-base-german-cased": "https://hf-mirror.com/bert-base-german-cased/resolve/main/tokenizer.json",
        "bert-large-uncased-whole-word-masking": (
            "https://hf-mirror.com/bert-large-uncased-whole-word-masking/resolve/main/tokenizer.json"
        ),
        "bert-large-cased-whole-word-masking": (
            "https://hf-mirror.com/bert-large-cased-whole-word-masking/resolve/main/tokenizer.json"
        ),
        "bert-large-uncased-whole-word-masking-finetuned-squad": (
            "https://hf-mirror.com/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
        ),
        "bert-large-cased-whole-word-masking-finetuned-squad": (
            "https://hf-mirror.com/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
        ),
        "bert-base-cased-finetuned-mrpc": (
            "https://hf-mirror.com/bert-base-cased-finetuned-mrpc/resolve/main/tokenizer.json"
        ),
        "bert-base-german-dbmdz-cased": (
            "https://hf-mirror.com/bert-base-german-dbmdz-cased/resolve/main/tokenizer.json"
        ),
        "bert-base-german-dbmdz-uncased": (
            "https://hf-mirror.com/bert-base-german-dbmdz-uncased/resolve/main/tokenizer.json"
        ),
        "TurkuNLP/bert-base-finnish-cased-v1": (
            "https://hf-mirror.com/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/tokenizer.json"
        ),
        "TurkuNLP/bert-base-finnish-uncased-v1": (
            "https://hf-mirror.com/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/tokenizer.json"
        ),
        "wietsedv/bert-base-dutch-cased": (
            "https://hf-mirror.com/wietsedv/bert-base-dutch-cased/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}


class BertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" BERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BertTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        """
        Initialize the BertTokenizerFast class.
        
        Args:
            self: The instance of the class.
            vocab_file (str): The file path to the vocabulary file. Defaults to None.
            tokenizer_file (str): The file path to the tokenizer file. Defaults to None.
            do_lower_case (bool): Flag indicating whether to convert tokens to lowercase. Defaults to True.
            unk_token (str): The special token for unknown tokens. Defaults to '[UNK]'.
            sep_token (str): The special token for separating sequences. Defaults to '[SEP]'.
            pad_token (str): The special token for padding sequences. Defaults to '[PAD]'.
            cls_token (str): The special token for classifying sequences. Defaults to '[CLS]'.
            mask_token (str): The special token for masking tokens. Defaults to '[MASK]'.
            tokenize_chinese_chars (bool): Flag indicating whether to tokenize Chinese characters. Defaults to True.
            strip_accents (str or None): Flag indicating whether to strip accents. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None.
        
        Raises:
            Exception: If an error occurs during the initialization process.
        """
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

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
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

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
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the BertTokenizerFast model to the specified directory.
        
        Args:
            self (BertTokenizerFast): The instance of the BertTokenizerFast class.
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (Optional[str]): An optional prefix for the saved vocabulary files. Defaults to None.
        
        Returns:
            Tuple[str]: A tuple containing the names of the saved files.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

__all__ = ['BertTokenizerFast']
