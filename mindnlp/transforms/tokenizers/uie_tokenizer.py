# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
UIE Tokenizer
"""

import re
from typing import Optional
import numpy as np
from tokenizers import Tokenizer
from mindspore.dataset.text.transforms import Implementation
from mindnlp.abc import PreTrainedTokenizer
from mindnlp.models.ernie.ernie_config import ERNIE_SUPPORT_LIST
from mindnlp.configs import MINDNLP_TOKENIZER_CONFIG_URL_BASE

PRETRAINED_VOCAB_MAP = {
    model: MINDNLP_TOKENIZER_CONFIG_URL_BASE.format(
        re.search(r"^[^-]*", model).group(), model
    )
    for model in ERNIE_SUPPORT_LIST
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "uie-base": 2048,
    "uie-medium": 2048,
    "uie-mini": 2048,
    "uie-micro": 2048,
    "uie-nano": 2048,
    "uie-base-en": 512,
    "uie-senta-base": 2048,
    "uie-senta-medium": 2048,
    "uie-senta-mini": 2048,
    "uie-senta-micro": 2048,
    "uie-senta-nano": 2048,
    "uie-base-answer-extractor": 2048,
    "uie-base-qa-filter": 2048,
}


class UIETokenizer(PreTrainedTokenizer):
    """
    Tokenizer used for UIE text process.

    Args:
        vocab (Vocab): Vocabulary used to look up words.
        return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.

    """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP

    def __init__(self, vocab: str, **kwargs):
        return_token = kwargs.pop("return_token", False)

        if isinstance(vocab, str):
            self._tokenizer = Tokenizer.from_file(vocab)
        else:
            raise ValueError(f"only support string, but got {vocab}")
        self.return_token = return_token
        self.implementation = Implementation.PY

        super().__init__(**kwargs)

    def __call__(
        self,
        text_input,
        pair=None,
        max_length: Optional[int] = None,
        truncation: bool = None,
        padding: bool = False,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_offsets_mapping: bool = False,
        return_position_ids: bool = False,
    ):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if isinstance(text_input, str):
            text_input = np.array(text_input)
        elif not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}."
            )
        return self._execute_py(
            text_input,
            pair,
            max_length,
            truncation,
            padding,
            return_token_type_ids,
            return_attention_mask,
            return_offsets_mapping,
            return_position_ids,
        )

    def execute_py(
        self,
        text_input,
        pair,
        max_length,
        truncation,
        padding,
        return_token_type_ids,
        return_attention_mask,
        return_offsets_mapping,
        return_position_ids,
    ):
        """
        Execute method.
        """
        return self._execute_py(
            text_input,
            pair,
            max_length,
            truncation,
            padding,
            return_token_type_ids,
            return_attention_mask,
            return_offsets_mapping,
            return_position_ids,
        )

    def _execute_py(
        self,
        text_input,
        pair,
        max_length,
        truncation,
        padding,
        return_token_type_ids,
        return_attention_mask,
        return_offsets_mapping,
        return_position_ids,
    ):
        """
        Execute method.
        """

        encoded_inputs = {}

        text_input = self._convert_to_unicode(text_input)
        pair = self._convert_to_unicode(pair)
        if return_position_ids is True:
            self._tokenizer.no_padding()
            self._tokenizer.no_truncation()
            ids = self._tokenizer.encode(text_input, pair=pair).ids
            pos_ids = list(range(len(ids))) + [0] * (max_length - len(ids))
            encoded_inputs["position_ids"] = np.array(pos_ids)

        if padding is True:
            self._tokenizer.enable_padding(length=max_length)
        if truncation is True:
            self._tokenizer.enable_truncation(max_length=max_length)

        tokens = self._tokenizer.encode(text_input, pair=pair)

        if return_token_type_ids is True:
            encoded_inputs["token_type_ids"] = np.array(tokens.type_ids)
        if return_attention_mask is True:
            encoded_inputs["attention_mask"] = np.array(tokens.attention_mask)
        if return_offsets_mapping is True:
            encoded_inputs["offset_mapping"] = np.array(tokens.offsets)

        encoded_inputs["input_ids"] = np.array(tokens.ids)

        return encoded_inputs

    def _convert_to_unicode(self, text_input):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text_input, str):
            return text_input
        if isinstance(text_input, bytes):
            return text_input.decode("utf-8", "ignore")
        if isinstance(text_input, np.ndarray):
            if text_input.dtype.type is np.bytes_:
                text_input = np.char.decode(text_input, "utf-8")
            return str(text_input)
        raise ValueError(
            f"Unsupported string type: {type(text_input)}, {text_input.dtype}")

    def _convert_token_to_id(self, token):
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index
