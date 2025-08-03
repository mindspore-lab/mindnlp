# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for DeepSeek Coder."""

from typing import List, Optional, Union

from mindnlp.transformers.models.llama import LlamaTokenizerFast


class DeepseekCoderTokenizerFast(LlamaTokenizerFast):
    """
    Construct a "fast" DeepSeek Coder tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`LlamaTokenizerFast`].

    For more details, check the doc on HuggingFace's website.
    """

    vocab_files_names = LlamaTokenizerFast.vocab_files_names
    pretrained_vocab_files_map = {}
    max_model_input_sizes = {}
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None
    padding_side = "left"

    def __init__(self, *args, **kwargs):
        kwargs.pop("legacy", None)
        super().__init__(*args, legacy=False, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> List[str]:
        return super().save_vocabulary(save_directory, filename_prefix=filename_prefix)


class DeepseekCoderTokenizer(DeepseekCoderTokenizerFast):
    """
    Construct a DeepSeek Coder tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from [`DeepseekCoderTokenizerFast`].

    For more details, check the doc on HuggingFace's website.
    """

    def convert_ids_to_tokens(
            self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            token = self._tokenizer.id_to_token(index)
            tokens.append(token if token is not None else "")
        return tokens

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        token = self._tokenizer.id_to_token(int(index))
        return token if token is not None else ""


__all__ = ["DeepseekCoderTokenizer",
           "DeepseekCoderTokenizerFast"] 