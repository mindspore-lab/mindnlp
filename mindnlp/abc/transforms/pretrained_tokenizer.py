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
# pylint: disable=E1121
"""
Cell mixin
"""

import os
from typing import Union, List, Optional
from mindspore import log as logger
from mindspore.dataset.transforms.transforms import PyTensorOperation

from tokenizers import AddedToken, Tokenizer

from mindnlp.utils.download import cached_path
from mindnlp.abc.mixins import SpecialTokensMixin

class PreTrainedTokenizer(SpecialTokensMixin, PyTensorOperation):
    """
    Pretrained Tokenizer abstract class.
    """

    _tokenizer: Tokenizer = None

    def __init__(self, **kwargs):
       # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        """from_pretrained"""
        cache_dir = kwargs.pop("cache_dir", None)
        _ = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        # Get files from url, cache, or disk depending on the case
        # Load tokenizer
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_vocab_map:
                archive_file = cls.pretrained_vocab_map[pretrained_model_name_or_path]
                folder_name = pretrained_model_name_or_path
            elif os.path.isdir(pretrained_model_name_or_path):
                archive_file = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            else:
                raise ValueError(f'not found model of {pretrained_model_name_or_path}.')

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = str(cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    folder_name=folder_name
                )[0])
            except EnvironmentError as exc:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = f"Couldn't reach server at '{archive_file}' to download pretrained weights."
                else:
                    format1 = ", ".join(cls.pretrained_model_archive_map.keys())
                    format2 = ["tokenizer.json"]
                    msg = (
                        f"Model name '{pretrained_model_name_or_path}' "
                        f"was not found in model name list ({format1}). "
                        f"We assumed '{archive_file}' "
                        f"was a path or url to model weight files named one of {format2} but "
                        f"couldn't find any such file at this path or url."
                    )
                raise EnvironmentError(msg) from exc

            if resolved_archive_file == archive_file:
                logger.info("loading tokenizer file %s", archive_file)
            else:
                logger.info("loading tokenizer file %s from cache at %s", archive_file, resolved_archive_file)
        else:
            raise ValueError("the argument 'pretrained_model_name_or_path' should be "
                             "a string of model name or checkpoint path, but got `None`.")

        return cls(resolved_archive_file, *init_inputs, **kwargs)


    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:

        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def encode(self, text_input):
        """encode funtion"""
        tokens = self._tokenizer.encode(text_input)
        return tokens

    def decode(self, ids:list):
        """decode function"""
        return self._tokenizer.decode(ids)

    def id_to_token(self, index: int) -> Optional[str]:
        """index to token."""
        return self._tokenizer.id_to_token(int(index))

    def token_to_id(self, token: str):
        """token to index."""
        return self._convert_token_to_id_with_added_voc(token)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def tokenize(self):
        """tokenize."""

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)
