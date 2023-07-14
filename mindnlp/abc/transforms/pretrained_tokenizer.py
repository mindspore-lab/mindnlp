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
from typing import Union, List, Optional, Dict
from mindspore import log as logger
from mindspore.dataset.transforms.transforms import PyTensorOperation

from tokenizers import AddedToken, Tokenizer

from mindnlp.configs import DEFAULT_ROOT
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
        # Added tokens - We store this for both slow and fast tokenizers
        # until the serialization of Fast tokenizers is updated
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []

        self._decode_use_source_tokenizer = False

        # By default, cleaning tokenization spaces for both fast and slow tokenizers
        self.clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        """from_pretrained"""
        cache_dir = kwargs.pop("cache_dir", os.path.join(DEFAULT_ROOT, 'models'))
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

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        return text

    def id_to_token(self, index: int) -> Optional[str]:
        """index to token."""
        return self._tokenizer.id_to_token(int(index))

    def token_to_id(self, token: str):
        """token to index."""
        return self._convert_token_to_id_with_added_voc(token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """convert tokens to string."""
        return " ".join(tokens)

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
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def tokenize(self):
        """tokenize."""

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string
