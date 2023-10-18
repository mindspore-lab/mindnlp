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
# pylint: disable=W1203
# pylint: disable=C0103
# pylint: disable=R1710
"""Tokenization classes for LLaMA."""
import logging
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sentencepiece as spm

import mindspore
from tokenizers import AddedToken
from ...tokenization_utils import PreTrainedTokenizer
# from mindnlp.configs import HF_TOKENIZER_CONFIG_URL_BASE

LLAMA_SUPPORT_LIST = [
    "meta-llama/Llama-2-7b-hf",
]


VOCAB_FILES_NAMES = {
    "vocab_file": "tokenizer.model"
}
PRETRAINED_VOCAB_MAP = {
    "meta-llama/Llama-2-7b-hf": "https://huggingface.co/Moon99/hello-llama/resolve/main/tokenizer.model"
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'llama-7b-hf': 2048,
}
SPIECE_UNDERLINE = "▁"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class LlamaTokenizer(PreTrainedTokenizer):
    """
        Tokenizer used for T5 text process.
        Args:
            vocab (Vocab): Vocabulary used to look up words.
            return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.
        Examples:
            >>> from mindspore.dataset import text
            >>> from mindnlp.transforms import T5Tokenizer
            >>> text = "Believing that faith can triumph over everything is in itself the greatest belief"
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-base')
            >>> tokens = tokenizer.encode(text)
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=True,
        spaces_between_special_tokens=False,
        legacy=None,  # not used
        **kwargs,
    ):
        self._tokenizer_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        return_token = kwargs.pop('return_token', False)
        self.return_token = return_token

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token


        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self._tokenizer = self.get_spm_processor()

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self._tokenizer_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            # legacy=legacy,
            **kwargs,
        )

    def get_spm_processor(self):
        """Get SentencePieceProcessor Tokenizer."""
        tokenizer = spm.SentencePieceProcessor(**self._tokenizer_kwargs)
        tokenizer.Load(self.vocab_file)
        return tokenizer

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self._tokenizer.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # TODO: add special tokens
        return vocab

    def execute_py(self, text_input):
        """Execute method."""
        return self.tokenize(text_input)

    def _execute_py(self, text_input):
        """Execute method."""
        return self._tokenize(text_input)

    def tokenize(self, text_input) -> List[str]:
        return self._execute_py(text_input)

    def batch_encode(self, texts, max_length, **kwargs):
        """Implement basic encode logic for encoding a sequence of sequences or a pair of sequences."""
        all_inputs_ids = []
        all_attention_mask = []

        for text in texts:
            assert isinstance(text, str)
            tokens = self._tokenize(text)
            inputs_ids = self.convert_tokens_to_ids(tokens)
            inputs_ids = [self._tokenizer.bos_id()] + inputs_ids + [self._tokenizer.eos_id()]
            attention_mask = [1] * len(inputs_ids)

            # pad
            if self.pad_token is None:
                self.pad_token = self.eos_token
                pad_id = self._tokenizer.eos_id()
            if len(inputs_ids) < max_length:
                inputs_ids += [pad_id] * (max_length - len(inputs_ids))
                attention_mask += [0] * (max_length - len(attention_mask))
            else:
                inputs_ids = inputs_ids[:max_length]
                attention_mask = attention_mask[:max_length]

            all_inputs_ids.append(inputs_ids)
            all_attention_mask.append(attention_mask)

        # convert to numpy
        return_pts = kwargs.pop('return_pts', False)
        if not return_pts:
            return np.array(all_inputs_ids), np.array(all_attention_mask)

        return {
            'input_ids':mindspore.Tensor(np.array(all_inputs_ids), dtype=mindspore.int64), 
            'attention_mask':mindspore.Tensor(np.array(all_attention_mask), dtype=mindspore.int32)
        }


    def _tokenize(self, text_input):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self._tokenizer.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        text_input = self._convert_to_unicode(text_input)

        tokens = self._tokenizer.encode(text_input, out_type=str)
        if self.return_token:
            return tokens
        # return ids
        return np.array(self.convert_tokens_to_ids(tokens))

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._tokenizer.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self._tokenizer.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self._tokenizer.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self._tokenizer.decode(current_sub_tokens)
        return out_string

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
        raise ValueError(f"Unsupported string type: {type(text_input)}, {text_input.dtype}")

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logging.error(f"Vocabulary path {save_directory} should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as f:
                content_spiece_model = self._tokenizer.serialized_model_proto()
                f.write(content_spiece_model)

        return (out_vocab_file,)

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
        raise ValueError(f"Unsupported string type: {type(text_input)}, {text_input.dtype}")

__all__ = ['LlamaTokenizer']
