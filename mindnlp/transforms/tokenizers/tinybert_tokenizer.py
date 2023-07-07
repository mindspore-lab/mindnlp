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
TinyBert Tokenizer
"""
import os
from mindspore import log as logger
from mindspore.dataset.text import Vocab as msVocab
from mindspore.dataset.text.transforms import Implementation
import numpy as np
from tokenizers.implementations import BertWordPieceTokenizer

from mindnlp.abc import PreTrainedTokenizer
from mindnlp.utils.download import cached_path
from mindnlp.vocab import Vocab

PRETRAINED_VOCAB_MAP = {
    "tinybert_4L_zh": "https://download.mindspore.cn/toolkits/mindnlp/models/tinybert/tinybert_4L_zh/vocab.txt",
    "tinybert_6L_zh": "https://download.mindspore.cn/toolkits/mindnlp/models/tinybert/tinybert_6L_zh/vocab.txt"
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tinybert_4L_zh": 312,
    "tinybert_6L_zh": 768
}

class TinyBertTokenizer(PreTrainedTokenizer):
    """
    Tokenizer used for TinyBert text process.
    """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_map = PRETRAINED_VOCAB_MAP

    def __init__(
        self,
        vocab=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        if isinstance(vocab, msVocab):
            vocab_dict = vocab.vocab()
        elif isinstance(vocab, Vocab):
            vocab_dict = vocab.vocab
        elif isinstance(vocab, str):
            if not os.path.isfile(vocab):
                raise ValueError(f"{vocab} is not a file.")
            if not vocab.endswith('.txt'):
                raise ValueError(f"{vocab} is not a txt file.")
            vocab_dict = msVocab.from_file(vocab).vocab()
        else:
            raise ValueError(
                f'only support Vocab class from mindspore or mindnlp, \
                    and a vocab.txt, but got {vocab}'
            )

        return_token = kwargs.pop('return_token', False)

        self._tokenizer = BertWordPieceTokenizer(vocab=vocab_dict, lowercase=do_lower_case)

        self.return_token = return_token
        self.implementation = Implementation.PY

    def save(self, save_path: str):
        """
        save tokenizer
        """
        # check save_path
        if not save_path.endswith('.txt'):
            raise ValueError(f"{save_path} is not a txt file.")

        vocab_dict = self._tokenizer.get_vocab()
        sorted_array = sorted(vocab_dict.items(), key=lambda item: item[1])

        with open(save_path, 'w', encoding='utf-8') as file:
            for token, _ in sorted_array:
                file.write(token+"\n")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        """from_pretrained"""
        cache_dir = kwargs.pop("cache_dir", None)
        _ = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        # Get files from url, cache, or disk depending on the case
        # Load tokenizer
        folder_name = None
        if pretrained_model_name_or_path is not None:

            # model name
            if pretrained_model_name_or_path in cls.pretrained_vocab_map:
                archive_file = cls.pretrained_vocab_map[pretrained_model_name_or_path]
                folder_name = pretrained_model_name_or_path
            # dir
            elif os.path.isdir(pretrained_model_name_or_path):
                archive_file = os.path.join(pretrained_model_name_or_path, "vocab.txt")
            # file
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
                if pretrained_model_name_or_path in cls.pretrained_vocab_map:
                    msg = f"Couldn't reach server at '{archive_file}' to download pretrained weights."
                else:
                    format1 = ", ".join(cls.pretrained_vocab_map.keys())
                    format2 = ["vocab.txt"]
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

    def __call__(self, text_input):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if isinstance(text_input, str):
            text_input = np.array(text_input)
        elif not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}.")
        return super().__call__(text_input)

    def execute_py(self, text_input):
        """
        Execute method.
        """
        return self._execute_py(text_input)

    def _execute_py(self, text_input):
        """
        Execute method.
        """
        text = self._convert_to_unicode(text_input)
        output = self._tokenizer.encode(text)
        if self.return_token is True:
            return np.array(output.tokens)
        return np.array(output.ids)

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

    def _convert_token_to_id(self, token):
        return self._tokenizer.token_to_id(token)
