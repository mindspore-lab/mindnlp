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
GPT2Tokenizer
"""
import numpy as np
from mindspore.dataset.text.transforms import Implementation
from mindspore import Tensor
from tokenizers import Tokenizer
from ...tokenization_utils import PreTrainedTokenizer


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt_bigcode": 2048,
}


class GPTBigCodeTokenizer(PreTrainedTokenizer):
    """
        Tokenizer used for GPT2 text process.

        Args:
            vocab (Vocab): Vocabulary used to look up words.
            return_token (bool): Whether to return token. If True: return tokens. False: return ids. Default: True.

        """
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs
    ):
        """
        Initializes a new instance of the GPTBigCodeTokenizer class.
        
        Args:
            self (GPTBigCodeTokenizer): The instance of the class itself.
            tokenizer_file (str): The file path of the tokenizer file to be used. Only string values are supported.
            unk_token (str): The token to represent unknown words. Default is 'endoftext'.
            bos_token (str): The token to represent the beginning of a sentence. Default is 'endoftext'.
            eos_token (str): The token to represent the end of a sentence. Default is 'endoftext'.
            add_prefix_space (bool): Whether to add a prefix space before the input text. Default is False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the tokenizer_file is not of type string.
        
        """
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs)

        return_token = kwargs.pop('return_token', False)

        if isinstance(tokenizer_file, str):
            self._tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            raise ValueError(f'only support string, but got {tokenizer_file}')

        self.return_token = return_token
        self.implementation = Implementation.PY

    def execute_py(self, text_input):
        """
        Execute method.
        """
        return self._execute_py(text_input)

    def _execute_py(self, text_input):
        """
        Execute method.
        """
        text_input = self._convert_to_unicode(text_input)
        tokens = self._tokenizer.encode(text_input)
        if self.return_token is True:
            return np.array(tokens.tokens)
        return {"input_ids": Tensor(np.array(tokens.ids)), "attention_mask": Tensor(np.array(tokens.attention_mask))}

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
        """
        Converts the given token to its corresponding ID using the GPTBigCodeTokenizer.
        
        Args:
            self (GPTBigCodeTokenizer): An instance of the GPTBigCodeTokenizer class.
            token (str): The token to be converted to ID.
        
        Returns:
            int: The ID corresponding to the given token. Returns self.unk_token_id if the token is not found.
        
        Raises:
            None.
        """
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index


__all__ = ['GPTBigCodeTokenizer']
