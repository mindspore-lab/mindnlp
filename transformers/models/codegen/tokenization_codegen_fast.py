# coding=utf-8
# Copyright 2022 The Salesforce authors, The Open AI Team Authors and The HuggingFace Inc. team.
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
# ============================================================================
"""Fast Tokenization classes for CodeGen."""


import json
import re
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore
from tokenizers import pre_tokenizers

from mindnlp.utils import logging
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_codegen import CodeGenTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Salesforce/codegen-350M-mono": "https://hf-mirror.com/Salesforce/codegen-350M-mono/resolve/main/vocab.json",
    },
    "merges_file": {
        "Salesforce/codegen-350M-mono": "https://hf-mirror.com/Salesforce/codegen-350M-mono/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "Salesforce/codegen-350M-mono": (
            "https://hf-mirror.com/Salesforce/codegen-350M-mono/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Salesforce/codegen-350M-mono": 2048,
}


class CodeGenTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CodeGen tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import CodeGenTokenizerFast
        ...
        >>> tokenizer = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono")
        >>> tokenizer("Hello world")["input_ids"]
        [15496, 995]
        >>> tokenizer(" Hello world")["input_ids"]
        [18435, 995]
        ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (CodeGen tokenizer detect beginning of words by the preceding space).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = CodeGenTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        """
        Initializes an instance of the CodeGenTokenizerFast class.

        Args:
            vocab_file (str): The path to the vocabulary file. Default is None.
            merges_file (str): The path to the merges file. Default is None.
            tokenizer_file (str): The path to the tokenizer file. Default is None.
            unk_token (str): The unknown token to be used. Default is 'endoftext'.
            bos_token (str): The beginning of sequence token. Default is 'endoftext'.
            eos_token (str): The end of sequence token. Default is 'endoftext'.
            add_prefix_space (bool): Whether to add prefix space. Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            ValueError: If attempting to add a BOS token using the fast tokenizer. Suggests using the slow tokenizer instead.
            JSONDecodeError: If the pre_tokenizer state cannot be decoded from JSON.
            AttributeError: If the pre_tokenizer class cannot be found.
        """
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        if kwargs.pop("add_bos_token", False):
            model_id = kwargs.pop("name_or_path", "")
            raise ValueError(
                "Currenty GPT2's fast tokenizer does NOT support adding a BOS token. "
                "Instead you should use GPT2's slow tokenizer class `CodeGenTokenizer` as follows: \n"
                f"`CodeGenTokenizer.from_pretrained('{model_id}')`\nor\n"
                f"`AutoTokenizer.from_pretrained('{model_id}', use_fast=False)`\n"
                "This issue will be fixed soon, see: https://github.com/huggingface/tokenizers/pull/1005."
                " so that the fast tokenizer works correctly."
            )

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        This method '_batch_encode_plus' in the class 'CodeGenTokenizerFast' encodes a batch of inputs into tokenized and encoded representations.

        Args:
            *args: Variable length positional arguments.
            **kwargs:
                Variable length keyword arguments.

                - is_split_into_words (bool, optional):
                Specifies if the input is already split into words. Defaults to False.

        Returns:
            BatchEncoding: A dictionary-like object containing the tokenized and encoded representations of the input batch.

        Raises:
            AssertionError: If the 'add_prefix_space' attribute is not set to True and the 'is_split_into_words' argument is True.
                In such cases, the method requires the instantiation of 'CodeGenTokenizerFast'
                with 'add_prefix_space=True' for using it with pretokenized inputs.
        """
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        Encodes the input data into a batch encoding using the CodeGenTokenizerFast.

        Args:
            self: An instance of the CodeGenTokenizerFast class.

        Returns:
            A BatchEncoding object containing the encoded input data.

        Raises:
            AssertionError: If the 'is_split_into_words' keyword argument is set to True and the CodeGenTokenizerFast instance
                            was not instantiated with 'add_prefix_space=True'. This is necessary to use pretokenized inputs.

        """
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary files generated by the tokenizer model.

        Args:
            self (CodeGenTokenizerFast): The instance of the CodeGenTokenizerFast class.
            save_directory (str): The directory path where the vocabulary files will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the filename of the saved vocabulary files.
                Defaults to None if not provided.

        Returns:
            Tuple[str]: A tuple containing the filenames of the saved vocabulary files.

        Raises:
            SpecificException: Describes when a specific exception might be raised during the save operation.
            AnotherException: Describes when another type of exception might be raised during the save operation.
            AnyOtherException: Describes any other exception that the function may raise.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "mindspore.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        truncate_before_pattern: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens
        and clean up tokenization spaces.
        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            truncate_before_pattern (`List[str]`, *optional*, defaults to `None`):
                A list of regular expression strings that will be used to truncate the returned string. This can be
                used to remove extra pieces of code (e.g. truncate if observing a comment symbol "#" at the beginning
                of a new line).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        decoded_text = super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)

        return decoded_text

    def truncate(self, completion, truncate_before_pattern):
        """
        Truncate the completion string before a given pattern using a list of truncate_before_patterns.
        
        Args:
            self (CodeGenTokenizerFast): An instance of the CodeGenTokenizerFast class.
            completion (str): The completion string to be truncated.
            truncate_before_pattern (list): A list of patterns to truncate the completion string before.
                Each pattern is compiled using the re.compile() method with the re.MULTILINE flag.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        def find_re(string, pattern, start_pos):
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]

        prints = list(re.finditer("^print", completion, re.MULTILINE))

        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        defs = list(re.finditer("^def", completion, re.MULTILINE))

        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        terminals_pos = [
            pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
        ]

        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        return completion

__all__ = ['CodeGenTokenizerFast']
