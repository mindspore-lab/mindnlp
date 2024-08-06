# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""


import json
from typing import Optional, Tuple

from tokenizers import pre_tokenizers

from mindnlp.utils import logging
from .tokenization_gpt2 import GPT2Tokenizer
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "gpt2": "https://hf-mirror.com/gpt2/resolve/main/vocab.json",
        "gpt2-medium": "https://hf-mirror.com/gpt2-medium/resolve/main/vocab.json",
        "gpt2-large": "https://hf-mirror.com/gpt2-large/resolve/main/vocab.json",
        "gpt2-xl": "https://hf-mirror.com/gpt2-xl/resolve/main/vocab.json",
        "distilgpt2": "https://hf-mirror.com/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "gpt2": "https://hf-mirror.com/gpt2/resolve/main/merges.txt",
        "gpt2-medium": "https://hf-mirror.com/gpt2-medium/resolve/main/merges.txt",
        "gpt2-large": "https://hf-mirror.com/gpt2-large/resolve/main/merges.txt",
        "gpt2-xl": "https://hf-mirror.com/gpt2-xl/resolve/main/merges.txt",
        "distilgpt2": "https://hf-mirror.com/distilgpt2/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "gpt2": "https://hf-mirror.com/gpt2/resolve/main/tokenizer.json",
        "gpt2-medium": "https://hf-mirror.com/gpt2-medium/resolve/main/tokenizer.json",
        "gpt2-large": "https://hf-mirror.com/gpt2-large/resolve/main/tokenizer.json",
        "gpt2-xl": "https://hf-mirror.com/gpt2-xl/resolve/main/tokenizer.json",
        "distilgpt2": "https://hf-mirror.com/distilgpt2/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}


class GPT2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import GPT2TokenizerFast
        ...
        >>> tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
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
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = GPT2Tokenizer

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
        __init__

        Initializes a new instance of the GPT2TokenizerFast class.

        Args:
            self: The instance of the class.
            vocab_file (str, optional): The path to the vocabulary file. Defaults to None.
            merges_file (str, optional): The path to the merges file. Defaults to None.
            tokenizer_file (str, optional): The path to the tokenizer file. Defaults to None.
            unk_token (str, optional): The unknown token. Defaults to 'endoftext'.
            bos_token (str, optional): The beginning of sentence token. Defaults to 'endoftext'.
            eos_token (str, optional): The end of sentence token. Defaults to 'endoftext'.
            add_prefix_space (bool, optional): A flag indicating whether to add a prefix space. Defaults to False.

        Returns:
            None.

        Raises:
            None.
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

        self.add_bos_token = kwargs.pop("add_bos_token", False)

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        This method '_batch_encode_plus' is defined in the class 'GPT2TokenizerFast'. It takes the following parameters:

        Args:
            self: (object) The instance of the class.

        Returns:
            (BatchEncoding) An instance of the 'BatchEncoding' class containing the encoded inputs.

        Raises:
            AssertionError: If the 'add_prefix_space' is False and the 'is_split_into_words' is also False,
                an assertion error is raised with the message 'You need to instantiate GPT2TokenizerFast with
                add_prefix_space=True to use it with pretokenized inputs'.
            Any other exceptions: raised by the 'super()._batch_encode_plus' method.
        """
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """Encodes the input text into a batch of numerical representations using the GPT2TokenizerFast.

        Args:
            self (GPT2TokenizerFast): An instance of the GPT2TokenizerFast class.

        Returns:
            BatchEncoding: A dictionary-like object containing the encoded inputs.

        Raises:
            AssertionError: If the 'is_split_into_words' parameter is set to True and the 'add_prefix_space' parameter
            is set to False, an AssertionError is raised.

        Note:
            This method is intended to be used with pretokenized inputs. If the 'is_split_into_words' parameter
            is set to True, make sure to instantiate the GPT2TokenizerFast class with 'add_prefix_space=True'.

        Example:
            ```python
            >>> tokenizer = GPT2TokenizerFast()
            >>> encoded_inputs = tokenizer._encode_plus("Hello, world!")
            >>> print(encoded_inputs)
            {'input_ids': [15496, 259, 114, 616], 'attention_mask': [1, 1, 1, 1]}
            >>> tokenizer = GPT2TokenizerFast(add_prefix_space=True)
            >>> encoded_inputs = tokenizer._encode_plus("Hello, world!", is_split_into_words=True)
            AssertionError: You need to instantiate GPT2TokenizerFast with add_prefix_space=True to use it with pretokenized inputs.
            ```
        """
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the GPT2TokenizerFast model to the specified directory.

        Args:
            self (GPT2TokenizerFast): An instance of the GPT2TokenizerFast class.
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (Optional[str], optional): An optional prefix to be added to the saved vocabulary file(s).
                Defaults to None.

        Returns:
            Tuple[str]: A tuple containing the file path(s) where the vocabulary was saved.

        Raises:
            None

        Note:
            - The 'self' parameter represents the instance of the GPT2TokenizerFast class calling this method.
            - The 'save_directory' parameter should be a valid directory path where the vocabulary files will be saved.
            - The 'filename_prefix' parameter allows an optional prefix to be added to the saved vocabulary file(s). If not provided, no prefix will be added.
            - The method returns a tuple containing the file path(s) where the vocabulary files were saved.

        Example:
            ```python
            >>> tokenizer = GPT2TokenizerFast()
            >>> tokenizer.save_vocabulary("path/to/save", filename_prefix="vocab")
            >>> # The vocabulary files will be saved with the prefix "vocab" in the specified directory.
            >>> # The method will return a tuple containing the file paths.
            ```
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    @property
    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://hf-mirror.com/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"

__all__ = ['GPT2TokenizerFast']
