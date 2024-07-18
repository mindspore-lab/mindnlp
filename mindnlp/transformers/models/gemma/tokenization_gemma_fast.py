# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Gemma Tokenizer"""
import os
from shutil import copyfile
from typing import Optional, Tuple

from tokenizers import processors

from mindnlp.utils import is_sentencepiece_available, logging
from ...tokenization_utils_fast import PreTrainedTokenizerFast


if is_sentencepiece_available():
    from .tokenization_gemma import GemmaTokenizer
else:
    GemmaTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}


class GemmaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Gemma tokenizer fast. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no prefix space. Normalization is applied to replace  `" "` with `"▁"`

    Example:
        ```python
        >>> from transformers import GemmaTokenizerFast
        ...
        >>> tokenizer = GemmaTokenizerFast.from_pretrained("hf-internal-testing/dummy-gemma")
        >>> tokenizer.encode("Hello this is a test")
        [2, 4521, 736, 603, 476, 2121]
        ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://hf-mirror.com/docs/tokenizers/api/post-processors) documentation.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<bos>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The padding token
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = GemmaTokenizer
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        """
        Initialize GemmaTokenizerFast object.

        Args:
            self (object): The GemmaTokenizerFast object itself.
            vocab_file (str, optional): Path to the vocabulary file. Default is None.
            tokenizer_file (str, optional): Path to the tokenizer file. Default is None.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Default is False.
            unk_token (str, optional): Unknown token to be used. Default is '<unk>'.
            bos_token (str, optional): Beginning of sentence token. Default is '<bos>'.
            eos_token (str, optional): End of sentence token. Default is '<eos>'.
            pad_token (str, optional): Padding token. Default is '<pad>'.
            add_bos_token (bool, optional): Whether to add the beginning of sentence token. Default is True.
            add_eos_token (bool, optional): Whether to add the end of sentence token. Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        Checks if the slow tokenizer can be saved.

        Args:
            self: An instance of the GemmaTokenizerFast class.

        Returns:
            bool:
                A boolean value indicating whether the slow tokenizer can be saved.
                Returns True if the vocab_file exists, otherwise False.

        Raises:
            None.
        """
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.update_post_processor
    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        """
        Adds an end-of-sentence (EOS) token to the GemmaTokenizerFast object.

        Args:
            self: An instance of the GemmaTokenizerFast class.

        Returns:
            None.

        Raises:
            None.

        This method adds an EOS token to the GemmaTokenizerFast object.
        The EOS token is used to mark the end of a sentence or text sequence.
        It is commonly used in natural language processing tasks such as language modeling and text generation.
        By adding an EOS token, the GemmaTokenizerFast object can handle text sequences more effectively,
        allowing for better analysis and processing.
        """
        return self._add_eos_token

    @property
    def add_bos_token(self):
        """
        This method adds the beginning of sentence (BOS) token to the tokenizer.

        Args:
            self (GemmaTokenizerFast): The instance of GemmaTokenizerFast class.

        Returns:
            None.

        Raises:
            None.
        """
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        """Sets the value of the add_eos_token property in the GemmaTokenizerFast class.

        Args:
            self (GemmaTokenizerFast): The instance of GemmaTokenizerFast.
            value (bool): The value to set for the add_eos_token property.
                It determines whether to add an end-of-sequence token to the tokenized output.

        Returns:
            None.

        Raises:
            None.
        """
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        """
        Method: add_bos_token

        Description:
        Setter method for adding a beginning of sentence (BOS) token to the GemmaTokenizerFast.

        Args:
            self: (GemmaTokenizerFast) The instance of GemmaTokenizerFast.
            value: (bool) A boolean value indicating whether to add the BOS token.
                True enables adding the BOS token, while False disables it.

        Returns:
            None.

        Raises:
            None.
        """
        self._add_bos_token = value
        self.update_post_processor()

    # Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the GemmaTokenizerFast instance to the specified directory with an optional filename prefix.

        Args:
            self (GemmaTokenizerFast): The instance of the GemmaTokenizerFast class.
            save_directory (str): The directory path where the vocabulary will be saved.
            filename_prefix (Optional[str], optional): An optional prefix to be added to the filename. Defaults to None.

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.

        Raises:
            ValueError:
                If the fast tokenizer does not have the necessary information to save the vocabulary for a slow
                tokenizer.
            OSError: If the save_directory provided is not a valid directory path.
            IOError: If an error occurs during the file copying process.
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    # Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build inputs with special tokens for the GemmaTokenizerFast.

        Args:
            self (GemmaTokenizerFast): An instance of the GemmaTokenizerFast class.
            token_ids_0 (list): A list of token IDs representing the first sequence.
            token_ids_1 (list, optional): A list of token IDs representing the second sequence.
                Defaults to None.

        Returns:
            list: A list of token IDs representing the input sequences with added special tokens.

        Raises:
            None.

        This method takes two sequences of token IDs and adds special tokens, such as
        beginning of sequence (bos) and end of sequence (eos) tokens. The special tokens
        are added based on the configuration of the tokenizer.

        The token_ids_0 parameter is a list of token IDs representing the first sequence.
        This parameter is required.

        The token_ids_1 parameter is an optional list of token IDs representing the second
        sequence. If provided, the method concatenates the first and second sequences with
        the special tokens in between.

        The method returns a list of token IDs representing the input sequences with the
        special tokens added.

        Example:
            ```python
            >>> tokenizer = GemmaTokenizerFast()
            >>> token_ids_0 = [101, 202, 303]
            >>> token_ids_1 = [404, 505]
            >>> inputs = tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
            >>> print(inputs)
            Output:
            [101, 202, 303, 102, 404, 505, 102]
            ```
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

__all__ = ['GemmaTokenizerFast']
