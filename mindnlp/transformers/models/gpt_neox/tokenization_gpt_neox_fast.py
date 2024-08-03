# coding=utf-8
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for GPTNeoX."""
import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers, processors

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ....utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class GPTNeoXTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-NeoX-20B tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import GPTNeoXTokenizerFast
        ...
        >>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("openai-community/gpt2")
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
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            Token for padding a sequence.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPTNeoX tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespaces.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_bos_token=False,
        add_eos_token=False,
        add_prefix_space=False,
        **kwargs,
    ):
        """Initialize a new instance of the GPTNeoXTokenizerFast class.

        Args:
            self (GPTNeoXTokenizerFast): The instance of the class.
            vocab_file (str, optional): The file path to the vocabulary file. Defaults to None.
            merges_file (str, optional): The file path to the merges file. Defaults to None.
            tokenizer_file (str, optional): The file path to the tokenizer file. Defaults to None.
            unk_token (str, optional): The unknown token. Defaults to 'endoftext'.
            bos_token (str, optional): The beginning of sentence token. Defaults to 'endoftext'.
            eos_token (str, optional): The end of sentence token. Defaults to 'endoftext'.
            pad_token (str, optional): The padding token. Defaults to None.
            add_bos_token (bool, optional): Whether to add the beginning of sentence token. Defaults to False.
            add_eos_token (bool, optional): Whether to add the end of sentence token. Defaults to False.
            add_prefix_space (bool, optional): Whether to add prefix space. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    @property
    def add_eos_token(self):
        """
        Adds an end-of-sequence (EOS) token to the tokenizer.

        Args:
            self: The current instance of the GPTNeoXTokenizerFast class.
                Type: GPTNeoXTokenizerFast
                Purpose: Represents the tokenizer instance to which the end-of-sequence token is added.

        Returns:
            None.

        Raises:
            None.
        """
        return self._add_eos_token

    @property
    def add_bos_token(self):
        """
        Adds a beginning of sentence (BOS) token to the tokenizer.

        Args:
            self: An instance of the GPTNeoXTokenizerFast class.

        Returns:
            None.

        Raises:
            None.

        This method adds a BOS token to the tokenizer.
        The BOS token is used to indicate the start of a sentence or a sequence.

        Note:
            The BOS token is specific to the GPTNeoXTokenizerFast class and cannot be used with other tokenizers.
        """
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        """
        Method to set the flag for adding an end-of-sequence token in the GPTNeoXTokenizerFast class.

        Args:
            self (GPTNeoXTokenizerFast): An instance of the GPTNeoXTokenizerFast class.
                Represents the tokenizer object on which the end-of-sequence token flag is being set.
            value (bool): A boolean value indicating whether to add an end-of-sequence token.
                If True, an end-of-sequence token will be added; if False, it will not be added.

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
        Sets the value of the 'add_bos_token' attribute and updates the post-processor.

        Args:
            self (GPTNeoXTokenizerFast): The instance of the GPTNeoXTokenizerFast class.
            value: The new value to be assigned to the 'add_bos_token' attribute.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method is a setter for the 'add_bos_token' attribute of the GPTNeoXTokenizerFast class.
            It allows setting a new value for the attribute and automatically triggers the update_post_processor method.

            The 'add_bos_token' attribute determines whether to add a beginning of sentence (BOS) token during tokenization.
            When 'add_bos_token' is set to True, a BOS token will be added at the beginning of each tokenized sequence.
            When 'add_bos_token' is set to False, no BOS token will be added.

            After setting the new value for 'add_bos_token', the update_post_processor method is called to update the
            post-processor based on the new value. The update_post_processor method handles any necessary adjustments
            to the post-processing logic, if required.

            Note that changing the 'add_bos_token' attribute value will impact the tokenization process and the
            resulting tokenized sequences.

        Example:
            ```python
            >>> tokenizer = GPTNeoXTokenizerFast()
            >>> tokenizer.add_bos_token = True
            ```
            In the above example, the 'add_bos_token' attribute of the 'tokenizer' instance is set to True,
            which enables the addition of BOS tokens during tokenization.
        """
        self._add_bos_token = value
        self.update_post_processor()

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

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    # Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        This method builds inputs with special tokens for the GPTNeoXTokenizerFast class.
        
        Args:
            self (GPTNeoXTokenizerFast): The instance of the GPTNeoXTokenizerFast class.
            token_ids_0 (list): The list of token IDs for the first input sequence.
            token_ids_1 (list, optional): The list of token IDs for the second input sequence. Defaults to None.
        
        Returns:
            list: The list of token IDs with special tokens added based on the configuration of the tokenizer.
        
        Raises:
            None
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary files of the GPTNeoXTokenizerFast model to the specified directory.
        
        Args:
            self (GPTNeoXTokenizerFast): The instance of the GPTNeoXTokenizerFast class.
            save_directory (str): The directory path where the vocabulary files will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the generated vocabulary files.
                Defaults to None if not provided.
        
        Returns:
            Tuple[str]: A tuple containing the file paths of the saved vocabulary files.
        
        Raises:
            IOError: If there are issues with saving the vocabulary files to the specified directory.
            ValueError: If the provided save_directory is invalid or inaccessible.
            TypeError: If the provided filename_prefix is not a string.
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
            "No chat template is set for this tokenizer, falling back to a default class-level template. "
            "This is very error-prone, because models are often trained with templates different from the class "
            "default! Default chat templates are a legacy feature and will be removed in Transformers v4.43, at which "
            "point any code depending on them will stop working. We recommend setting a valid chat template before "
            "then to ensure that this model continues working without issues."
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"

__all__ = ['GPTNeoXTokenizerFast']
