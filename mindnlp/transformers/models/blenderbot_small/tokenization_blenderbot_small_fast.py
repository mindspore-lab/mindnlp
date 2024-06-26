# coding=utf-8
# Copyright 2021, The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Fast tokenization class for BlenderbotSmall."""
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ....utils import logging
from .tokenization_blenderbot_small import BlenderbotSmallTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}


class BlenderbotSmallTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" BlenderbotSmall tokenizer (backed by HuggingFace's *tokenizers* library).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = BlenderbotSmallTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs,
    ):
        """
        Initializes a BlenderbotSmallTokenizerFast object.
        
        Args:
            self: The instance of the class.
            vocab_file (str): Path to the vocabulary file. Default is None.
            merges_file (str): Path to the merges file. Default is None.
            unk_token (str): The unknown token. Default is 'endoftext'.
            bos_token (str): The beginning of sentence token. Default is 'endoftext'.
            eos_token (str): The end of sentence token. Default is 'endoftext'.
            add_prefix_space (bool): Whether to add prefix space. Default is False.
            trim_offsets (bool): Whether to trim offsets. Default is True.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the vocab_file or merges_file is not provided.
            TypeError: If unk_token, bos_token, eos_token are not strings.
            ValueError: If add_prefix_space or trim_offsets are not boolean values.
        """
        super().__init__(
            ByteLevelBPETokenizer(
                vocab=vocab_file,
                merges=merges_file,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        This method builds inputs with special tokens for the BlenderbotSmallTokenizerFast class.
        
        Args:
            self: The instance of the class.
            token_ids_0 (list): The token IDs for the first input sequence.
            token_ids_1 (list, optional): The token IDs for the second input sequence. Defaults to None.
        
        Returns:
            None: The method modifies the input sequences by adding special tokens.
        
        Raises:
            None.
        """
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall
        does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    # Copied from transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )

__all__ = ['BlenderbotSmallTokenizerFast']
