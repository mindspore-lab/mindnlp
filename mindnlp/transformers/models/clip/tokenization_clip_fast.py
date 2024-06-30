# coding=utf-8
# Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""


from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers

from mindnlp.utils import logging
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_clip import CLIPTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/clip-vit-base-patch32": "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai/clip-vit-base-patch32": "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "openai/clip-vit-base-patch32": (
            "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai/clip-vit-base-patch32": 77,
}


class CLIPTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CLIP tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = CLIPTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # hack to enable padding
        **kwargs,
    ):
        """
        Initialize the CLIPTokenizerFast class.
        
        Args:
            self (object): The instance of the CLIPTokenizerFast class.
            vocab_file (str, optional): Path to the vocabulary file. Default is None.
            merges_file (str, optional): Path to the merges file. Default is None.
            tokenizer_file (str, optional): Path to the tokenizer file. Default is None.
            unk_token (str, optional): The unknown token. Default is 'endoftext'.
            bos_token (str, optional): The beginning of sequence token. Default is '<|startoftext|>'.
            eos_token (str, optional): The end of sequence token. Default is 'endoftext'.
            pad_token (str, optional): The padding token. Default is 'endoftext'.
        
        Returns:
            None.
        
        Raises:
            ValueError: Raised if the backend tokenizer pre_tokenizer does not match the expected format.
                The CLIP tokenizer in this version has been heavily modified from transformers version 4.17.0. To
                resolve this issue, convert the existing tokenizer to be compatible with this version using
                `CLIPTokenizerFast.from_pretrained("path_to_local_folder_or_hub_repo", from_slow=True)`.
                If using an older tokenizer version, revert to a version prior to 4.17.0 of transformers.
        """
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        if not isinstance(self.backend_tokenizer.pre_tokenizer, pre_tokenizers.Sequence):
            raise ValueError(
                "The `backend_tokenizer` provided does not match the expected format. The CLIP tokenizer has been"
                " heavily modified from transformers version 4.17.0. You need to convert the tokenizer you are using"
                " to be compatible with this version.The easiest way to do so is"
                ' `CLIPTokenizerFast.from_pretrained("path_to_local_folder_or_hub_repo, from_slow=True)`. If you want'
                " to use your existing tokenizer, you will have to revert to a version prior to 4.17.0 of"
                " transformers."
            )

        self._wrap_decode_method_backend_tokenizer()

    # Very ugly hack to enable padding to have a correct decoding see https://github.com/huggingface/tokenizers/issues/872
    def _wrap_decode_method_backend_tokenizer(self):
        """
        This method '_wrap_decode_method_backend_tokenizer' is a private method within the 'CLIPTokenizerFast' class.
        It wraps the 'decode' method of the backend tokenizer by modifying its behavior.

        Args:
            self (CLIPTokenizerFast): The instance of the CLIPTokenizerFast class itself.
                It is used to access the backend_tokenizer attribute and modify the decode method.

        Returns:
            None: This method does not return any value explicitly,
                but it modifies the behavior of the 'decode' method of the backend tokenizer.

        Raises:
            None: However, potential exceptions that could be raised during the execution of the modified 'decode'
                method of the backend tokenizer should be handled within that method.
        """
        orig_decode_method = self.backend_tokenizer.decode

        def new_decode_method(*args, **kwargs):
            text = orig_decode_method(*args, **kwargs)
            text = text.replace(self.backend_tokenizer.model.end_of_word_suffix, " ").strip()
            return text

        self.backend_tokenizer.decode = new_decode_method

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A CLIP sequence has the following format:

        - single sequence: `<|startoftext|> X <|endoftext|>`

        Pairs of sequences are not the expected use case, but they will be handled without a separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        if token_ids_1 is None:
            return bos_token + token_ids_0 + eos_token
        return bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of
        zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        if token_ids_1 is None:
            return len(bos_token + token_ids_0 + eos_token) * [0]
        return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary generated by the CLIPTokenizerFast model to the specified directory.
        
        Args:
            self (CLIPTokenizerFast): The instance of the CLIPTokenizerFast class.
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (Optional[str], optional): An optional prefix to be included in the saved filenames.
                Default is None.
        
        Returns:
            Tuple[str]: A tuple containing the filenames of the saved vocabulary files.
        
        Raises:
            This method does not raise any exceptions.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

__all__ = ['CLIPTokenizerFast']
