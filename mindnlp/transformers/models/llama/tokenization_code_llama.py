# coding=utf-8
# Copyright 2023 MetaAI and the HuggingFace Inc. team. All rights reserved.
#
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
"""Tokenization classes for Code LLaMA."""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from mindnlp.utils import logging, requires_backends
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-code-tokenizer": "https://hf-mirror.com/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-code-tokenizer": "https://hf-mirror.com/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "hf-internal-testing/llama-code-tokenizer": 2048,
}
SPIECE_UNDERLINE = "▁"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class CodeLlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a CodeLlama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as
    there is no padding token in the original model.

    The default configuration match that of
    [codellama/CodeLlama-7b-Instruct-hf](https://hf-mirror.com/codellama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json)
    which supports prompt infilling.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        prefix_token (`str`, *optional*, defaults to `"▁<PRE>"`):
            Prefix token used for infilling.
        middle_token (`str`, *optional*, defaults to `"▁<MID>"`):
            Middle token used for infilling.
        suffix_token (`str`, *optional*, defaults to `"▁<SUF>"`):
            Suffix token used for infilling.
        eot_token (`str`, *optional*, defaults to `"▁<EOT>"`):
            End of text token used for infilling.
        fill_token (`str`, *optional*, defaults to `"<FILL_ME>"`):
            The token used to split the input between the prefix and suffix.
        suffix_first (`bool`, *optional*, defaults to `False`):
            Whether the input prompt and suffix should be formatted with the suffix first.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

                - `nbest_size = {0,1}`: No sampling is performed.
                - `nbest_size > 1`: samples from the nbest_size results.
                - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
            BPE-dropout.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether to add a beginning of sequence token at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether to add an end of sequence token at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to clean up the tokenization spaces.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        prefix_token="▁<PRE>",
        middle_token="▁<MID>",
        suffix_token="▁<SUF>",
        eot_token="▁<EOT>",
        fill_token="<FILL_ME>",
        suffix_first=False,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        additional_special_tokens=None,
        use_default_system_prompt=False,
        **kwargs,
    ):
        """
        This method initializes an instance of the CodeLlamaTokenizer class.

        Args:
            self: The instance of the class.
            vocab_file (str): The path to the vocabulary file.
            unk_token (str, optional): The unknown token, default is '<unk>'.
            bos_token (str, optional): The beginning of sequence token, default is '<s>'.
            eos_token (str, optional): The end of sequence token, default is '</s>'.
            prefix_token (str, optional): The prefix token, default is '▁<PRE>'.
            middle_token (str, optional): The middle token, default is '▁<MID>'.
            suffix_token (str, optional): The suffix token, default is '▁<SUF>'.
            eot_token (str, optional): The end of text token, default is '▁<EOT>'.
            fill_token (str, optional): The fill token, default is '<FILL_ME>'.
            suffix_first (bool): Indicates whether suffix comes before prefix.
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for the sentencepiece model.
            add_bos_token (bool, optional): Whether to add the bos token, default is True.
            add_eos_token (bool, optional): Whether to add the eos token, default is False.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces, default is False.
            additional_special_tokens (list, optional): Additional special tokens to include.
            use_default_system_prompt (bool, optional): Whether to use the default system prompt.

        Returns:
            None.

        Raises:
            MissingBackendError: If the required backend 'protobuf' is not available.
        """
        requires_backends(self, "protobuf")
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token

        self.use_default_system_prompt = use_default_system_prompt
        # mark tokens special to skip them
        additional_special_tokens = additional_special_tokens or []
        for token in [prefix_token, middle_token, suffix_token, eot_token]:
            additional_special_tokens += [token] if token is not None else []

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self._prefix_token = prefix_token
        self._middle_token = middle_token
        self._suffix_token = suffix_token
        self._eot_token = eot_token
        self.fill_token = fill_token
        self.suffix_first = suffix_first
        self.sp_model = self.get_spm_processor()

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            prefix_token=prefix_token,
            middle_token=middle_token,
            suffix_token=suffix_token,
            eot_token=eot_token,
            fill_token=fill_token,
            sp_model_kwargs=self.sp_model_kwargs,
            suffix_first=suffix_first,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            additional_special_tokens=additional_special_tokens,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        """
        Returns the length of the unknown token in the CodeLlamaTokenizer.

        Args:
            self (CodeLlamaTokenizer): An instance of the CodeLlamaTokenizer class.

        Returns:
            int: The length of the unknown token. If the unknown token is not found, it returns 0.

        Raises:
            None.

        """
        return len(self.sp_model.encode(str(self.unk_token)))

    def get_spm_processor(self):
        """
        This method initializes and returns a SentencePieceProcessor object for tokenizing text using
        the SentencePiece library.

        Args:
            self: The instance of the CodeLlamaTokenizer class.

        Returns:
            spm.SentencePieceProcessor: A tokenizer object of type spm.SentencePieceProcessor.

        Raises:
            None:
                However, potential exceptions that may occur during the method execution include:

                - FileNotFoundError: If the specified vocab_file cannot be found.
                - IOError: If there are issues with reading the vocab_file.
                - ValueError: If the provided sp_model_kwargs are invalid or missing required information.
                - Any other relevant exceptions that may occur during the loading and initialization of the tokenizer.
        """
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf()
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    @property
    def prefix_token(self):
        """
        Returns the prefix token used for tokenizing code in the CodeLlamaTokenizer class.

        Args:
            self: An instance of the CodeLlamaTokenizer class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the prefix token that is used for tokenizing code in the CodeLlamaTokenizer class.
        The prefix token serves as a marker or indicator to identify the start of a code block or expression.
        It is used during the tokenization process to correctly identify and separate different parts of the code.

        Note that the prefix token is an internal attribute of the CodeLlamaTokenizer class, and it is not meant to
        be modified directly. To change the prefix token, use the appropriate setter method or modify the underlying
        implementation of the class if necessary.

        Example:
            ```python
            >>> tokenizer = CodeLlamaTokenizer()
            >>> prefix = tokenizer.prefix_token
            >>> print(prefix)
            >>> # Output: '>>'
            ```
        """
        return self._prefix_token

    @property
    def prefix_id(self):
        """
        Method to retrieve the ID associated with the prefix token in the CodeLlamaTokenizer class.

        Args:
            self (CodeLlamaTokenizer): The instance of the CodeLlamaTokenizer class.

        Returns:
            None: If the prefix token is None, the method returns None.
                Otherwise, it returns the ID associated with the prefix token.

        Raises:
            None
        """
        if self._prefix_token is None:
            return None
        return self.convert_tokens_to_ids(self.prefix_token)

    @property
    def middle_token(self):
        """
        This method 'middle_token' is a property method defined in the class 'CodeLlamaTokenizer' that
        retrieves the middle token stored in the instance.

        Args:
            self (CodeLlamaTokenizer): The instance of the CodeLlamaTokenizer class.
                This parameter refers to the current instance of the class.

        Returns:
            None: This method returns the middle token stored in the instance.
                If no middle token is set, it returns None.

        Raises:
            None.
        """
        return self._middle_token

    @property
    def middle_id(self):
        """
        Get the middle ID of the CodeLlamaTokenizer instance.

        Args:
            self (CodeLlamaTokenizer): The instance of the CodeLlamaTokenizer class.

        Returns:
            None: If the middle token is None.

        Raises:
            None.

        This method returns the middle ID of the CodeLlamaTokenizer instance.
        If the middle token is None, it returns None.
        The middle ID is obtained by converting the middle token to its corresponding ID using the
        'convert_tokens_to_ids' method.
        """
        if self._middle_token is None:
            return None
        return self.convert_tokens_to_ids(self.middle_token)

    @property
    def suffix_token(self):
        """
        Method to retrieve the suffix token associated with the CodeLlamaTokenizer instance.

        Args:
            self (CodeLlamaTokenizer): The instance of CodeLlamaTokenizer.
                This parameter refers to the instance of the CodeLlamaTokenizer class on which the method is being called.

        Returns:
            None: This method returns the suffix token corresponding to the CodeLlamaTokenizer instance.
                The suffix token is a property value associated with the instance.

        Raises:
            None
        """
        return self._suffix_token

    @property
    def suffix_id(self):
        """
        Returns the ID of the suffix token.

        Args:
            self (CodeLlamaTokenizer): The instance of the CodeLlamaTokenizer class.

        Returns:
            None: If the suffix token is None.

        Raises:
            None.

        This method retrieves the ID corresponding to the suffix token.
        If the suffix token is None, the method returns None.
        The suffix token is obtained by converting the suffix token to its corresponding ID using
        the convert_tokens_to_ids method.
        """
        if self._suffix_token is None:
            return None
        return self.convert_tokens_to_ids(self.suffix_token)

    @property
    def eot_token(self):
        """
        This method 'eot_token' in the class 'CodeLlamaTokenizer' retrieves the end-of-text token.

        Args:
            self (CodeLlamaTokenizer): The instance of the CodeLlamaTokenizer class.

        Returns:
            None: This method returns the end-of-text token value stored in the instance.

        Raises:
            None.
        """
        return self._eot_token

    @property
    def eot_id(self):
        """
        This method 'eot_id' is a property in the 'CodeLlamaTokenizer' class.

        Args:
            self: The instance of the 'CodeLlamaTokenizer' class.

        Returns:
            None: If the '_eot_token' attribute is None, the method returns None.
            int: If the '_eot_token' attribute is not None, the method returns the integer value obtained
                by converting the token to its corresponding ID using the 'convert_tokens_to_ids' method.

        Raises:
            No specific exceptions are raised by this method.
        """
        if self._eot_token is None:
            return None
        return self.convert_tokens_to_ids(self.eot_token)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.get_vocab
    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, prefix, suffix=None, suffix_first=False, **kwargs) -> List[int]:
        """
        Tokenizes the given prefix and suffix to generate a list of integers representing tokens.

        Args:
            self (CodeLlamaTokenizer): An instance of the CodeLlamaTokenizer class.
            prefix (str): The prefix string to tokenize.
            suffix (str, optional): The suffix string to tokenize. Defaults to None.
            suffix_first (bool, optional): Flag indicating whether to place the suffix before the prefix.
                Defaults to False.

        Returns:
            List[int]: A list of integers representing the tokens generated from the prefix and suffix.

        Raises:
            ValueError: If the input includes a prefix and a suffix used for the infilling task,
                or if the model does not support infilling.

        Note:
            - The `prefix` and `suffix` parameters are used to split the input on the `fill_token` token to
            create a suffix and prefix.
            - If only a prefix is provided, the method tokenizes the prefix and returns the resulting tokens.
            - If a prefix and suffix are provided, the method tokenizes both and returns the tokens in the
            specified order.
            - The `suffix_first` parameter takes precedence over the class attribute `suffix_first`
            if both are provided.
            - The method removes special tokens from the beginning of the tokens list if they match the
            specified conditions.
            - The method replaces occurrences of the `SPIECE_UNDERLINE` token in the prefix with a space.
        """
        # add a prefix space to `prefix`
        if self.fill_token is not None and self.fill_token in prefix and suffix is None:
            prefix, suffix = prefix.split(self.fill_token)

        if len(prefix) > 0:
            prefix = SPIECE_UNDERLINE + prefix.replace(SPIECE_UNDERLINE, " ")

        if suffix is None or len(suffix) < 1:
            tokens = super().tokenize(prefix, **kwargs)
            if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
                tokens = tokens[1:]
            return tokens

        prefix_tokens = self._tokenize(prefix)  # prefix has an extra `SPIECE_UNDERLINE`

        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            raise ValueError(
                "The input either includes a `prefix` and a `suffix` used for the infilling task,"
                f"  or can be split on the {self.fill_token} token, creating a suffix and prefix,"
                " but the model does not support `infilling`."
            )
        suffix_tokens = self._tokenize(suffix)  # make sure CodeLlama sp model does not mess up

        suffix_first = suffix_first if suffix_first is not None else self.suffix_first
        if suffix_first:
            # format as " <PRE> <SUF>{suf} <MID> {pre}"
            return [self.prefix_token, self.suffix_token] + suffix_tokens + [self.middle_token] + prefix_tokens
        # format as " <PRE> {pre} <SUF>{suf} <MID>"
        return [self.prefix_token] + prefix_tokens + [self.suffix_token] + suffix_tokens + [self.middle_token]

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        tokens = self.sp_model.encode(text, out_type=str)
        if not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens
        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        for _, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.save_vocabulary
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
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Method to build inputs with special tokens in the CodeLlamaTokenizer class.

        Args:
            self: Reference to the current instance of the class.
            token_ids_0 (list): List of token IDs for the first input sequence.
            token_ids_1 (list, optional): List of token IDs for the second input sequence. Defaults to None.

        Returns:
            list: A list representing the input sequences with special tokens added based on the configuration settings.

        Raises:
            None.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

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

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    @property
    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.default_chat_template
    def default_chat_template(self):
        """
        LLaMA uses [INST] and [/INST] to indicate user messages, and <<SYS>> and <</SYS>> to indicate system messages.
        Assistant messages do not have special tokens, because LLaMA chat models are generally trained with strict
        user/assistant/user/assistant message ordering, and so assistant messages can be identified from the ordering
        rather than needing special tokens. The system message is partly 'embedded' in the first user message, which
        results in an unusual token ordering when it is present. This template should definitely be changed if you wish
        to fine-tune a model with more flexible role ordering!

        The output should look something like:

            <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos><bos>[INST] Prompt [/INST] Answer <eos>
            <bos>[INST] Prompt [/INST]

        The reference for this chat template is [this code
        snippet](https://github.com/facebookresearch/llama/blob/556949fdfb72da27c2f4a40b7f0e4cf0b8153a28/llama/generation.py#L320-L362)
        in the original repository.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://hf-mirror.com/docs/transformers/main/chat_templating for more information.\n"
        )
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
        template = template.replace("USE_DEFAULT_PROMPT", "true" if self.use_default_system_prompt else "false")
        default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

        return template

    def __getstate__(self):
        """
        Method: __getstate__

        Description:
            This method is used to retrieve the state of the CodeLlamaTokenizer object for serialization purposes.
            It returns a dictionary representing the current state of the object.

        Args:
            self: The instance of the CodeLlamaTokenizer class.

        Returns:
            None: This method does not return any value. Instead, it modifies the state dictionary and returns None.

        Raises:
            None.
        """
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        """
        This method '__setstate__' is defined within the 'CodeLlamaTokenizer' class to set the internal state of the
        object based on the provided dictionary 'd'. It reconstructs the object's state including the SentencePiece
        model by loading it from a serialized proto.

        Args:
            self (CodeLlamaTokenizer): The instance of the CodeLlamaTokenizer class.
            d (dict): A dictionary containing the state information to be set.
                It should include the necessary attributes for the object's state reconstruction.

        Returns:
            None: This method does not return any value explicitly.
                It operates by modifying the internal state of the object.

        Raises:
            None:
                However, potential exceptions that could be raised during the execution may include but are not limited to:

                - TypeError: If the input 'd' is not a dictionary.
                - ValueError: If the input 'd' does not contain the required state information.
                - Any exceptions related to the SentencePieceProcessor initialization or loading process.
        """
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

__all__ = ['CodeLlamaTokenizer']
