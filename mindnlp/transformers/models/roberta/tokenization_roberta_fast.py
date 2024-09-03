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
"""Fast Tokenization classes for RoBERTa."""
import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers, processors

from mindnlp.utils import logging
from .tokenization_roberta import RobertaTokenizer
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "roberta-base": "https://hf-mirror.com/roberta-base/resolve/main/vocab.json",
        "roberta-large": "https://hf-mirror.com/roberta-large/resolve/main/vocab.json",
        "roberta-large-mnli": "https://hf-mirror.com/roberta-large-mnli/resolve/main/vocab.json",
        "distilroberta-base": "https://hf-mirror.com/distilroberta-base/resolve/main/vocab.json",
        "roberta-base-openai-detector": "https://hf-mirror.com/roberta-base-openai-detector/resolve/main/vocab.json",
        "roberta-large-openai-detector": (
            "https://hf-mirror.com/roberta-large-openai-detector/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "roberta-base": "https://hf-mirror.com/roberta-base/resolve/main/merges.txt",
        "roberta-large": "https://hf-mirror.com/roberta-large/resolve/main/merges.txt",
        "roberta-large-mnli": "https://hf-mirror.com/roberta-large-mnli/resolve/main/merges.txt",
        "distilroberta-base": "https://hf-mirror.com/distilroberta-base/resolve/main/merges.txt",
        "roberta-base-openai-detector": "https://hf-mirror.com/roberta-base-openai-detector/resolve/main/merges.txt",
        "roberta-large-openai-detector": (
            "https://hf-mirror.com/roberta-large-openai-detector/resolve/main/merges.txt"
        ),
    },
    "tokenizer_file": {
        "roberta-base": "https://hf-mirror.com/roberta-base/resolve/main/tokenizer.json",
        "roberta-large": "https://hf-mirror.com/roberta-large/resolve/main/tokenizer.json",
        "roberta-large-mnli": "https://hf-mirror.com/roberta-large-mnli/resolve/main/tokenizer.json",
        "distilroberta-base": "https://hf-mirror.com/distilroberta-base/resolve/main/tokenizer.json",
        "roberta-base-openai-detector": (
            "https://hf-mirror.com/roberta-base-openai-detector/resolve/main/tokenizer.json"
        ),
        "roberta-large-openai-detector": (
            "https://hf-mirror.com/roberta-large-openai-detector/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}


class RobertaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" RoBERTa tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2
    tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import RobertaTokenizerFast
        ...
        >>> tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        >>> tokenizer("Hello world")["input_ids"]
        [0, 31414, 232, 2]
        >>> tokenizer(" Hello world")["input_ids"]
        [0, 20920, 232, 2]
        ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

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
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = RobertaTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs,
    ):
        """
        Initializes a new instance of the `RobertaTokenizerFast` class.

        Args:
            self: The instance of the class itself.
            vocab_file (str, optional): The path to the vocabulary file. Default is None.
            merges_file (str, optional): The path to the merges file. Default is None.
            tokenizer_file (str, optional): The path to the tokenizer file. Default is None.
            errors (str, optional): Specifies the error handling during tokenization. Default is 'replace'.
            bos_token (str, optional): The beginning of sentence token. Default is '<s>'.
            eos_token (str, optional): The end of sentence token. Default is '</s>'.
            sep_token (str, optional): The separator token. Default is '</s>'.
            cls_token (str, optional): The classification token. Default is '<s>'.
            unk_token (str, optional): The unknown token. Default is '<unk>'.
            pad_token (str, optional): The padding token. Default is '<pad>'.
            mask_token (str or AddedToken, optional): The masking token. Default is '<mask>'.
            add_prefix_space (bool, optional): Specifies if a space should be added as a prefix to each token.
                Default is False.
            trim_offsets (bool, optional): Specifies if offsets should be trimmed. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            None.
        """
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # The lists 'sep' and 'cls' must be cased in tuples for the object `post_processor_class`
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

    @property
    def mask_token(self) -> str:
        """
        Return:
            `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
            having been set.

        Roberta tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *<mask>*.
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on Roberta.
        """
        # Mask token behave like a normal word, i.e. include the space before it
        # So we set lstrip to True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        This method, _batch_encode_plus, is a part of the RobertaTokenizerFast class and is responsible for batch
        encoding inputs.

        Args:
            self: This parameter represents the instance of the class and is required for accessing the class
                attributes and methods.

        Returns:
            BatchEncoding: This method returns a BatchEncoding object that contains the batch-encoded inputs.

        Raises:
            AssertionError: This method may raise an AssertionError if the condition 'self.add_prefix_space or not
                is_split_into_words' is not met, indicating that the class needs to be instantiated with
                add_prefix_space=True to use it with pretokenized inputs.
        """
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        Encodes the inputs into a batch of tokenized sequences using the fast version of the Roberta tokenizer.

        Args:
            self (RobertaTokenizerFast): An instance of the RobertaTokenizerFast class.

        Returns:
            BatchEncoding: A dictionary-like object containing the encoded sequences.

        Raises:
            AssertionError: If `is_split_into_words` is `True` but `add_prefix_space` is `False`.

        Note:
            This method is intended to be used with pretokenized inputs. To use it with pretokenized inputs,
            the `add_prefix_space` parameter of the `RobertaTokenizerFast` class should be set to `True`.
        """
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary to disk.

        Args:
            self (RobertaTokenizerFast): An instance of the RobertaTokenizerFast class.
            save_directory (str): The directory path where the vocabulary files will be saved.
            filename_prefix (Optional[str], default=None): An optional prefix to add to the filenames of the
                vocabulary files. If not provided, no prefix will be added.

        Returns:
            Tuple[str]: A tuple containing the filenames of the saved vocabulary files.

        Raises:
            None.

        Note:
            The saved vocabulary files will be stored in the specified directory with the following filenames:

            - If a filename prefix is provided, the files will be named as: "{filename_prefix}_vocab.json" and
            "{filename_prefix}_merges.txt".
            - If no filename prefix is provided, the files will be named as: "vocab.json" and "merges.txt".
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Builds inputs with special tokens for the RobertaTokenizerFast class.

        Args:
            self (RobertaTokenizerFast): The instance of the RobertaTokenizerFast class.
            token_ids_0 (List[int]): The list of token IDs for the first sequence.
            token_ids_1 (List[int], optional): The list of token IDs for the second sequence. Defaults to None.

        Returns:
            None

        Raises:
            None

        Description:
            This method takes in two sequences of token IDs, token_ids_0 and token_ids_1, and builds a new list of
            token IDs with special tokens added. The special tokens include the beginning of sequence
            (bos_token_id) and the end of sequence (eos_token_id).

            The method first adds the bos_token_id to the beginning of the token_ids_0 list, followed by all the
            token IDs in token_ids_0, and then adds the eos_token_id to the end of the list.
            If token_ids_1 is provided, the method appends the eos_token_id, followed by all the token IDs in
            token_ids_1, and finally adds another eos_token_id to the end of the list.

            If token_ids_1 is not provided, the method simply returns the list output containing the special tokens
            and token_ids_0. If token_ids_1 is provided, the method returns the list output containing the
            special tokens, token_ids_0, special tokens, and token_ids_1.

        Example:
            ```python
            >>> tokenizer = RobertaTokenizerFast()
            >>> token_ids_0 = [10, 20, 30]
            >>> token_ids_1 = [40, 50, 60]
            >>> output = tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
            >>> print(output)
            >>> # Output: [0, 10, 20, 30, 2, 2, 40, 50, 60, 2]
            ```

        Note:
            The bos_token_id and eos_token_id are specific token IDs used to mark the beginning and end of a sequence
            respectively.
        """
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

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

__all__ = ['RobertaTokenizerFast']
