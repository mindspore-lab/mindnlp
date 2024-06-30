# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for CpmBee."""
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict

from mindnlp.utils import logging
from ...tokenization_utils import PaddingStrategy, PreTrainedTokenizer, TensorType
from ...tokenization_utils_base import AddedToken, BatchEncoding, TextInput, TruncationStrategy


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openbmb/cpm-bee-10b": "https://hf-mirror.com/openbmb/cpm-bee-10b/blob/main/vocab.txt",
        "openbmb/cpm-bee-5b": "https://hf-mirror.com/openbmb/cpm-bee-5b/blob/main/vocab.txt",
        "openbmb/cpm-bee-2b": "https://hf-mirror.com/openbmb/cpm-bee-2b/blob/main/vocab.txt",
        "openbmb/cpm-bee-1b": "https://hf-mirror.com/openbmb/cpm-bee-1b/blob/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openbmb/cpm-bee-10b": 4096,
    "openbmb/cpm-bee-5b": 4096,
    "openbmb/cpm-bee-2b": 4096,
    "openbmb/cpm-bee-1b": 4096,
}


class _PrevExtTableStates(TypedDict):

    """
    Represents the state of a previous external table.
    
    This class inherits from TypedDict and provides a structured way to store the state information of a previous
    external table. The class '_PrevExtTableStates' can be used to store and retrieve various attributes related
    to the state of the table.

    Attributes:
        <attribute1>: <description of attribute1>
        <attribute2>: <description of attribute2>
        ...
        <attributeN>: <description of attributeN>

    Note:
        The '_PrevExtTableStates' class is intended to be used in conjunction with external tables and provides
        a convenient way to manage and retrieve the state information associated with them.

    Example:
        ```python
        >>> # Create an instance of '_PrevExtTableStates'
        >>> table_state = _PrevExtTableStates()
        ...
        >>> # Set the attributes of the table state
        >>> table_state['name'] = 'my_table'
        >>> table_state['last_modified'] = '2022-01-01 12:00:00'
        >>> table_state['column_count'] = 5
        ...
        >>> # Retrieve the attributes of the table state
        >>> table_name = table_state['name']
        >>> last_modified = table_state['last_modified']
        >>> column_count = table_state['column_count']
        ```

    Inheritance:
        TypedDict
    """
    ext_table: Dict[int, str]
    token_id_table: Dict[str, Dict[int, int]]


CPMBeeInputType = Union[str, Dict[str, "CPMBeeInputType"]]


def rel_to_bucket(n_up: int, n_down: int, max_depth: int = 8):
    """
    Calculates the relative position of an item in a bucket based on the number of items above and below it.

    Args:
        n_up (int): The number of items above the item.
        n_down (int): The number of items below the item.
        max_depth (int, optional): The maximum depth of the bucket. Defaults to 8.

    Returns:
        int: The relative position of the item in the bucket.

    Raises:
        None.

    """
    ret = n_up * max_depth + n_down
    if ret == 0:
        return ret
    else:
        # bucket 1 is reserved for incontext samples
        return ret + 1


class _DictTree(TypedDict):

    """A Python class representing a dictionary-like tree structure.

    The _DictTree class is a subclass of TypedDict, which allows for the creation of dictionary-like objects that
    have a fixed set of keys and specific value types for each key.
    The _DictTree class provides additional functionality to represent a hierarchical tree structure.

    Attributes:
        None

    Methods:
        None

    Inheritance:
        TypedDict:
            A built-in class in the typing module that provides a way to specify the expected structure of a dictionary-like object.

    Note:
        The _DictTree class is not meant to be instantiated directly, but rather serves as a base class for creating custom tree structures.

    Usage:
        To create a custom tree structure using the _DictTree class, simply inherit from it and define the desired
        tree structure by specifying the keys and value types using the TypedDict syntax.

    Example:
        ```python
        >>> class MyTree(_DictTree):
        >>>     name: str
        >>>     children: List[MyTree]
        ...
        >>>     def __init__(self, name: str, children: List[MyTree]) -> None:
        >>>         self.name = name
        >>>         self.children = children
        ...
        >>>     def add_child(self, child: MyTree) -> None:
        >>>         self.children.append(child)
        ...
        >>>     def remove_child(self, child: MyTree) -> None:
        >>>         self.children.remove(child)

        >>>     def get_children(self) -> List[MyTree]:
        >>>         return self.children
        ```
        In the above example, a custom tree structure is defined with two keys: 'name' of type str and 'children' of
        type List[MyTree].
        The class provides methods to add and remove children, as well as retrieve the list of children.

    """
    value: str
    children: List["_DictTree"]
    depth: int
    segment_id: int
    need_predict: bool


class CpmBeeTokenizer(PreTrainedTokenizer):
    r"""
    Construct a CPMBee tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        line_token (`str`, *optional*, defaults to `"\n"`):
            The line token.
        space_token (`str`, *optional*, defaults to `" "`):
            The space token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The mask token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        padding_side (`str`, *optional*, defaults to `"left"`):
            The padding side. CPM-Bee will use left padding by default.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names: List[str] = [
        "input_ids",
        "attention_mask",
        "input_id_sub",
        "position",
        "context",
        "sample_ids",
        "num_segments",
        "segment",
        "segment_rel_offset",
        "segment_rel",
    ]
    add_prefix_space = False

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        line_token="\n",
        space_token=" ",
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>",
        padding_side="left",
        **kwargs,
    ):
        """
        Initialize a CpmBeeTokenizer object.

        Args:
            vocab_file (str): The path to the file containing the vocabulary.
            bos_token (str, optional): The beginning of sentence token.
            eos_token (str, optional): The end of sentence token.
            line_token (str, optional): The token used to represent a new line.
            space_token (str, optional): The token used to represent a space.
            unk_token (str, optional): The token used to represent unknown words.
            mask_token (str, optional): The token used for masking.
            pad_token (str, optional): The token used for padding.
            padding_side (str, optional): The side to apply padding.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the vocab_file does not exist.
            TypeError: If any of the arguments are of incorrect type.
        """
        self.encoder: Dict[str, int] = {}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            line_token=line_token,
            space_token=space_token,
            unk_token=unk_token,
            mask_token=mask_token,
            pad_token=pad_token,
            padding_side=padding_side,
            **kwargs,
        )

        with open(vocab_file, "r", encoding="utf-8") as reader:
            for token in reader.readlines():
                token = token.rstrip("\n")
                if len(token) == 0:
                    continue
                self.encoder[token] = len(self.encoder)

        self.encoder[" "] = self.encoder["</_>"]
        self.encoder["\n"] = self.encoder["</n>"]
        del self.encoder["</_>"]
        del self.encoder["</n>"]

        self.decoder = {v: k for k, v in self.encoder.items()}

        self._max_word_len = max(len(x) for x in self.encoder.keys())
        self.cpmbee_special_tokens = {k: v for k, v in self.encoder.items() if k.startswith("<") and k.endswith(">")}

        self.ext_table: Dict[int, str] = {}
        self.ext_table_rev: Dict[str, int] = {}

        self.token_id_table: Dict[str, Dict[int, int]] = {}
        self.ext_special_tokens = []

        self.ext_args_for_model = [
            "input_id_subs",
            "input_pos",
            "context",
            "segment_ids",
            "segment_rel_offset",
            "segment_rel",
            "sample_ids",
            "num_segments",
            "predict_segments",
            "answer_placeholders",
            "ext_table",
            "token_id_table",
        ]

    @property
    def bod_token_id(self):
        """
        Returns the token ID for the beginning of document (BOD) token.

        Args:
            self: An instance of the CpmBeeTokenizer class.

        Returns:
            None: This method returns the token ID corresponding to the BOD token in the encoder dictionary.

        Raises:
            None.
        """
        return self.encoder[self.bod_token]

    @property
    def eod_token_id(self):
        """
        Method to retrieve the token ID corresponding to the end-of-document token in the CpmBeeTokenizer class.

        Args:
            self: An instance of the CpmBeeTokenizer class.

        Returns:
            None: The method returns the token ID of the end-of-document token in the tokenizer's encoder.

        Raises:
            None.
        """
        return self.encoder[self.eod_token]

    @property
    def newline_id(self):
        """
        Returns the ID of the newline token in the CpmBeeTokenizer.

        Args:
            self (CpmBeeTokenizer): An instance of the CpmBeeTokenizer class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.encoder[self.line_token]

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary used by the CpmBeeTokenizer instance.

        Args:
            self:
                The CpmBeeTokenizer instance.

                - This parameter is of type 'CpmBeeTokenizer'.
                - It represents the instance of the CpmBeeTokenizer class on which the method is called.

        Returns:
            int:
                An integer representing the size of the vocabulary.

                - The returned value represents the total number of unique tokens in the vocabulary.

        Raises:
            None.

        Example:
            ```python
            >>> tokenizer = CpmBeeTokenizer()
            >>> tokenizer.vocab_size()
            5000
            ```
        """
        return len(self.encoder)

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder)

    def get_vocab(self):
        """
        Get the vocabulary of the CpmBeeTokenizer instance.

        Args:
            self (CpmBeeTokenizer): The instance of the CpmBeeTokenizer class.
                This parameter represents the current instance of the tokenizer.

        Returns:
            dict: A dictionary containing the combined encoder and added tokens encoder.
                The keys represent tokens, and the values represent their corresponding IDs.

        Raises:
            None.
        """
        return dict(self.encoder, **self.added_tokens_encoder)

    def get_piece(self, text: str) -> str:
        """
        Match with maximum length.
        """
        len_text = len(text)
        for i in range(len(text)):
            sub = text[: len_text - i]
            if (sub in self.encoder) or (sub in self.added_tokens_encoder):
                return sub
        return text[0]

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        r"""
        Override the `tokenize` to meet the needs of CPMBee:

        1. Mark the special token with `<` and `>`. The `<>` will be ignored.
        2. Split sentences by the marked special tokens.
        3. Record the marked special token by `ext_table` and `ext_table_rev`.
        4. Tokenize the sentence without special tokens.
        """
        for_cpmbee = kwargs.get("for_cpmbee", False)
        all_special_tokens_extended = {
            str(t): t for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        }

        sentence_split = [""]
        is_special_token = False
        for i, c in enumerate(text):
            if is_special_token:
                if c == "<":
                    tail = sentence_split.pop(-1)
                    sentence_split[-1] += tail
                    sentence_split.append(c)
                    is_special_token = False
                elif c == ">":
                    # end of special token
                    sentence_split[-1] += c
                    if sentence_split[-1] == "<>":
                        continue
                    is_special_token = False
                    sentence_split.append("")
                else:
                    sentence_split[-1] += c
            else:
                if c == "<":
                    is_special_token = True
                    sentence_split.append(c)
                else:
                    sentence_split[-1] += c
        if is_special_token:
            tail = sentence_split.pop(-1)
            sentence_split[-1] += tail

        output_tokens = []
        for i, part in enumerate(sentence_split):
            if (i & 1) == 1:
                # special token
                output_tokens.append(part)
                if for_cpmbee and (part not in self.encoder) and (part not in self.ext_table_rev):
                    self.ext_table_rev[part] = len(self.ext_table_rev) + self.vocab_size
                    self.ext_table[self.ext_table_rev[part]] = part
            else:
                output_tokens.extend(self._tokenize(part, for_cpmbee=for_cpmbee))

        # drop spaces
        for i, token in enumerate(output_tokens):
            if token in self.added_tokens_encoder:
                token = all_special_tokens_extended.get(token, None)
                left = output_tokens[i - 1] if i > 0 else None
                right = output_tokens[i + 1] if i < len(output_tokens) - 1 else None
                if isinstance(token, AddedToken):
                    if token.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        output_tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if token.lstrip and left:
                        output_tokens[i - 1] = left.rstrip()  # Opposite here
                else:
                    if right:
                        output_tokens[i + 1] = right.lstrip()
                    if left:
                        output_tokens[i - 1] = left.rstrip()

        skipped_tokens = []
        for token in output_tokens:
            if not token:
                continue
            skipped_tokens.append(token)

        return skipped_tokens

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary.

        Do NOT take care of added tokens. Record the unk tokens and special tokens in `ext_table` and `ext_table_rev`.
        """
        for_cpmbee = kwargs.get("for_cpmbee", False)
        output_tokens = []

        part_st = 0
        last_unk = None
        while part_st < len(text):
            piece = self.get_piece(text[part_st:])
            if piece in self.encoder or self.added_tokens_encoder:
                if last_unk is None:
                    output_tokens.append(piece)
                else:
                    if for_cpmbee and (last_unk not in self.ext_table_rev):
                        self.ext_table_rev[last_unk] = len(self.ext_table_rev) + self.vocab_size
                        self.ext_table[self.ext_table_rev[last_unk]] = last_unk
                    output_tokens.append(last_unk)
                    output_tokens.append(piece)
                    last_unk = None
            else:
                if last_unk is None:
                    last_unk = piece
                else:
                    last_unk += piece
            part_st += len(piece)
        if last_unk is not None:
            # part end with UNK
            if for_cpmbee and (last_unk not in self.ext_table_rev):
                self.ext_table_rev[last_unk] = len(self.ext_table_rev) + self.vocab_size
                self.ext_table[self.ext_table_rev[last_unk]] = last_unk
            output_tokens.append(last_unk)

        return output_tokens

    def check(self, token):
        """
        Checks if a token is present in the encoder.

        Args:
            self (CpmBeeTokenizer): An instance of the CpmBeeTokenizer class.
            token (Any): The token to be checked in the encoder.

        Returns:
            None.

        Raises:
            None.
        """
        return token in self.encoder

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens into a single string.

        Args:
            self (CpmBeeTokenizer): An instance of the CpmBeeTokenizer class.
            tokens (List[str]): A list of tokens to be converted into a string.

        Returns:
            str: A string representation of the tokens.

        Raises:
            None.

        This method takes in two parameters, self and tokens. The self parameter is an instance of the CpmBeeTokenizer
        class and is used to access the class's attributes and methods. The tokens parameter is a
        list of strings representing individual tokens.

        The function returns a string that is obtained by concatenating all the tokens together using the ''.join() method.
        This method does not modify the original list of tokens.

        No exceptions are raised by this method.
        """
        return "".join(tokens)

    def _convert_token_to_id(self, token: str):
        """Converts a token (str) in an id using the vocab and ext_table."""
        if token in self.encoder:
            return self.encoder.get(token)
        elif token in self.ext_table_rev:
            return self.ext_table_rev[token]
        elif token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        else:
            return self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab and ext_table."""
        if index in self.ext_table:
            return self.ext_table[index]
        elif index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index]
        else:
            if index >= 0:
                return self.decoder[index]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a file.

        Args:
            self (CpmBeeTokenizer): The instance of the CpmBeeTokenizer class.
            save_directory (str): The directory where the vocabulary file will be saved.
            filename_prefix (Optional[str]): An optional prefix to prepend to the filename. Default is None.

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.

        Raises:
            IOError: If there is an issue with reading or writing the vocabulary file.
            ValueError: If the provided save_directory is not a valid directory.
            KeyError: If any of the keys used for encoding tokens are not found in the encoder dictionary.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        index = 0
        self.encoder["</n>"] = self.encoder["\n"]
        del self.encoder["\n"]
        self.encoder["</_>"] = self.encoder[" "]
        del self.encoder[" "]
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.encoder.items(), key=lambda x: x[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def __call__(self, text, *args, **kwargs):
        r"""
        CPMBee `call` method will use `_tokenize_cpmbee` when the input type is dict.
        """
        if isinstance(text, dict):
            return self._batch_tokenize_cpmbee([text], *args, **kwargs)
        elif isinstance(text, (list, tuple)):
            if isinstance(text[0], dict):
                return self._batch_tokenize_cpmbee(text, *args, **kwargs)
            else:
                return super().__call__(text, *args, **kwargs)
        else:
            return super().__call__(text, *args, **kwargs)

    # 分词
    def _tokenize_cpmbee(self, data: TextInput, *args, **kwargs) -> List[str]:
        """
        A tokenize method to process dict data. Exclusive for CPMBee.
        """
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, Dict):
            raise TypeError(
                "CpmBeeTokenizer input data should be dict or str in dict format, but got {}".format(type(data))
            )

        # 1. prepare answer placeholder
        answer_placeholders = []

        def _put_placeholder(data: Any, path: List[str] = []):
            if isinstance(data, dict):
                ret = {}
                for k, v in data.items():
                    ret[k] = _put_placeholder(v, path + [k])
                return ret
            else:
                answer_placeholders.append(path)
                return "<ans_{}>".format(len(answer_placeholders))

        data["<ans>"] = _put_placeholder(data["<ans>"])

        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,
        ) = self.convert_data_to_id(data, shuffle_answer=False, max_depth=8)

        # <ans> mapping from sub to id
        sub_ans_map: Dict[int, int] = {}
        for fake_id, token_sub in table_states["token_id_table"]["<ans>"].items():
            token = table_states["ext_table"][fake_id]
            if token.startswith("<ans_") and token.endswith(">"):
                ans_id = int(token[5:-1])
                sub_ans_map[token_sub] = ans_id

        tmp_input_ids = []
        tmp_input_sub = []
        tmp_input_seg = []

        # get predict segments
        predict_segments: List[Tuple[int, int]] = []
        for i in range(input_ids.shape[0]):
            if context[i] == 0:
                if input_ids[i] == self.encoder["<ans>"]:
                    # is ans
                    # (segment_id, ans_id)
                    predict_segments.append((segment_ids[i], sub_ans_map[input_id_subs[i]]))
            else:
                tmp_input_ids.append(input_ids[i])
                tmp_input_sub.append(input_id_subs[i])
                tmp_input_seg.append(segment_ids[i])

        if len(predict_segments) == 0:
            raise ValueError("No answer to predict")

        input_ids = np.array(tmp_input_ids, dtype=np.int32)  # all context
        input_id_subs = np.array(tmp_input_sub, dtype=np.int32)  # [0, 0, 0, 0, 1, 0, 0, 2, 0, ...]
        context = np.full_like(tmp_input_ids, 1, dtype=np.int8)  # [1, 1, 1, ...]
        segment_ids = np.array(tmp_input_seg, dtype=np.int32)  # [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, ...]
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)  # [0, 0, 0, 0, ...]
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)  # [0, 0, 0, ...]
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)  # [n_seg, n_seg, n_seg, ...]
        input_pos = np.arange(input_ids.shape[0], dtype=np.int32)  # [0, 1, 2, 3, 4, ...]

        return (
            self.prepare_for_model(
                input_ids.tolist(),
                input_id_subs=input_id_subs.tolist(),
                input_pos=input_pos.tolist(),
                context=context.tolist(),
                segment_ids=segment_ids.tolist(),
                segment_rel_offset=segment_rel_offset.tolist(),
                segment_rel=segment_rel.tolist(),
                sample_ids=sample_ids.tolist(),
                num_segments=num_segments.tolist(),
                **kwargs,
            ),
            predict_segments,
            answer_placeholders,
            table_states["ext_table"],
            table_states["token_id_table"],
        )

    def _batch_tokenize_cpmbee(self, data_lst, *args, **kwargs):
        """
        Batched _token_cpmbee.
        """
        return_tensors = kwargs.get("return_tensors", None)
        batch_outputs = {}
        segment_rel_pack = []
        other_info = []

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []

        for data in data_lst:
            self.ext_table = {}
            self.ext_table_rev = {}
            self.token_id_table = {}
            (outputs, predict_segments, answer_placeholders, ext_table, token_id_table) = self._tokenize_cpmbee(
                data,
                truncation=None,
                padding=PaddingStrategy.DO_NOT_PAD.value,
                max_length=None,
                pad_to_multiple_of=None,
                return_attention_mask=False,
                return_tensors=None,
            )
            rev_ext_table = {}
            for token, mp in token_id_table.items():
                if token == "<ans>":
                    continue
                token_id = self.encoder[token]
                for fake_id, token_sub in mp.items():
                    if token_sub > 0:
                        if (token_id, token_sub) not in batch_ext_table_map:
                            batch_ext_table_map[(token_id, token_sub)] = len(batch_ext_table_ids) + self.vocab_size
                            batch_ext_table_ids.append(token_id)
                            batch_ext_table_sub.append(token_sub)
                        rev_ext_table[batch_ext_table_map[(token_id, token_sub)]] = ext_table[fake_id]
                    else:
                        rev_ext_table[token_id] = ext_table[fake_id]

            segment_rel_pack.append(np.array(outputs.pop("segment_rel")))
            other_info.append(
                {
                    "predict_segments": predict_segments,
                    "answer_placeholders": answer_placeholders,
                    "ext_table": rev_ext_table,
                }
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        max_length = max(len(item) for item in batch_outputs[self.model_input_names[0]])
        batch_size = len(batch_outputs[self.model_input_names[0]])
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in batch_outputs.items()}

            for k, v in inputs.items():
                required_input = v

                needs_to_be_padded = len(required_input) != max_length

                if needs_to_be_padded:
                    difference = max_length - len(required_input)
                    batch_outputs[k][i] = [self.pad_token_id] * difference + required_input

        max_num_rels = 0
        for rel in segment_rel_pack:
            max_num_rels = max(max_num_rels, rel.shape[0])
        padded_rels = np.zeros((len(segment_rel_pack), max_num_rels), dtype=np.int32)
        for i, rel in enumerate(segment_rel_pack):
            padded_rels[i, : rel.shape[0]] = rel
        batch_outputs["segment_rel"] = padded_rels
        batch_outputs["batch_ext_table_ids"] = np.array(batch_ext_table_ids, dtype=np.int32)
        batch_outputs["batch_ext_table_sub"] = np.array(batch_ext_table_sub, dtype=np.int32)
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        batch_outputs["other_info"] = other_info

        return batch_outputs

    def convert_data_to_id(
        self,
        data: Any,
        prev_ext_states: Optional[_PrevExtTableStates] = None,
        shuffle_answer: bool = True,
        max_depth: int = 8,
    ):
        """
        Parse a dict to data ids. Exclusive for CPMBee. It will

        1. parse the dict to segments and get segment_rel, which for calculating of position_bias.
        2. tokenize every segment.
        """
        root: _DictTree = {
            "value": "<root>",
            "children": [],
            "depth": 0,
            "segment_id": 0,
            "need_predict": False,
        }

        segments = [root]

        def _build_dict_tree(data: CPMBeeInputType, depth: int, need_predict: bool) -> List[_DictTree]:
            if isinstance(data, dict):
                ret_list: List[_DictTree] = []
                curr_items = list(data.items())
                if need_predict and shuffle_answer:
                    access_idx = np.arange(len(curr_items))
                    np.random.shuffle(access_idx)
                    curr_items = [curr_items[idx] for idx in access_idx]
                for k, v in curr_items:
                    child_info: _DictTree = {
                        "value": k,
                        "children": [],
                        "depth": depth,
                        "segment_id": len(segments),
                        "need_predict": False,  # only leaves are contexts
                    }
                    segments.append(child_info)
                    child_info["children"] = _build_dict_tree(
                        v, depth + 1, need_predict or (depth == 1 and k == "<ans>")
                    )  # elements in <root>.<ans>

                    ret_list.append(child_info)
                return ret_list
            else:
                assert isinstance(data, str), "Invalid data {}".format(data)
                ret: _DictTree = {
                    "value": data,
                    "children": [],
                    "depth": depth,
                    "segment_id": len(segments),
                    "need_predict": need_predict,
                }
                segments.append(ret)
                return [ret]

        root["children"] = _build_dict_tree(data, 1, False)

        num_segments = len(segments)
        segment_rel = np.zeros((num_segments * num_segments,), dtype=np.int32)

        def _build_segment_rel(node: _DictTree) -> List[Tuple[int, int]]:
            ret: List[Tuple[int, int]] = [(node["segment_id"], node["depth"])]
            for child in node["children"]:
                sub = _build_segment_rel(child)
                for seg_id_1, depth_1 in sub:
                    for seg_id_2, depth_2 in ret:
                        n_up = min(depth_1 - node["depth"], max_depth - 1)
                        n_down = min(depth_2 - node["depth"], max_depth - 1)
                        segment_rel[seg_id_1 * num_segments + seg_id_2] = rel_to_bucket(
                            n_up, n_down, max_depth=max_depth
                        )
                        segment_rel[seg_id_2 * num_segments + seg_id_1] = rel_to_bucket(
                            n_down, n_up, max_depth=max_depth
                        )
                ret.extend(sub)
            return ret

        _build_segment_rel(root)

        input_ids: List[int] = []
        input_id_subs: List[int] = []
        segment_bound: List[Tuple[int, int]] = []

        if prev_ext_states is not None:
            self.ext_table = prev_ext_states["ext_table"]
            self.token_id_table = prev_ext_states["token_id_table"]

        for seg in segments:
            # tokenize
            tokens = self.convert_tokens_to_ids(self.tokenize(seg["value"], for_cpmbee=True))

            token_id_subs = []
            reid_token_ids = []
            for idx in tokens:
                if idx in self.ext_table:
                    # unk or special token
                    token = self.ext_table[idx]
                    if token.startswith("<") and token.endswith(">"):
                        # special token
                        if "_" in token:
                            token_name = token[1:-1].split("_", maxsplit=1)[0]
                        else:
                            token_name = token[1:-1]
                        token_name = "<{}>".format(token_name)
                    else:
                        token_name = "<unk>"

                    if token_name not in self.token_id_table:
                        self.token_id_table[token_name] = {}
                    if idx not in self.token_id_table[token_name]:
                        self.token_id_table[token_name][idx] = len(self.token_id_table[token_name])
                    if token_name not in self.encoder:
                        raise ValueError("Invalid token {}".format(token))
                    reid_token_ids.append(self.encoder[token_name])
                    token_id_subs.append(self.token_id_table[token_name][idx])
                else:
                    reid_token_ids.append(idx)
                    token_id_subs.append(0)
            tokens = [self.bos_token_id] + reid_token_ids
            token_id_subs = [0] + token_id_subs
            # eos_id 表示 no need_predict
            if not seg["need_predict"]:  # eos
                tokens = tokens + [self.eos_token_id]
                token_id_subs = token_id_subs + [0]
            else:
                # no eos
                pass
            begin = len(input_ids)
            input_ids.extend(tokens)
            input_id_subs.extend(token_id_subs)
            end = len(input_ids)
            segment_bound.append((begin, end))

        ids = np.array(input_ids, dtype=np.int32)
        id_subs = np.array(input_id_subs, dtype=np.int32)
        segs = np.zeros((ids.shape[0],), dtype=np.int32)  # 按segment_bound对seg编号
        context = np.zeros((ids.shape[0],), dtype=np.int8)
        for i, (begin, end) in enumerate(segment_bound):
            if not segments[i]["need_predict"]:
                context[begin:end] = 1
            segs[begin:end] = i

        curr_ext_table_states: _PrevExtTableStates = {
            "ext_table": self.ext_table,
            "token_id_table": self.token_id_table,
        }
        return ids, id_subs, context, segs, segment_rel, num_segments, curr_ext_table_states

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        # for CPMBee, encode all the model arguments
        for arg in self.ext_args_for_model:
            v = kwargs.get(arg, None)
            if v is not None:
                encoded_inputs[arg] = v

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def prepare_for_finetune(
        self,
        data_list: List[Dict],
        max_length: int = 2048
    ):
        """
        Prepares the input data for fine-tuning.
        
        Args:
            self (CpmBeeTokenizer): The instance of the CpmBeeTokenizer class.
            data_list (List[Dict]): A list of dictionaries containing the input data.
            max_length (int, optional): The maximum length of the input data. Defaults to 2048.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        _inputs: List[NDArray[np.int32]] = []
        _inputs_sub: List[NDArray[np.int32]] = []
        _context: List[NDArray[np.int8]] = []
        _sample_ids: List[NDArray[np.int32]] = []
        _segments: List[NDArray[np.int32]] = []
        _num_segments: List[NDArray[np.int32]] = []
        _segment_rel_offset: List[NDArray[np.int32]] = []
        _segment_rel: List[NDArray[np.int32]] = []
        _spans: List[List[int]] = []
        _raw_data: List[List[Any]] = []

        raw_data = {}
        for data in data_list:
            (
                input_ids,
                input_id_subs,
                context,
                segment_ids,
                segment_rel,
                n_segments,
                _
            ) = self.convert_data_to_id(data)

            input_ids = input_ids[: max_length]
            context = context[: max_length]
            segment_ids = segment_ids[: max_length]
            raw_data["input"] = data
            raw_data["samples"] = []

            sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
            segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
            num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)

            _inputs.append(input_ids)
            _inputs_sub.append(input_id_subs)
            _context.append(context)
            _sample_ids.append(sample_ids)
            _segments.append(segment_ids)
            _num_segments.append(num_segments)
            _segment_rel_offset.append(segment_rel_offset)
            _segment_rel.append(segment_rel)
            _spans.append([input_ids.shape[0]])
            _raw_data.append([raw_data])

        batch_size = len(_inputs)
        inputs = np.zeros((batch_size, max_length), dtype=np.int32)
        inputs_sub = np.zeros((batch_size, max_length), dtype=np.int32)
        context = np.zeros((batch_size, max_length), dtype=np.int8)
        sample_ids = np.zeros((batch_size, max_length), dtype=np.int32)
        segments = np.zeros((batch_size, max_length), dtype=np.int32)
        num_segments = np.zeros((batch_size, max_length), dtype=np.int32)
        segment_rel_offset = np.zeros((batch_size, max_length), dtype=np.int32)
        tgt = np.full((batch_size, max_length), -100, dtype=np.int32)

        max_rel = 0
        for i in range(batch_size):
            max_rel = max(max_rel, _segment_rel[i].shape[0])
        segment_rel = np.zeros((batch_size, max_rel), dtype=np.int32)
        spans = np.zeros((batch_size, max_length), dtype=np.int32)
        length = np.zeros((batch_size,), dtype=np.int32)

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []
        raw_data_list: List[Any] = []

        for i in range(batch_size):
            instance_length = _inputs[i].shape[0]
            rel_size = _segment_rel[i].shape[0]
            inputs[i, :instance_length] = _inputs[i]
            inputs_sub[i, :instance_length] = _inputs_sub[i]
            context[i, :instance_length] = _context[i]
            sample_ids[i, :instance_length] = _sample_ids[i]
            segments[i, :instance_length] = _segments[i]
            num_segments[i, :instance_length] = _num_segments[i]
            segment_rel_offset[i, :instance_length] = _segment_rel_offset[i]
            segment_rel[i, :rel_size] = _segment_rel[i]

            span_begin = 0
            for span_id, span_end in enumerate(_spans[i]):
                spans[i, span_begin:span_end] = span_id
                span_begin = span_end
            length[i] = instance_length
            raw_data_list.extend(_raw_data[i])

            for j in range(instance_length):
                idx, idx_sub = _inputs[i][j], _inputs_sub[i][j]
                tgt_idx = idx
                if idx_sub > 0:
                    # need to be in ext table
                    if (idx, idx_sub) not in batch_ext_table_map:
                        batch_ext_table_map[(idx, idx_sub)] = len(batch_ext_table_map)
                        batch_ext_table_ids.append(idx)
                        batch_ext_table_sub.append(idx_sub)
                    tgt_idx = batch_ext_table_map[(idx, idx_sub)] + self.vocab_size
                if j > 1 and context[i, j - 1] == 0:
                    if idx != self.bos_token_id:
                        tgt[i, j - 1] = tgt_idx
                    else:
                        tgt[i, j - 1] = self.eos_token_id
            if context[i, instance_length - 1] == 0:
                tgt[i, instance_length - 1] = self.eos_token_id

        if len(batch_ext_table_map) == 0:
            # placeholder
            batch_ext_table_ids.append(0)
            batch_ext_table_sub.append(1)

        return BatchEncoding({
            "input_ids": inputs,
            "input_id_sub": inputs_sub,
            "length": length,
            "context": context > 0,
            "sample_ids": sample_ids,
            "num_segments": num_segments,
            "segment": segments,
            "segment_rel_offset": segment_rel_offset,
            "segment_rel": segment_rel,
            "span": spans,
            "labels": tgt,
            "ext_table_ids": np.array(batch_ext_table_ids, dtype=np.int32),
            "ext_table_sub": np.array(batch_ext_table_sub, dtype=np.int32)
        }, tensor_type="ms")

__all__ = ['CpmBeeTokenizer']
