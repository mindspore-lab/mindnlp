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
"""Tokenization classes for ChatGLM."""
import os
from typing import List, Optional, Union, Dict
import numpy as np
import sentencepiece as spm

from mindnlp.utils import logging, PaddingStrategy
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import EncodedInput, BatchEncoding

logger = logging.get_logger(__name__)

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "THUDM/chatglm-6b": 2048,
}


class TextTokenizer:
    """TextTokenizer"""
    def __init__(self, model_path):
        """
        Initializes a TextTokenizer object with the provided model path.
        
        Args:
            self: The TextTokenizer object.
            model_path (str): The path to the model file. This file contains the pre-trained sentence piece model.
                The model file must be in the SentencePiece format (.model file extension).
                This parameter is required to successfully initialize the TextTokenizer object.
            
        Returns:
            None.
            
        Raises:
            FileNotFoundError: If the provided model_path does not exist or is not accessible.
            TypeError: If the provided model_path is not a string or is of an unsupported type.
        
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.num_tokens = self.sp.vocab_size()

    def encode(self, text):
        """
        Encode the given text using the SentencePiece tokenizer.
        
        Args:
            self (TextTokenizer): The instance of the TextTokenizer class.
            text (str): The input text to be encoded.
            
        Returns:
            None: This method does not return any value directly.
                Instead, it encodes the input text using the SentencePiece tokenizer.
        
        Raises:
            None.
        """
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]):
        """
        Decodes a list of integer IDs into a text sequence using the SentencePiece tokenizer.
        
        Args:
            self (TextTokenizer): An instance of the TextTokenizer class.
            ids (List[int]): A list of integer IDs representing a text sequence.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.sp.DecodeIds(ids)

    def tokenize(self, text):
        """
        Tokenizes the input text using the SentencePiece tokenizer.
        
        Args:
            self (TextTokenizer): An instance of the TextTokenizer class.
            text (str): The input text to be tokenized.
            
        Returns:
            None.
        
        Raises:
            This method does not raise any exceptions.
        """
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a list of tokens into a string using the SentencePiece model.
        
        Args:
            self (TextTokenizer): An instance of the TextTokenizer class.
            tokens (list): A list of tokens to be converted into a string.
        
        Returns:
            None: This method does not return any value. The conversion result is directly applied within the method.
        
        Raises:
            None.
        """
        return self.sp.DecodePieces(tokens)

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a list of tokens into their corresponding ids using the TextTokenizer.
        
        Args:
            self (TextTokenizer): An instance of the TextTokenizer class.
                Represents the current instance of the TextTokenizer.
            tokens (list): A list of tokens to be converted into ids.
                Each token is a string representing a linguistic unit.
        
        Returns:
            None: This method does not return a value directly but updates the internal state of the TextTokenizer instance
                by converting the tokens into their corresponding ids.
        
        Raises:
            None.
        """
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        """
        Converts a token to its corresponding ID using the TextTokenizer.
        
        Args:
            self (TextTokenizer): The instance of the TextTokenizer class.
            token (str): The token to be converted to its corresponding ID.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the token is not a string type.
            ValueError: If the token does not exist in the tokenizer's vocabulary.
        """
        return self.sp.PieceToId(token)

    def convert_id_to_token(self, idx):
        """
        Converts an index to its corresponding token.
        
        Args:
            self (TextTokenizer): The instance of the TextTokenizer class.
            idx (int): The index of the token to be converted. It should be a non-negative integer.
        
        Returns:
            None: This method does not return any value. It modifies the internal state of the TextTokenizer instance.
        
        Raises:
            None.
        """
        return self.sp.IdToPiece(idx)

    def __len__(self):
        """
        This method returns the number of tokens in the TextTokenizer object.
        
        Args:
            self (TextTokenizer): The instance of the TextTokenizer class.
            
        Returns:
            int: The number of tokens in the TextTokenizer object.
        
        Raises:
            None.
        """
        return self.num_tokens


class SPTokenizer:
    """SPTokenizer"""
    def __init__(
            self,
            vocab_file,
            num_image_tokens=20000,
            max_blank_length=80,
            byte_fallback=True,
    ):
        """
        __init__ method for SPTokenizer class.
        
        Args:
            self: SPTokenizer object.
                The instance of the SPTokenizer class.
        
            vocab_file: str
                The path to the vocabulary file. It should not be None.
        
            num_image_tokens: int, optional
                The number of image tokens. Default is 20000.
        
            max_blank_length: int, optional
                The maximum length of the blank. Default is 80.
        
            byte_fallback: bool, optional
                Determines whether to fallback to byte encoding if character encoding fails. Default is True.
        
        Returns:
            None.

        Raises:
            AssertionError
                If the vocab_file is None.
        """
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.num_image_tokens = num_image_tokens
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<unused_0>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]
        self.max_blank_length = max_blank_length
        self.byte_fallback = byte_fallback
        self.text_tokenizer = TextTokenizer(vocab_file)

    def _get_text_tokenizer(self):
        """
        Method _get_text_tokenizer in class SPTokenizer.

        Args:
            self: SPTokenizer object. The instance of the SPTokenizer class.

        Returns:
            text_tokenizer: This method returns the text tokenizer associated with the SPTokenizer instance.

        Raises:
            None.
        """
        return self.text_tokenizer

    @staticmethod
    def get_blank_token(length: int):
        """
        This method generates a blank token based on the specified length.

        Args:
            length (int): The length of the blank token to be generated. Must be greater than or equal to 2.

        Returns:
            None.

        Raises:
            AssertionError: If the specified length is less than 2.
        """
        assert length >= 2
        return f"<|blank_{length}|>"

    @staticmethod
    def get_tab_token():
        """
        This method is a static method belonging to the 'SPTokenizer' class. It returns a tab token.

        Args:
            This method takes no parameters.

        Returns:
            None.

        Raises:
            None.
        """
        return "<|tab|>"

    @property
    def num_text_tokens(self):
        """
        Returns the number of text tokens in the SPTokenizer object.

        Args:
            self (SPTokenizer): The SPTokenizer object.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_tokenizer.num_tokens

    @property
    def num_tokens(self):
        """
        Method to calculate the total number of tokens in the SPTokenizer instance.

        Args:
            self (SPTokenizer): The instance of the SPTokenizer class.

        Returns:
            None:
                This method does not return a value directly but calculates the total number of tokens based on the
                sum of image tokens and text tokens.

        Raises:
            None.
        """
        return self.num_image_tokens + self.num_text_tokens

    @staticmethod
    def _encode_whitespaces(text: str, max_len: int = 80):
        """
        This method encodes whitespaces in the input text.

        Args:
            text (str): The input text to be encoded, containing whitespaces.
            max_len (int, optional): The maximum length of whitespaces to be encoded. Defaults to 80.

        Returns:
            None.

        Raises:
            None.
        """
        text = text.replace("\t", SPTokenizer.get_tab_token())
        for i in range(max_len, 1, -1):
            text = text.replace(" " * i, SPTokenizer.get_blank_token(i))
        return text

    def _preprocess(self, text: str, linebreak=True, whitespaces=True):
        """
        Preprocesses the given text by replacing linebreaks and encoding whitespaces.

        Args:
            self (SPTokenizer): An instance of the SPTokenizer class.
            text (str): The text to be preprocessed.
            linebreak (bool, optional): Determines whether linebreaks should be replaced. Defaults to True.
            whitespaces (bool, optional): Determines whether whitespaces should be encoded. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        if linebreak:
            text = text.replace("\n", "<n>")
        if whitespaces:
            text = self._encode_whitespaces(text, max_len=self.max_blank_length)
        return text

    def encode(
            self, text: str, linebreak=True, whitespaces=True, add_dummy_prefix=True
    ) -> List[int]:
        """
        Args:
            text: Text to encode.
            linebreak: Whether to encode newline (\n) in text.
            whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
            special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
            add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tmp = self._get_text_tokenizer().encode(text)
        tokens = [x + self.num_image_tokens for x in tmp]
        return tokens if add_dummy_prefix else tokens[2:]

    def postprocess(self, text):
        """
        postprocess method in SPTokenizer class.

        Args:
            self (SPTokenizer): The instance of SPTokenizer class.
            text (str): The input text to be post-processed, containing special tokens.

        Returns:
            None.

        Raises:
            None.
        """
        text = text.replace("<n>", "\n")
        text = text.replace(SPTokenizer.get_tab_token(), "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def decode(self, text_ids: List[int]) -> str:
        """
        Decode the given text ids using the SPTokenizer.

        Args:
            self: The instance of the SPTokenizer class.
            text_ids (List[int]): A list of integers representing the text ids to be decoded.

        Returns:
            str: The decoded text corresponding to the provided text ids.

        Raises:
            ValueError: If the input text_ids is empty.
            TypeError: If the input text_ids is not a list of integers.
        """
        ids = [int(_id) - self.num_image_tokens for _id in text_ids]
        ids = [_id for _id in ids if _id >= 0]
        text = self._get_text_tokenizer().decode(ids)
        text = self.postprocess(text)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        """Decode a list of tokens into a string.

        Args:
            self: An instance of the SPTokenizer class.
            tokens (List[str]): A list of tokens to be decoded into a string.

        Returns:
            str: The decoded string.

        Raises:
            None.

        This method takes a list of tokens and converts them into a string using the text tokenizer.
        The resulting string is then passed through the postprocessing method to remove any unwanted characters or
        formatting. The method returns the final decoded string as output.
        This method is a part of the SPTokenizer class and requires an instance of the class to be called.
        """
        text = self._get_text_tokenizer().convert_tokens_to_string(tokens)
        text = self.postprocess(text)
        return text

    def tokenize(
            self, text: str, linebreak=True, whitespaces=True, add_dummy_prefix=True
    ) -> List[str]:
        """
        Args:
            text: Text to encode.
            linebreak: Whether to encode newline (\n) in text.
            whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
            special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
            add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self._get_text_tokenizer().tokenize(text)
        return tokens if add_dummy_prefix else tokens[2:]

    def __getitem__(self, x: Union[int, str]):
        """
        This method is used to retrieve items from the SPTokenizer instance based on the provided key.

        Args:
            self (SPTokenizer): The instance of the SPTokenizer class.
            x (Union[int, str]): The key used to retrieve the item. It can be either an integer or a string.
                If x is an integer, it represents the index of the item to retrieve.
                If x is a string, it represents the token to retrieve.

        Returns:
            None: This method does not return any value directly.
                The retrieved item is indirectly obtained based on the key provided.

        Raises:
            ValueError: Raised when the key provided is neither a string nor an integer, indicating an invalid key type.
        """
        if isinstance(x, int):
            if x < self.num_image_tokens:
                return f"<image_{x}>"
            return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)
        if isinstance(x, str):
            if x.startswith("<image_") and x.endswith(">") and x[7:-1].isdigit():
                return int(x[7:-1])
            return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens
        raise ValueError("The key should be str or int.")


class ChatGLMTokenizer(PreTrainedTokenizer):
    """
    Construct a ChatGLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """
    vocab_files_names = {"vocab_file": "ice_text.model"}
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            remove_space=False,
            bos_token='<sop>',
            eos_token='<eop>',
            end_token='</s>',
            mask_token='[MASK]',
            gmask_token='[gMASK]',
            padding_side="left",
            pad_token="<pad>",
            unk_token="<unk>",
            num_image_tokens=20000,
            **kwargs
    ) -> None:
        """
        Initializes a ChatGLMTokenizer object.

        Args:
            vocab_file (str): The file path to the vocabulary file.
            do_lower_case (bool, optional): Flag indicating whether to convert all tokens to lowercase. Defaults to False.
            remove_space (bool, optional): Flag indicating whether to remove spaces from tokens. Defaults to False.
            bos_token (str, optional): The beginning of sentence token. Defaults to '<sop>'.
            eos_token (str, optional): The end of sentence token. Defaults to '<eop>'.
            end_token (str, optional): The end token. Defaults to '</s>'.
            mask_token (str, optional): The mask token. Defaults to '[MASK]'.
            gmask_token (str, optional): The global mask token. Defaults to '[gMASK]'.
            padding_side (str, optional): The side to pad tokens on. Defaults to 'left'.
            pad_token (str, optional): The padding token. Defaults to '<pad>'.
            unk_token (str, optional): The unknown token. Defaults to '<unk>'.
            num_image_tokens (int, optional): The number of image tokens. Defaults to 20000.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            None: This method does not raise any exceptions.
        """
        self.sp_tokenizer = SPTokenizer(vocab_file, num_image_tokens=num_image_tokens)
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            padding_side=padding_side,
            bos_token=bos_token,
            eos_token=eos_token,
            end_token=end_token,
            mask_token=mask_token,
            gmask_token=gmask_token,
            pad_token=pad_token,
            unk_token=unk_token,
            num_image_tokens=num_image_tokens,
            **kwargs
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.vocab_file = vocab_file

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.end_token = end_token
        self.mask_token = mask_token
        self.gmask_token = gmask_token
        """ Initialisation """

    @property
    def gmask_token_id(self) -> Optional[int]:
        """
        This method returns the token ID of the gmask token in the ChatGLMTokenizer.

        Args:
            self (ChatGLMTokenizer): The instance of the ChatGLMTokenizer class.

        Returns:
            Optional[int]: Returns the token ID of the gmask token if it exists, otherwise returns None.

        Raises:
            None
        """
        if self.gmask_token is None:
            return None
        return self.convert_tokens_to_ids(self.gmask_token)

    @property
    def end_token_id(self) -> Optional[int]:
        """
        Returns:
            `Optional[int]`:
                Id of the end of context token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self.end_token is None:
            return None
        return self.convert_tokens_to_ids(self.end_token)

    @property
    def vocab_size(self):
        """ Returns vocab size """
        return self.sp_tokenizer.num_tokens

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def preprocess_text(self, inputs):
        """
        preprocess_text method in the ChatGLMTokenizer class preprocesses the input text based on the specified configuration.

        Args:
            self (ChatGLMTokenizer): The instance of the ChatGLMTokenizer class.
            inputs (str): The input text to be preprocessed.

        Returns:
            str:
                The preprocessed text based on the specified configuration.

                - If self.remove_space is True, leading and trailing spaces are removed,
                and consecutive spaces within the text are replaced with a  single space.
                - If self.do_lower_case is True, the text is converted to lowercase. The preprocessed text is returned.

        Raises:
            None
        """
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs

        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, **kwargs):
        """ Returns a tokenized string. """
        text = self.preprocess_text(text)

        seq = self.sp_tokenizer.tokenize(text)

        return seq

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens into a single string representation.

        Args:
            self (ChatGLMTokenizer): An instance of the ChatGLMTokenizer class.
            tokens (List[str]): A list of tokens to be converted into a string representation.

        Returns:
            str: The string representation of the given list of tokens.

        Raises:
            None.

        Note:
            - The tokens should be generated using the sp_tokenizer of the ChatGLMTokenizer instance.
            - The resulting string may contain whitespace and punctuation marks based on the original tokenization.

        Example:
            ```python
            >>> tokenizer = ChatGLMTokenizer()
            >>> tokens = ['Hello', ',', 'how', 'are', 'you', '?']
            >>> string_representation = tokenizer.convert_tokens_to_string(tokens)
            ```
        """
        return self.sp_tokenizer.decode_tokens(tokens)

    def _decode(
            self,
            token_ids: Union[int, List[int]],
            **kwargs
    ) -> str:
        """
        This method decodes the given token IDs into a string representation.

        Args:
            self (ChatGLMTokenizer): The instance of the ChatGLMTokenizer class.
            token_ids (Union[int, List[int]]): The token IDs to be decoded. It can be a single integer or a list of integers.

        Returns:
            str: The decoded string representation of the token IDs.

        Raises:
            None.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if len(token_ids) == 0:
            return ""
        if self.pad_token_id in token_ids:  # remove pad
            token_ids = list(filter((self.pad_token_id).__ne__, token_ids))
        return super()._decode(token_ids, **kwargs)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.sp_tokenizer[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_tokenizer[index]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        gmask_id = self.sp_tokenizer[self.gmask_token]
        eos_id = self.sp_tokenizer[self.eos_token]
        token_ids_0 = token_ids_0 + [gmask_id, self.sp_tokenizer[self.bos_token]]
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [eos_id]
        return token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                - The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        bos_token_id = self.sp_tokenizer[self.bos_token]
        mask_token_id = self.sp_tokenizer[self.mask_token]
        gmask_token_id = self.sp_tokenizer[self.gmask_token]
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if max_length is not None:
            if "attention_mask" not in encoded_inputs:
                if bos_token_id in required_input:
                    context_length = required_input.index(bos_token_id)
                else:
                    context_length = seq_length
                attention_mask = np.ones((1, seq_length, seq_length))
                attention_mask = np.tril(attention_mask)
                attention_mask[:, :, :context_length] = 1
                attention_mask = np.bool_(attention_mask < 0.5)
                encoded_inputs["attention_mask"] = attention_mask

            if "position_ids" not in encoded_inputs:
                if bos_token_id in required_input:
                    context_length = required_input.index(bos_token_id)
                else:
                    context_length = seq_length
                position_ids = np.arange(seq_length, dtype=np.int64)
                mask_token = mask_token_id if mask_token_id in required_input else gmask_token_id
                if mask_token in required_input:
                    mask_position = required_input.index(mask_token)
                    position_ids[context_length:] = mask_position
                block_position_ids = np.concatenate(
                    [np.zeros(context_length, dtype=np.int64),
                     np.arange(1, seq_length - context_length + 1, dtype=np.int64)])
                encoded_inputs["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = np.pad(encoded_inputs["attention_mask"],
                                                          pad_width=[(0, 0), (difference, 0), (difference, 0)],
                                                          mode='constant', constant_values=True)
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                    "token_type_ids"
                ]
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = np.pad(encoded_inputs["position_ids"],
                                                        pad_width=[(0, 0), (difference, 0)])
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs

__all__ = ['ChatGLMTokenizer']
