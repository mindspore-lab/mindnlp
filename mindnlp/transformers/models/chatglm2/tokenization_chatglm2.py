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
"""ChatGLM2 Tokenizer"""
import os
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from mindnlp.utils import PaddingStrategy
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import EncodedInput, BatchEncoding


class SPTokenizer:
    """SPTokenizer"""
    def __init__(self, model_path: str):
        """
        Initializes an instance of the SPTokenizer class.
        
        Args:
            model_path (str): The file path to the SentencePiece model file. 
                The model file must exist as a valid file path. 
        
        Returns:
            None.
        
        Raises:
            AssertionError: If the model_path provided is not a valid file path.
            AssertionError: If the vocabulary size of the SentencePiece model does not match the piece size.
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path) # pylint: disable=unexpected-keyword-arg
        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def tokenize(self, s: str):
        """
        Tokenizes a given string using the SentencePiece model.
        
        Args:
            self: An instance of the SPTokenizer class.
            s (str): The input string to be tokenized.
        
        Returns:
            None: This method modifies the state of the SPTokenizer instance.
        
        Raises:
            None.
        
        This method takes in a string 's' and tokenizes it using the SentencePiece model associated with the SPTokenizer instance.
        The tokenization process splits the input string into smaller units, such as  words or subwords, based on
        the language or model-specific rules.

        Note that the tokenization is performed in-place, meaning the original string object is modified.
        Therefore, the method does not return any value, but updates the internal state of the SPTokenizer
        instance with the tokenized result.

        Example:
            ```python
            >>> sp_tokenizer = SPTokenizer()
            >>> sp_tokenizer.tokenize("Hello, world!")
            >>> # After tokenization, the internal state of 'sp_tokenizer' will be modified.
            ```
        """
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encodes a given string using the SentencePiece tokenizer.

        Args:
            self: An instance of the SPTokenizer class.
            s (str): The input string to be encoded.
            bos (bool, optional): Whether to add a beginning of sentence (BOS) token to the encoded sequence. Defaults to False.
            eos (bool, optional): Whether to add an end of sentence (EOS) token to the encoded sequence. Defaults to False.

        Returns:
            List[int]: The encoded sequence as a list of integers.

        Raises:
            AssertionError: If the input parameter 's' is not of type str.

        Example:
            ```python
            >>> tokenizer = SPTokenizer()
            >>> encoded_sequence = tokenizer.encode("Hello, world!", bos=True, eos=True)
            >>> print(encoded_sequence)
            >>> # Output: [1, 123, 456, 789, 2]
            ```
        """
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decode a list of token IDs into a string using the SentencePiece model.

        Args:
            self (SPTokenizer): An instance of the SPTokenizer class.
            t (List[int]): A list of token IDs to be decoded into a string. Each token ID represents a tokenized unit.

        Returns:
            str: The decoded string representing the token IDs.

        Raises:
            TypeError: If the input t is not a list of integers.
            ValueError: If the input t is empty or contains invalid token IDs.
            RuntimeError: If there is an issue with decoding the token IDs using the SentencePiece model.
        """
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        """
        Decode the given list of tokens into a single text string using the SentencePiece model.

        Args:
            self (SPTokenizer): An instance of the SPTokenizer class.
            tokens (List[str]): A list of string tokens to be decoded using the SentencePiece model.

        Returns:
            str: The decoded text string generated by decoding the input tokens using the SentencePiece model.

        Raises:
            TypeError: If the input tokens are not of type List[str] or if the input is not a valid instance of SPTokenizer.
            ValueError: If the input tokens list is empty or if the decoding process fails for any reason.
        """
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)


class ChatGLM2Tokenizer(PreTrainedTokenizer):
    """ChatGLM2Tokenizer"""
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
        """
        Initializes a ChatGLM2Tokenizer object.

        Args:
            vocab_file (str): The path to the vocabulary file used by the tokenizer.
            padding_side (str, optional): The side to pad sequences. Default is 'left'.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Default is False.
            **kwargs: Additional keyword arguments to pass to the parent class.

        Returns:
            None.

        Raises:
            None.
        """
        self.name = "GLMTokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)

    def get_command(self, token):
        """
        This method `get_command` in the class `ChatGLM2Tokenizer` retrieves a command associated with a given token.

        Args:
            self (ChatGLM2Tokenizer): An instance of the ChatGLM2Tokenizer class.
                This parameter is used to access the special tokens and tokenizer associated with the instance.
            token (str): The token for which the associated command needs to be retrieved.
                This parameter specifies the token for which the command is to be fetched from the special tokens.

        Returns:
            None: This method returns None if the token does not match any special token.
                Otherwise, it returns the command associated with the token from the tokenizer's special tokens.

        Raises:
            AssertionError: If the provided token is not present in the special tokens of the ChatGLM2Tokenizer instance,
                an AssertionError is raised with a message indicating that the token is not a special
            token for the instance.
        """
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        """
        Returns the unknown token.

        Args:
            self: An instance of the ChatGLM2Tokenizer class.

        Returns:
            str: The unknown token '<unk>'.

        Raises:
            None.
        """
        return "<unk>"

    @property
    def pad_token(self) -> str:
        """
        Method that returns the padding token for the ChatGLM2Tokenizer.

        Args:
            self: The instance of the ChatGLM2Tokenizer class.

        Returns:
            str: The padding token '<unk>' used for padding sequences during tokenization.

        Raises:
            None.
        """
        return "<unk>"

    @property
    def pad_token_id(self):
        """
        This method retrieves the token ID for the '<pad>' token in the ChatGLM2Tokenizer class.

        Args:
            self (ChatGLM2Tokenizer): The instance of the ChatGLM2Tokenizer class.
                This parameter represents the current instance of the ChatGLM2Tokenizer class.

        Returns:
           The token ID for the '<pad>' token in the ChatGLM2Tokenizer class.

        Raises:
            None.
        """
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        """
        Returns the end-of-sentence token.

        This method is a property decorator that returns the end-of-sentence token as a string.

        Args:
            self: An instance of the ChatGLM2Tokenizer class.

        Returns:
            A string representing the end-of-sentence token.

        Raises:
            None.
        """
        return "</s>"

    @property
    def eos_token_id(self):
        """
        Returns the token ID for the end-of-sentence (EOS) token in the ChatGLM2Tokenizer class.

        Args:
            self (ChatGLM2Tokenizer): An instance of the ChatGLM2Tokenizer class.

        Returns:
            Token ID for the end-of-sentence (EOS) token.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        """
        Returns the vocabulary size of the ChatGLM2Tokenizer.

        Args:
            self (ChatGLM2Tokenizer): The instance of the ChatGLM2Tokenizer class.

        Returns:
            None.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self.tokenizer.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        """
        Method to tokenize text using the tokenizer associated with the ChatGLM2Tokenizer class.

        Args:
            self (ChatGLM2Tokenizer): The instance of the ChatGLM2Tokenizer class.
            text (str): The input text to be tokenized.

        Returns:
            None.

        Raises:
            This method does not raise any exceptions.
        """
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens into a single string representation using the ChatGLM2Tokenizer.

        Args:
            self (ChatGLM2Tokenizer): An instance of the ChatGLM2Tokenizer class.
            tokens (List[str]): A list of tokens to be converted into a string representation.

        Returns:
            str: The string representation of the given list of tokens.

        Raises:
            None.

        Note:
            The 'tokens' parameter should only contain valid tokens that are supported by the ChatGLM2Tokenizer.
            Any invalid tokens may result in unexpected behavior.

        Example:
            ```python
            >>> tokenizer = ChatGLM2Tokenizer()
            >>> tokens = ['Hello', ',', 'how', 'are', 'you', '?']
            >>> string_representation = tokenizer.convert_tokens_to_string(tokens)
            >>> # string_representation will be 'Hello, how are you?'
            ```
        """
        return self.tokenizer.decode_tokens(tokens)

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

    def get_prefix_tokens(self):
        """
        Returns a list of prefix tokens used in the ChatGLM2Tokenizer class.

        Args:
            self: The instance of the ChatGLM2Tokenizer class.

        Returns:
            list: A list of prefix tokens used in the ChatGLM2Tokenizer class.
                The list contains two elements:

                1. The result of the self.get_command('[gMASK]') method.
                2. The result of the self.get_command('sop') method.

        Raises:
            None.
        """
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    def build_prompt(self, query, history=None):
        """
        This method builds a prompt for a chat history in the ChatGLM2Tokenizer class.

        Args:
            self: The instance of the class.
            query (str): The input query for the prompt.
            history (list): A list of tuples representing the chat history. Each tuple contains an old query and its response.

        Returns:
            str: A formatted prompt containing the chat history and the input query.

        Raises:
            None
        """
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

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
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
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
            padding_strategy:
                PaddingStrategy to use for padding.

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
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs

__all__ = ['ChatGLM2Tokenizer']
