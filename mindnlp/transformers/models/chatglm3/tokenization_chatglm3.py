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
"""ChatGLM3 Tokenizer"""
import os
import re
import json
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from mindnlp.utils import PaddingStrategy, logging
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import EncodedInput, BatchEncoding


logger = logging.get_logger(__name__)


class SPTokenizer:

    """
    The `SPTokenizer` class represents a SentencePiece Tokenizer for natural language processing tasks.
    It provides methods for tokenizing, encoding, and decoding text using a pre-trained model.
    
    Attributes:
        `sp_model` (SentencePieceProcessor): The SentencePieceProcessor object loaded with the provided model file.
        `n_words` (int): The total number of words in the tokenizer's vocabulary, including special tokens.
        `bos_id` (int): The ID of the beginning of sentence (BOS) token in the vocabulary.
        `eos_id` (int): The ID of the end of sentence (EOS) token in the vocabulary.
        `pad_id` (int): The ID of the padding token in the vocabulary.
        `special_tokens` (dict): A dictionary mapping special tokens to their corresponding IDs.
        `index_special_tokens` (dict): A dictionary mapping special token IDs to their corresponding tokens.
        `role_special_token_expression` (str): A regular expression pattern that matches role special tokens.

    Methods:
        `__init__`: Initializes the SPTokenizer object by loading the provided model file and setting up special tokens.
        `tokenize`: Tokenizes the input text using the SentencePieceProcessor object.
        `encode`: Encodes the input text into a sequence of token IDs.
        `decode`: Decodes a sequence of token IDs into the corresponding text.
        `decode_tokens`: Decodes a list of tokens into the corresponding text.
        `convert_token_to_id`: Converts a token (string) to its corresponding ID using the vocabulary.
        `convert_id_to_token`: Converts an ID (integer) to its corresponding token using the vocabulary.

    Note:
        The `SPTokenizer` class extends a base class, but the name of the base class is not provided in the code snippet.
    """
    def __init__(self, model_path: str):
        """
        Initializes an instance of the SPTokenizer class.

        Args:
            self: The instance of the SPTokenizer class.
            model_path (str): The path to the model file. It must be a valid file path.
                This file is used to initialize the SentencePieceProcessor object.

        Returns:
            None

        Raises:
            AssertionError: If the model_path is not a valid file path.
            AssertionError: If the vocabulary size of the SentencePieceProcessor object does not match the piece size.
            ValueError: If the model file is not found at the specified model_path.
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

        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
        self.role_special_token_expression = "|".join([re.escape(token) for token in special_tokens]) # for apply_chat_template

    def tokenize(self, s: str, encode_special_tokens=False):
        """
        This method tokenizes the input string 's' using the SentencePiece model 'sp_model' in the SPTokenizer class.

        Args:
            self: The instance of the SPTokenizer class.
            s (str): The input string to be tokenized.
            encode_special_tokens (bool): A flag indicating whether special tokens should be encoded.
                If set to True, the method will encode special tokens; otherwise, it will not. Defaults to False.

        Returns:
            list:
                A list of tokenized pieces generated by the SentencePiece model.

                - If encode_special_tokens is False, the method returns the tokenized pieces of the input string 's'.
                - If encode_special_tokens is True, the method returns the tokenized pieces with special tokens encoded.

        Raises:
            ValueError: If the input string 's' is empty or None.
            TypeError: If the input string 's' is not of type str.
            Exception: If the SentencePiece model 'sp_model' fails to tokenize the input string or encounters an error during tokenization.
        """
        if encode_special_tokens:
            last_index = 0
            t = []
            for match in re.finditer(self.role_special_token_expression, s):
                if last_index < match.start():
                    t.extend(self.sp_model.EncodeAsPieces(s[last_index:match.start()]))
                t.append(s[match.start():match.end()])
                last_index = match.end()
            if last_index < len(s):
                t.extend(self.sp_model.EncodeAsPieces(s[last_index:]))
            return t
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        This method encodes a given string using the SentencePiece model.

        Args:
            self: The instance of the SPTokenizer class.
            s (str): The input string to be encoded.
            bos (bool, optional): Boolean flag indicating whether to prepend the beginning of sentence token.
                Defaults to False.
            eos (bool, optional): Boolean flag indicating whether to append the end of sentence token. Defaults to False.

        Returns:
            List[int]: A list of integers representing the encoded tokens.

        Raises:
            AssertionError: If the input 's' is not a string.
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
        The 'decode' method decodes a list of integers into a string using the sp_model.

        Args:
            self: The instance of the SPTokenizer class.
            t (List[int]): A list of integers representing tokens to be decoded into a string.

        Returns:
            str: The decoded string.

        Raises:
            None.
        """
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.decode(buffer)
        return text

    def decode_tokens(self, tokens: List[str]) -> str:
        """
        Decode the given list of tokens into text using the SentencePiece model.

        Args:
            self (SPTokenizer): The instance of the SPTokenizer class.
            tokens (List[str]): A list of tokens to be decoded into text using the SentencePiece model.

        Returns:
            str: The decoded text generated from the input tokens.

        Raises:
            ValueError: If the input tokens list is empty or contains invalid token values.
            TypeError: If the input tokens are not in the expected format (list of strings).
            RuntimeError: If there is an issue decoding the tokens using the SentencePiece model.
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
        if index in self.index_special_tokens:
            return self.index_special_tokens[index]
        if index in [self.eos_id, self.bos_id, self.pad_id] or index < 0 or index > self.sp_model.vocab_size():
            return ""
        return self.sp_model.IdToPiece(index)


class ChatGLM3Tokenizer(PreTrainedTokenizer):

    """
    The 'ChatGLM3Tokenizer' class represents a tokenizer for a chat model that inherits from PreTrainedTokenizer.
    It provides methods for tokenizing, converting tokens to IDs, converting IDs to tokens, building model inputs
    with special tokens, padding, and saving the vocabulary.
    The class also provides methods for constructing chat inputs, building single messages, and extracting prefix tokens.
    Additionally, it offers properties for accessing special tokens and their IDs, as well as the vocabulary size.
    Furthermore, it provides a method for converting tokens to a string.

    Attributes:
        name (str): Name of the tokenizer.
        vocab_file (str): Path to the vocabulary file.
        tokenizer (SPTokenizer): Instance of the SPTokenizer for tokenization.
        special_tokens (dict): Dictionary of special tokens and their corresponding IDs.
        encode_special_tokens (bool): Flag indicating whether to encode special tokens.

    Properties:

    - unk_token (str): Property for accessing the unknown token.
    - pad_token (str): Property for accessing the padding token.
    - eos_token (str): Property for accessing the end-of-sequence token.
    - unk_token_id (int): Property for accessing the ID of the unknown token.
    - pad_token_id (int): Property for accessing the ID of the padding token.
    - eos_token_id (int): Property for accessing the ID of the end-of-sequence token.
    - vocab_size (int): Property for accessing the size of the vocabulary.

    Methods:
        get_command(token): Retrieves the ID of a given token.
        get_vocab(): Returns the vocabulary as a dictionary.
        _tokenize(text, **kwargs): Tokenizes the input text.
        _convert_token_to_id(token): Converts a token to its corresponding ID.
        _convert_id_to_token(index): Converts an ID to its corresponding token.
        convert_tokens_to_string(tokens): Converts a list of tokens to a string.
        save_vocabulary(save_directory, filename_prefix=None): Saves the vocabulary to a directory.
        get_prefix_tokens(): Retrieves prefix tokens.
        build_single_message(role, metadata, message): Constructs a single message with role, metadata, and message.
        build_chat_input(query, history=None, role='user'): Constructs chat input from a query and history.
        build_inputs_with_special_tokens(token_ids_0, token_ids_1=None): Builds model inputs with special tokens.
        _pad(encoded_inputs, max_length=None, padding_strategy=PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of=None, return_attention_mask=None):
            Pads encoded inputs according to specified parameters.

    The 'ChatGLM3Tokenizer' class provides a comprehensive set of methods for tokenization and model input construction,
    making it suitable for use in chat model applications.
    """
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
        self,
        vocab_file,
        padding_side="left",
        clean_up_tokenization_spaces=False,
        encode_special_tokens=False,
        **kwargs
    ):
        """
        Initialize a ChatGLM3Tokenizer object.

        Args:
            vocab_file (str): The path to the vocabulary file.
            padding_side (str, optional): Specifies whether padding should be added to the 'left' or 'right'
                side of the input sequences. Default is 'left'.
            clean_up_tokenization_spaces (bool, optional): If True, clean up tokenization spaces. Default is False.
            encode_special_tokens (bool, optional): If True, special tokens will be encoded. Default is False.
            **kwargs: Additional keyword arguments.

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
            "<unk>": self.tokenizer.pad_id,
            "<pad>": self.tokenizer.pad_id
        }
        self.encode_special_tokens = encode_special_tokens

        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    def get_command(self, token):
        """
        Retrieves the command associated with a given token.

        Args:
            self (ChatGLM3Tokenizer): An instance of the ChatGLM3Tokenizer class.
            token (str): The token for which the command needs to be retrieved.

        Returns:
            None: This method returns None.

        Raises:
            AssertionError: If the token is not a special token for the ChatGLM3Tokenizer instance.

        Note:
            This method checks if the given token is one of the special tokens stored in the self.special_tokens dictionary.
            If it is, the corresponding command is returned. Otherwise, an assertion error is raised if the token is not a
            special token for the ChatGLM3Tokenizer instance.
        """
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        """
        This method 'unk_token' in the class 'ChatGLM3Tokenizer' retrieves the unknown token from the tokenizer.

        Args:
            self: An instance of the ChatGLM3Tokenizer class.

        Returns:
            str: The unknown token retrieved from the tokenizer.

        Raises:
            No specific exceptions are raised within this method.
        """
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<unk>"))

    @property
    def pad_token(self) -> str:
        """
        This method returns the string representation of the padding token used in the ChatGLM3Tokenizer.

        Args:
            self: The instance of the ChatGLM3Tokenizer class.

        Returns:
            str: The string representation of the padding token.

        Raises:
            None
        """
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<pad>"))

    @property
    def eos_token(self) -> str:
        """
        Returns the end-of-sentence token as a string.

        Args:
            self: An instance of the ChatGLM3Tokenizer class.

        Returns:
            A string representing the end-of-sentence token.

        Raises:
            None.
        """
        return self.tokenizer.sp_model.IdToPiece(self.get_command("<eos>"))

    @property
    def unk_token_id(self) -> int:
        """
        This method returns the token ID corresponding to the '<unk>' token in the ChatGLM3Tokenizer class.

        Args:
            self: A reference to the instance of the ChatGLM3Tokenizer class.

        Returns:
            int: An integer representing the token ID of the '<unk>' token in the tokenizer.

        Raises:
            This method does not explicitly raise any exceptions.
        """
        return self.get_command("<unk>")

    @property
    def pad_token_id(self) -> int:
        """
        This method returns the token ID for the padding token within the ChatGLM3Tokenizer class.

        Args:
            self: An instance of the ChatGLM3Tokenizer class.

        Returns:
            int: The token ID corresponding to the '<pad>' token.

        Raises:
            - None
        """
        return self.get_command("<pad>")

    @property
    def eos_token_id(self):
        """
        Returns the ID of the end-of-sentence (EOS) token in the ChatGLM3Tokenizer class.

        Args:
            self (ChatGLM3Tokenizer): An instance of the ChatGLM3Tokenizer class.

        Returns:
            None.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self.get_command("<eos>")

    @unk_token.setter
    def unk_token(self, value):
        """
        Method 'unk_token' in the class 'ChatGLM3Tokenizer'.

        Args:
            self (object):
                Reference to the instance of ChatGLM3Tokenizer.

                - Purpose: Represents the current object instance.
                - Restrictions: Must be an instance of ChatGLM3Tokenizer.

            value (any):
                The new value to set for the unk_token attribute.

                - Purpose: Specifies the value to set for the unk_token attribute.
                - Restrictions: None.

        Returns:
            None:
                - Purpose: There is no return value from this method.

        Raises:
            None:
                No exceptions are raised explicitly within this method.
        """
        logger.warning("Setting unk_token is not supported, use the default one.")

    @pad_token.setter
    def pad_token(self, value):
        """Set the pad_token value for the ChatGLM3Tokenizer.

        This method sets the pad_token value for the ChatGLM3Tokenizer object.
        The pad_token value is used during tokenization to represent padding tokens.
        If this method is called, a warning message will be logged indicating that setting pad_token is not supported
        and the default pad_token value will be used instead.

        Args:
            self (ChatGLM3Tokenizer): The ChatGLM3Tokenizer object.
            value (Any): The value to set as the pad_token.

        Returns:
            None.

        Raises:
            None: This method does not raise any exceptions.
        """
        logger.warning("Setting pad_token is not supported, use the default one.")

    @eos_token.setter
    def eos_token(self, value):
        """
        Method to set the end-of-sequence token for the ChatGLM3Tokenizer class.

        Args:
            self (ChatGLM3Tokenizer): The instance of the ChatGLM3Tokenizer class.
            value (Any): The value to be set as the end-of-sequence token.
                This parameter is not used for setting the end-of-sequence token, as it is a read-only property.

        Returns:
            None.

        Raises:
            None.
        """
        logger.warning("Setting eos_token is not supported, use the default one.")

    @property
    def vocab_size(self):
        """
        This method retrieves the vocabulary size from the ChatGLM3Tokenizer instance.

        Args:
            self (ChatGLM3Tokenizer): The instance of the ChatGLM3Tokenizer class.
                It represents the tokenizer used for processing the text data.

        Returns:
            int: The vocabulary size of the tokenizer.
                It indicates the total number of unique words present in the tokenizer's vocabulary.

        Raises:
            None
        """
        return self.tokenizer.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        """
        This method tokenizes the input text using the specified tokenizer.

        Args:
            text (str): The input text to be tokenized.
            **kwargs: Additional keyword arguments to be passed to the tokenizer.

        Returns:
            None.

        Raises:
            Any exceptions raised by the underlying 'tokenizer.tokenize' method may be propagated.
        """
        return self.tokenizer.tokenize(text, encode_special_tokens=self.encode_special_tokens)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens into a string representation using the ChatGLM3Tokenizer.

        Args:
            self (ChatGLM3Tokenizer): An instance of the ChatGLM3Tokenizer class.
            tokens (List[str]): A list of tokens to be converted into a string.

        Returns:
            str: The string representation of the tokens.

        Raises:
            None.

        This method takes in an instance of the ChatGLM3Tokenizer class and a list of tokens as input.
        It then uses the tokenizer's 'decode_tokens' method to convert the tokens into a string representation.
        The resulting string is returned as the output.

        The 'self' parameter is a reference to the current instance of the ChatGLM3Tokenizer class.
        It is used to access the tokenizer object and its methods.

        The 'tokens' parameter is a list of strings representing the tokens to be converted into a string.
        The tokens should be in the same order as they were generated by the tokenizer.

        The return value is a string representation of the tokens.
        This can be useful for displaying or manipulating the tokens in a human-readable format.

        This method does not raise any exceptions.
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
        This method 'get_prefix_tokens' is defined within the 'ChatGLM3Tokenizer' class and retrieves a list of prefix tokens.

        Args:
            self: A reference to the instance of the class. It is used to access the instance variables and methods of the class.

        Returns:
            Returns a list of prefix tokens which is a combination of the commands '[gMASK]' and 'sop'.

        Raises:
            This method does not raise any exceptions.
        """
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    def build_single_message(self, role, metadata, message):
        """
        Builds a single message token for the ChatGLM3Tokenizer.

        Args:
            self (ChatGLM3Tokenizer): The instance of the ChatGLM3Tokenizer class.
            role (str): The role of the message sender. It should be one of ['system', 'user', 'assistant', 'observation'].
            metadata (str): The metadata associated with the message.
            message (str): The actual message content.

        Returns:
            list: A list of tokens representing the single message built from the role, metadata, and message.

        Raises:
            AssertionError: If the 'role' parameter is not one of ['system', 'user', 'assistant', 'observation'].
        """
        assert role in ["system", "user", "assistant", "observation"], role
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        tokens = role_tokens + message_tokens
        return tokens

    def build_chat_input(self, query, history=None, role="user"):
        """
        This method builds input for a chat conversation in the ChatGLM3Tokenizer class.

        Args:
            self: The instance of the ChatGLM3Tokenizer class.
            query (str): The user's input for the chat conversation.
            history (list): A list of dictionaries representing the chat history.
                Each dictionary should have the keys 'role' (str), 'metadata' (str), and 'content' (str).

                - The 'role' key specifies the role of the participant in the conversation (either 'user' or 'system').
                - The 'metadata' key contains optional metadata for the message.
                - The 'content' key contains the actual text content of the message.
            role (str): The role of the participant for the current input. It can be either 'user' or 'system'.

        Returns:
            None: This method builds the input for the chat conversation and does not return any value.

        Raises:
            TypeError: If the input_ids are not of the expected type.
            ValueError: If the return_tensors parameter is not set to 'ms'.
            KeyError: If the role provided is not valid (i.e., not 'user' or 'system').
            JSONDecodeError: If there is an error in decoding the JSON content of the message.
            AttributeError: If the 'tools' key is missing in the history item when the role is 'system'.
        """
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.get_command("<|assistant|>")])
        return self.batch_encode_plus([input_ids], return_tensors="ms", is_split_into_words=True)

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

__all__ = ['ChatGLM3Tokenizer']
