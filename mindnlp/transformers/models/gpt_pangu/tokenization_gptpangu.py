# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
PanGu_Alpha Tokenizer.
"""
import mindspore
import sentencepiece
import numpy as np

from mindnlp.utils.import_utils import is_jieba_available
from mindnlp.transformers.tokenization_utils import PreTrainedTokenizer

if is_jieba_available():
    import jieba
    jieba.add_word('<s>')
    jieba.add_word('</s>')
    jieba.add_word('<eot>')
    jieba.add_word('<unk>')
    jieba.add_word('<sep>')
    jieba.add_word('<pad>')


class GPTPanguTokenizer(PreTrainedTokenizer):

    """
    This class represents a tokenizer for the GPTPangu model, which is used for tokenizing Chinese text.
    It inherits from the PreTrainedTokenizer class.
    
    Attributes:
        sp (sentencepiece.SentencePieceProcessor): An instance of the SentencePieceProcessor class used for tokenization.
        translator (dict): A translation dictionary to replace spaces and newlines with special tokens.
        
    Properties:
        vocab_size (int): Returns the size of the vocabulary used by the tokenizer.
    
    Methods:
        __init__:
            Initializes the GPTPanguTokenizer object.
            
        get_vocab:
            Returns the vocabulary as a dictionary.
            
        build_inputs_with_special_tokens:
            Builds model inputs by adding special tokens to a sequence or a pair of sequences
            for sequence classification tasks.
            
        tokenize:
            Tokenizes a string.
            
        convert_tokens_to_ids:
            Converts a list of tokens to their corresponding IDs.
            
        convert_ids_to_tokens:
            Converts a list of IDs to their corresponding tokens.
            
        decode:
            Decodes a list of IDs into text.
    """
    # Ref: https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenization_jieba.py
    vocab_files_names = {
        "model_file": "vocab.model"
    }

    def __init__(
            self,
            model_file,
            **kwargs
    ):
        """
        Initializes a new instance of the GPTPanguTokenizer class.
        
        Args:
            self: An instance of the GPTPanguTokenizer class.
            model_file (str): The path to the model file used by the tokenizer.
                The model file should be in the format expected by the sentencepiece.SentencePieceProcessor.
                The tokenizer will load the model file during initialization.
        
        Returns:
            None.
        
        Raises:
            None.
        
        """
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        super().__init__(**kwargs)
        # special token ids
        # self.eos_token_id = self.sp.piece_to_id("<eot>")

    @property
    def vocab_size(self):
        """ Returns vocab size """
        return self.sp.vocab_size()

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
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
        if self.bos_token_id is not None:
            if token_ids_1 is None:
                return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            bos = [self.bos_token_id]
            sep = [self.sep_token_id]
            eos = [self.eos_token_id]
            return bos + token_ids_0 + sep + token_ids_1 + eos

        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        sep = [self.sep_token_id]
        eos = [self.eos_token_id]
        return token_ids_0 + sep + token_ids_1 + eos

    def tokenize(self, text, **kwargs):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        return seg_list

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a list of tokens into their corresponding token IDs using the GPTPanguTokenizer.

        Args:
            self (GPTPanguTokenizer): An instance of the GPTPanguTokenizer class.
            tokens (str or list): The tokens to be converted into token IDs.
                If a string is provided, it will be treated as a single token.

        Returns:
            list or None: A list of token IDs corresponding to the input tokens.
                Returns None if the input tokens are None.

        Raises:
            None

        Note:
            - If the input tokens are None, the method returns None.
            - If the input tokens are a string, the method calls the _convert_token_to_id_with_added_voc() method to
            convert it into a token ID.
            - If the input tokens contain special tokens, the method identifies their indices and splits the tokens
            into segments. Each segment is then encoded using the sp.encode() method and appended to the list of token
            IDs.
            - The method concatenates all the encoded segments and returns the final list of token IDs.

        Example:
            ```python
            >>> tokenizer = GPTPanguTokenizer()
            >>> tokens = ['Hello', 'world', '!']
            >>> ids = tokenizer.convert_tokens_to_ids(tokens)
            >>> # ids = [123, 456, 789]
            ```
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        special_tokens_index = [i for i, token in enumerate(tokens) if token in self.all_special_tokens]

        ids = []
        i = 0
        for j in special_tokens_index:
            new_seg = " ".join(tokens[i:j])
            ids.extend(self.sp.encode(new_seg))
            ids.append(self._convert_token_to_id(tokens[j]))
            i = j + 1

        new_seg = " ".join(tokens[i:])
        ids.extend(self.sp.encode(new_seg))

        return ids

        # new_seg = " ".join(tokens)
        # return self.sp.encode(new_seg)
        # # return tokens

    def _convert_token_to_id(self, token):
        """
        Converts a token to its corresponding ID using the GPTPanguTokenizer.

        Args:
            self (GPTPanguTokenizer): An instance of the GPTPanguTokenizer class.
            token (str): The token to be converted to its corresponding ID.

        Returns:
            None: This method does not return any value but performs the conversion operation internally.

        Raises:
            TypeError: If the token provided is not a string.
            ValueError: If the token does not exist in the tokenizer's vocabulary.
        """
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """
        Converts an index to its corresponding token using the GPTPanguTokenizer.

        Args:
            self (GPTPanguTokenizer): An instance of the GPTPanguTokenizer class.
            index (int): The index value to be converted to a token. It should be a non-negative integer.

        Returns:
            None

        Raises:
            None
        """
        return self.sp.id_to_piece(index)

    def convert_ids_to_tokens(self, ids):
        """
        Converts a list of token IDs to their corresponding tokens using the GPTPanguTokenizer.

        Args:
            self (GPTPanguTokenizer): An instance of the GPTPanguTokenizer class.
            ids (List[int]): A list of token IDs to be converted to tokens. Each ID represents a unique token.

        Returns:
            None

        Raises:
            None

        Note:
            The GPTPanguTokenizer must be initialized with a pretrained model before using this method.

        Example:
            ```python
            >>> tokenizer = GPTPanguTokenizer()
            >>> token_ids = [0, 1, 2]
            >>> tokenizer.convert_ids_to_tokens(token_ids)
            ['<s>', 'Hello', '</s>']
            ```
        """
        return self.decode(ids)

    def decode(self, ids, **kwargs):
        """
        Decode the given token IDs into text using the GPTPanguTokenizer.

        Args:
            self (GPTPanguTokenizer): An instance of the GPTPanguTokenizer class.
            ids (Union[mindspore.Tensor, np.ndarray, List[int]]): The token IDs to decode into text.
                If passed as a mindspore.Tensor or np.ndarray, it will be converted to a list of integers.
                This parameter is required.

        Returns:
            str: The decoded text corresponding to the provided token IDs.
                Whitespace characters ' ' will be replaced with spaces, '▂' will be replaced with spaces, and '▃' will
                be replaced with newline characters.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        if isinstance(ids, (mindspore.Tensor, np.ndarray)):
            ids = ids.tolist()

        if kwargs.get('skip_special_tokens', None) is True:
            ids = [token_id for token_id in ids if token_id not in self.all_special_ids]
        text = self.sp.decode(ids)
        if isinstance(text, list):
            text = text[0]
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')#.replace('⁇', self.unk_token)
        return text

__all__ = ['GPTPanguTokenizer']
