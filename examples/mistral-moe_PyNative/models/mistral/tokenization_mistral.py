# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Mistral模型的分词器实现."""

import os
from typing import List, Optional, Union

import sentencepiece as spm


class MistralTokenizer:
    """
    基于SentencePiece构建的Mistral分词器.
    
    参数:
        vocab_file (str): SentencePiece所需的词汇表文件(通常是`.model`文件).
        unk_token (str, optional): 未知token. 默认为"<unk>".
        bos_token (str, optional): 序列开始token. 默认为"<s>".
        eos_token (str, optional): 序列结束token. 默认为"</s>".
        pad_token (str, optional): 填充token. 默认为None.
        sp_model_kwargs (dict, optional): 传递给SentencePiece模型的额外参数. 默认为None.
        add_bos_token (bool, optional): 是否添加序列开始token. 默认为True.
        add_eos_token (bool, optional): 是否添加序列结束token. 默认为False.
    """

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs=None,
        add_bos_token=True,
        add_eos_token=False,
        **kwargs
    ):
        self.vocab_file = vocab_file
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model_kwargs = sp_model_kwargs or {}
        
        # 加载SentencePiece模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        
        # 设置特殊token
        self.unk_token_id = self.sp_model.unk_id()
        self.bos_token_id = self.sp_model.bos_id()
        self.eos_token_id = self.sp_model.eos_id()
        self.pad_token_id = self.sp_model.pad_id() if hasattr(self.sp_model, 'pad_id') else self.unk_token_id

    @property
    def vocab_size(self):
        """返回词汇表大小."""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """返回词汇表字典."""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def tokenize(self, text: str) -> List[str]:
        """对字符串进行分词."""
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """使用词汇表将token(字符串)转换为id."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """使用词汇表将索引(整数)转换为token(字符串)."""
        token = self.sp_model.id_to_piece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """将token序列(字符串)转换为单个字符串."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # Make sure that special tokens are not decoded using sentencepiece model
            if token in [self.bos_token, self.eos_token, self.unk_token, self.pad_token]:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating
        and adding special tokens.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> List[int]:
        """
        Converts a string or list of strings to a list of token ids.
        
        Args:
            text: The input text.
            add_special_tokens: Whether to add special tokens.
            padding: Whether to pad the sequence.
            max_length: Maximum length of the sequence.
            return_tensors: The type of tensor to return (not implemented yet).
            
        Returns:
            List of token ids.
        """
        if isinstance(text, str):
            tokens = self.tokenize(text)
            token_ids = [self._convert_token_to_id(token) for token in tokens]
            
            if add_special_tokens:
                token_ids = self.build_inputs_with_special_tokens(token_ids)
                
            if max_length is not None and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                
            if padding and max_length is not None:
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
                
            return token_ids
        else:
            # Handle batch encoding
            return [self.encode(t, add_special_tokens, padding, max_length, return_tensors) for t in text]

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Converts a sequence of token ids to a string.
        
        Args:
            token_ids: List of token ids.
            skip_special_tokens: Whether to remove special tokens.
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces.
            
        Returns:
            The decoded string.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
            
        tokens = [self._convert_id_to_token(idx) for idx in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [self.bos_token, self.eos_token, self.unk_token, self.pad_token]]
            
        text = self.convert_tokens_to_string(tokens)
        
        if clean_up_tokenization_spaces:
            text = text.strip()
            
        return text

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts token ids to tokens."""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(idx) for idx in ids]

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts tokens to token ids."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]
