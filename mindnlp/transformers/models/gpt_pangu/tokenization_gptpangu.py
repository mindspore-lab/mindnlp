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
# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
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
    # Ref: https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenization_jieba.py
    vocab_files_names = {
        "model_file": "vocab.model"
    }

    def __init__(
            self,
            model_file,
            **kwargs
    ):
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
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp.id_to_piece(index)

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def decode(self, ids, **kwargs):
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
