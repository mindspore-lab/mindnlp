# Copyright 2022 Huawei Technologies Co., Ltd
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
common process
"""

from mindspore.dataset import text

def common_process(dataset, tokenizer, language, vocab):
    '''
    common process

    Args:
        dataset (GeneratorDataset|ZipDataset): dataset needs to be process
        tokenizer (TextTensorOperation): Tokenizer you what to used
        language (str): The language column name
        vocab (Vocab): The vocab to be used, defaults to None. If None, a new vocab will be created

    Returns:
        - **dataset** (MapDataset) -dataset after process
        - **newVocab** (Vocab) -new vocab created from dataset if 'vocab' is None

    '''

    if vocab is None :
        dataset = dataset.map(tokenizer, language)
        new_vocab = text.Vocab.from_dataset(dataset, language)
        return dataset.map(text.Lookup(new_vocab), language), new_vocab

    dataset = dataset.map(tokenizer, language)
    return dataset.map(text.Lookup(vocab), language)
