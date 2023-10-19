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
"""Test the ErnieTokenizer"""
import pytest
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import ErnieTokenizer


def test_ernie_tokenizer_from_pretrained():
    """test ErnieTokenizer from pretrained."""
    texts = ['MindNLP是华为MindSpore自然语言处理套件！']
    test_dataset = GeneratorDataset(texts, 'text')

    ernie_tokenizer = ErnieTokenizer.from_pretrained('uie-base', return_token=True)
    test_dataset = test_dataset.map(operations=ernie_tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]

    assert len(dataset_after) == 20
    assert dataset_after.dtype == ms.string

@pytest.mark.skip("seems has errors on Github CI.")
def test_ernie_tokenizer_add_special_tokens():
    """test add special tokens."""
    ernie_tokenizer = ErnieTokenizer.from_pretrained('uie-base')
    cls_id = ernie_tokenizer.token_to_id("<CLS>")
    print(len(ernie_tokenizer))

    assert cls_id is None

    add_num = ernie_tokenizer.add_special_tokens({
        'cls_token': "<CLS>"
    })

    assert add_num == 1

    cls_id = ernie_tokenizer.token_to_id("<CLS>")
    assert cls_id == len(ernie_tokenizer) - 1
