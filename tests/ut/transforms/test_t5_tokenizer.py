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
"""Test the T5Tokenizer"""

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindnlp.transforms import T5Tokenizer

def test_t5_tokenizer_op():
    """test T5Tokenizer from pretrained."""
    texts = ['i make a small mistake when i\'m working!']
    test_dataset = GeneratorDataset(texts, 'text')

    tokenizer = T5Tokenizer.from_pretrained('t5-base', return_token=True)
    test_dataset = test_dataset.map(operations=tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]

    assert len(dataset_after) == 15
    assert dataset_after.dtype == ms.string
