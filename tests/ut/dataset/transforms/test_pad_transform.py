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
"""Test the AddToken"""

from mindspore.dataset import NumpySlicesDataset
from mindnlp.dataset.transforms import PadTransform, Truncate
from mindnlp.utils import less_min_pynative_first

def test_pad_transform():
    """test PadTransform"""
    dataset = NumpySlicesDataset(data={"text": [[1, 2, 3, 4, 5]]})

    pad_transform_op = PadTransform(10, 0)
    dataset = dataset.map(operations=pad_transform_op)

    data_after = next(dataset.create_tuple_iterator(output_numpy=True))[0]
    assert data_after.tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]

def test_pad_transform_with_seq_length():
    """test PadTransform with seq_length"""
    dataset = NumpySlicesDataset(data={"text": [[1, 2, 3, 4, 5]]})

    pad_transform_op = PadTransform(10, 0, True)
    if less_min_pynative_first:
        dataset = dataset.map(pad_transform_op, 'text', ['text', 'len'], ['text', 'len'])
    else:
        dataset = dataset.map(pad_transform_op, 'text', ['text', 'len'])

    data_after = next(dataset.create_tuple_iterator(output_numpy=True))
    data = data_after[0]
    seq_len = data_after[1]

    assert data.tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    assert seq_len == 5

def test_pad_transform_with_seq_length_multi_transform():
    """test PadTransform with seq_length in multi-transforms."""
    dataset = NumpySlicesDataset(data={"text": [[1, 2, 3, 4, 5]]})

    pad_transform_op = PadTransform(10, 0, True)
    truncate_token = Truncate(3)

    if less_min_pynative_first:
        dataset = dataset.map([truncate_token, pad_transform_op], 'text', ['text', 'len'], ['text', 'len'])
    else:
        dataset = dataset.map([truncate_token, pad_transform_op], 'text', ['text', 'len'])

    data_after = next(dataset.create_tuple_iterator(output_numpy=True))
    data = data_after[0]
    seq_len = data_after[1]

    assert data.tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    assert seq_len == 3
