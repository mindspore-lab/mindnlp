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
from mindnlp._legacy.transforms import AddToken
from mindnlp.utils import less_min_minddata_compatible

def test_addtoken_begin():
    """test addtoken by dataset.map"""
    dataset = NumpySlicesDataset(data={"text": [['a', 'b', 'c', 'd', 'e']]})
    # Data before
    # |           text            |
    # +---------------------------+
    # | ['a', 'b', 'c', 'd', 'e'] |
    # +---------------------------+
    add_token_op = AddToken(token='TOKEN', begin=True)
    dataset = dataset.map(operations=add_token_op)
    # Data after
    # |           text            |
    # +---------------------------+
    # | ['TOKEN', 'a', 'b', 'c', 'd', 'e'] |
    # +---------------------------+
    data_after = next(dataset.create_tuple_iterator(output_numpy=True))[0]
    if less_min_minddata_compatible:
        assert data_after.tolist() == [b'TOKEN', b'a', b'b', b'c', b'd', b'e']
    else:
        assert data_after.tolist() == ['TOKEN', 'a', 'b', 'c', 'd', 'e']

def test_addtoken_end():
    """test addtoken by dataset.map"""
    dataset = NumpySlicesDataset(data={"text": [['a', 'b', 'c', 'd', 'e']]})
    # Data before
    # |           text            |
    # +---------------------------+
    # | ['a', 'b', 'c', 'd', 'e'] |
    # +---------------------------+
    add_token_op = AddToken(token='TOKEN', begin=False)
    dataset = dataset.map(operations=add_token_op)
    # Data after
    # |           text            |
    # +---------------------------+
    # | ['a', 'b', 'c', 'd', 'e', 'TOKEN'] |
    # +---------------------------+
    data_after = next(dataset.create_tuple_iterator(output_numpy=True))[0]
    if less_min_minddata_compatible:
        assert data_after.tolist() == [b'a', b'b', b'c', b'd', b'e', b'TOKEN']
    else:
        assert data_after.tolist() == ['a', 'b', 'c', 'd', 'e', 'TOKEN']
