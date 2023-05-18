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
"""Test Glove Vocab"""

import pytest
from mindnlp import Vocab


@pytest.mark.download
def test_vocab_glove_from_pretrained():
    """test glove vocab from URL"""

    vocab_50d = Vocab.from_pretrained("glove.6B.50d")
    vocab_100d = Vocab.from_pretrained("glove.6B.100d")
    vocab_200d = Vocab.from_pretrained("glove.6B.200d")
    vocab_300d = Vocab.from_pretrained("glove.6B.300d")

    assert len(vocab_50d) == 4 * 10 ** 5 + 2
    assert len(vocab_100d) == 4 * 10 ** 5 + 2
    assert len(vocab_200d) == 4 * 10 ** 5 + 2
    assert len(vocab_300d) == 4 * 10 ** 5 + 2
