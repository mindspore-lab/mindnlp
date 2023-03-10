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
Test TinyBert
"""
import unittest

import mindspore
import numpy as np

from mindnlp.models import tinybert

class TestTinyBert(unittest.TestCase):
    """
    Test TinyBert Models
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up config
        """

        cls.bert_config = tinybert.BertConfig(
            vocab_size_or_config_json_file=200)

    def test_bert_embedding(self):
        """
        Test BertEmbeddings
        """

        bert_embeddings = tinybert.BertEmbeddings(self.bert_config)
        input_ids = mindspore.Tensor(np.random.randint(0, 1000, (2, 128)))
        output = bert_embeddings(input_ids)
        assert output.shape == (2, 128, self.bert_config.hidden_size)
