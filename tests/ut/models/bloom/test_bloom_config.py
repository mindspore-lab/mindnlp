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
"""Test Bloom Config"""

import gc
import unittest
import pytest

from mindnlp.models.bloom import BloomConfig


class TestBloomConfig(unittest.TestCase):
    r"""
    Test Bloom Config
    """
    @pytest.mark.download
    def test_bloom_config(self):
        r"""
        Test Bloom Config from_pretrained
        """

        config = BloomConfig.from_pretrained('bigscience/bloom-560m')
        assert config.n_layer == 24

    def tearDown(self) -> None:
        gc.collect()
