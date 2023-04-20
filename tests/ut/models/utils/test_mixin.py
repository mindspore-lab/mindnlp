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
# pylint:disable=W0212
"""Test CellUtilMixin"""
import unittest
import numpy as np

import mindspore
from mindspore import Tensor

from mindnlp.abc import CellUtilMixin

class TestCellUtilMixin(unittest.TestCase):
    r"""
    Test CellUtilMixin
    """
    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_create_extended_attention_mask_for_decoder(self):
        r"""
        Test CellUtilMixin.create_extended_attention_mask_for_decoder
        """
        model = CellUtilMixin()

        input_ids = Tensor(np.random.randn(512, 32), mindspore.float32)
        shape = (512, 4)

        outputs = model.create_extended_attention_mask_for_decoder(shape, input_ids)
        assert outputs.shape == (512, 1, 4, 32)

    def test_invert_attention_mask(self):
        r"""
        Test CellUtilMixin.invert_attention_mask
        """
        model = CellUtilMixin()

        input_ids = Tensor(np.random.randn(512, 32), mindspore.float32)

        outputs = model.invert_attention_mask(input_ids)
        assert outputs.shape == (512, 1, 1, 32)

    def test_get_extended_attention_mask(self):
        r"""
        Test CellUtilMixin.get_extended_attention_mask
        """
        model = CellUtilMixin()

        input_ids = Tensor(np.random.randn(512, 32, 4), mindspore.float32)
        shape = (512, 4)

        outputs = model.get_extended_attention_mask(input_ids, shape)
        assert outputs.shape == (512, 1, 32, 4)

    def test_get_head_mask(self):
        r"""
        Test CellUtilMixin.get_head_mask
        """
        model = CellUtilMixin()

        input_ids = Tensor(np.random.randn(512, 32), mindspore.float32)
        num_hidden_layer = 5

        outputs = model.get_head_mask(input_ids, num_hidden_layer)
        assert outputs.shape == (512, 1, 32, 1, 1)

    def test__convert_head_mask_to_5d(self):
        r"""
        Test CellUtilMixin._convert_head_mask_to_5d
        """
        model = CellUtilMixin()

        input_ids = Tensor(np.random.randn(512, 32), mindspore.float32)
        num_hidden_layer = 5

        outputs = model._convert_head_mask_to_5d(input_ids, num_hidden_layer)
        assert outputs.shape == (512, 1, 32, 1, 1)
