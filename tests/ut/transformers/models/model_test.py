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
"""ModelTest test case"""
import gc
import unittest

import mindspore
from mindnlp import ms_jit

class ModelTest(unittest.TestCase):
    r"""
    Test model bert
    """
    def setUp(self):
        """setup"""
        self.use_amp = mindspore.get_context('device_target') == 'Ascend'

    def tearDown(self) -> None:
        """tear down"""
        gc.collect()

    def modeling(self, model, inputs, jit):
        """modeling"""
        def forward(input_ids):
            outputs, pooled = model(input_ids)
            return outputs, pooled

        if jit:
            forward = ms_jit(forward)
        outputs = forward(inputs)

        return outputs
