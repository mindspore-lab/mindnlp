# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
test loss
"""
import pytest
import numpy as np

import mindspore
from mindspore.common.tensor import Tensor

from mindtext.common.loss.binary_cross_entropy import BinaryCrossEntropy
from mindtext.common.loss import CrossEntropy

@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_binary_cross_entropy():
    '''this function is test BinaryCrossEntropy'''
    weight = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 3.3, 2.2]]), mindspore.float32)
    loss = BinaryCrossEntropy(weight=weight, reduction='mean')
    logits = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]), mindspore.float32)
    labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]), mindspore.float32)
    output = loss(logits, labels)
    assert output == 1.8952923

@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_cross_entropy():
    '''this function is test CrossEntropy'''
    loss = CrossEntropy(sparse=True)
    logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mindspore.float32)
    labels_np = np.array([1]).astype(np.int32)
    labels = Tensor(labels_np)
    output = loss(logits, labels)
    assert output == 67
