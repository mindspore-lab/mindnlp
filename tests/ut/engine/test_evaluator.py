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
"""Test Evaluator with Callback function"""

import unittest
import numpy as np

import mindspore.nn as nn
import mindspore.dataset as ds

from text.engine.evaluator import Evaluator
from text.common.metrics import Accuracy
from text.engine.callbacks.timer_callback import TimerCallback


class MyDataset:
    """Dataset"""
    def __init__(self):
        np.random.seed(1)
        self.data = np.random.randn(100, 3).astype(np.float32)  # 自定义数据
        self.label = list(np.random.choice([0, 1]).astype(np.float32) for i in range(100)) # 自定义标签
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

class MyModel(nn.Cell):
    """Model"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 2)
    def construct(self, inputs):
        output = self.fc(inputs)
        return output

class TestEvaluatorRun(unittest.TestCase):
    r"""
    Test Evaluator Run
    """
    def setUp(self):
        self.input = None
        net = MyModel()
        dataset_generator = MyDataset()
        metric = Accuracy()
        callbacks = [TimerCallback()]
        eval_dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        self.evaluator = Evaluator(network=net, eval_dataset=eval_dataset, metrics=metric,
                                   callbacks=callbacks, batch_size=10)

    def test_evaluator_run_pynative(self):
        self.evaluator.run(mode='pynative')

    def test_evaluator_run_graph(self):
        self.evaluator.run(mode='graph')
