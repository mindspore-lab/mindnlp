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
# pylint: disable=C0103

import unittest
import numpy as np

from mindspore import nn
import mindspore.dataset as ds

from mindnlp.engine.evaluator import Evaluator
from mindnlp.common.metrics import Accuracy
from mindnlp.engine.callbacks.timer_callback import TimerCallback


class MyDataset:
    """Dataset"""
    def __init__(self):
        np.random.seed(1)
        self.data = np.random.randn(100, 3).astype(np.float32)
        self.label = list(np.random.choice([0, 1]).astype(np.float32) for i in range(100))
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

class MyModel(nn.Cell):
    """Model"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 2)
    def construct(self, data):
        output = self.fc(data)
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

    def test_evaluator_run(self):
        """test evaluator run pynative"""
        self.evaluator.run(tgt_columns='label')

    def test_evaluator_run_jit(self):
        """test evaluator run graph"""
        self.evaluator.run(tgt_columns='label', jit=False)
