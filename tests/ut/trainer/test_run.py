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
"""Test Trainer _run function"""

import unittest
import numpy as np
import mindspore.nn as nn
import mindspore.dataset as ds
from text.engine.trainer import Trainer
from text.engine.callbacks.timer_callback import TimerCallback
np.random.seed(1)

class MyDataset:
    """Dataset"""
    def __init__(self):
        self.data = np.random.randn(100, 3).astype(np.float32)  # 自定义数据
        self.label = np.random.randn(100, 1).astype(np.float32)  # 自定义标签
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

class MyModel(nn.Cell):
    """Model"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 1)
    def construct(self, inputs):
        output = self.fc(inputs)
        return output

class TestTrainerRun(unittest.TestCase):
    r"""
    Test Trainer Run
    """
    def setUp(self):
        self.input = None
        net = MyModel()
        loss_fn = nn.MSELoss()
        optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)
        dataset_generator = MyDataset()
        train_dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        callbacks = []
        timer_callback = TimerCallback(print_steps=10)
        callbacks.append(timer_callback)
        self.trainer = Trainer(network=net, train_dataset=train_dataset, epochs=2,
                               loss_fn=loss_fn, optimizer=optimizer, callbacks=callbacks)

    def test_trainer_run_pynative(self):
        self.trainer.run(mode='pynative')

    def test_trainer_run_graph(self):
        self.trainer.run(mode='graph')
