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
from text.engine.evaluator import Evaluator
from text.common.metrics import Accuracy
from text.engine.callbacks.timer_callback import TimerCallback
from text.engine.callbacks.earlystop_callback import EarlyStopCallback

np.random.seed(1)

class MyDataset:
    """Dataset"""
    def __init__(self):
        self.data = np.random.randn(20, 3).astype(np.float32)
        self.label = list(np.random.choice([0, 1]).astype(np.float32) for i in range(20))
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
        # 1. define dataset
        self.dataset_generator = MyDataset()
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        # 2. define Models & Loss & Optimizer
        self.net = MyModel()
        self.loss_fn = nn.MSELoss()
        self.optimizer = nn.Adam(self.net.trainable_params(), learning_rate=0.01)
        # 3. define forward function
        net = self.net
        loss_fn = self.loss_fn
        def foward_fn(*data):
            logits = net(data[0])
            loss = loss_fn(logits, data[1])
            return loss, logits
        self.foward_fn = foward_fn
        # 4. define callbacks
        self.timer_callback_epochs = TimerCallback(print_steps=-1)
        self.earlystop_callback = EarlyStopCallback(patience=2)
        self.callbacks = [self.timer_callback_epochs, self.earlystop_callback]
        # 5. define metrics
        self.metric = Accuracy()
        # 6. define evaluator
        evaluator = Evaluator(network=self.net, eval_dataset=eval_dataset, metrics=self.metric, batch_size=4)
        # 7. define trainer
        self.pure_trainer = Trainer(train_dataset=train_dataset, evaluator=evaluator, foward_fn=self.foward_fn,
                                    epochs=2, batch_size=4, optimizer=self.optimizer)

    def test_pure_trainer_pynative(self):
        # 8. trainer run
        self.pure_trainer.run(mode='pynative')

    def test_pure_trainer_graph(self):
        self.pure_trainer.run(mode='graph')

    def test_trainer_timer_pynative(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        evaluator = Evaluator(network=self.net, eval_dataset=eval_dataset, metrics=self.metric,
                              callbacks=self.timer_callback_epochs, batch_size=4)
        trainer = Trainer(train_dataset=train_dataset, evaluator=evaluator, foward_fn=self.foward_fn,
                          epochs=2, batch_size=4, optimizer=self.optimizer)
        trainer.run(mode='pynative')

    def test_trainer_timer_graph(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        evaluator = Evaluator(network=self.net, eval_dataset=eval_dataset, metrics=self.metric,
                              callbacks=self.timer_callback_epochs, batch_size=4)
        trainer = Trainer(train_dataset=train_dataset, evaluator=evaluator, foward_fn=self.foward_fn,
                          epochs=2, batch_size=4, optimizer=self.optimizer)
        trainer.run(mode='graph')

    def test_trainer_earlystop_pynative(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        evaluator = Evaluator(network=self.net, eval_dataset=eval_dataset, metrics=self.metric,
                              callbacks=self.earlystop_callback, batch_size=4)
        trainer = Trainer(train_dataset=train_dataset, evaluator=evaluator, foward_fn=self.foward_fn,
                          epochs=10, batch_size=4, optimizer=self.optimizer, callbacks=self.timer_callback_epochs)
        trainer.run(mode='pynative')

    def test_trainer_earlystop_graph(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=False)
        evaluator = Evaluator(network=self.net, eval_dataset=eval_dataset, metrics=self.metric,
                              callbacks=self.earlystop_callback, batch_size=4)
        trainer = Trainer(train_dataset=train_dataset, evaluator=evaluator, foward_fn=self.foward_fn,
                          epochs=10, batch_size=4, optimizer=self.optimizer, callbacks=self.timer_callback_epochs)
        trainer.run(mode='graph')
