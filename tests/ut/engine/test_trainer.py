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
"""Test Trainer run function"""
# pylint: disable=C0103
# pylint: disable=W0621
import unittest
import numpy as np
from ddt import ddt, data
from mindspore import nn, Tensor
import mindspore.dataset as ds

from mindnlp.engine.trainer import Trainer
from mindnlp.metrics import Accuracy
from mindnlp.engine.callbacks.timer_callback import TimerCallback
from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
from mindnlp.engine.callbacks.best_model_callback import BestModelCallback
from mindnlp.engine.callbacks.checkpoint_callback import CheckpointCallback

np.random.seed(1)

class MyDataset:
    """Dataset"""
    def __init__(self):
        self.data = np.random.randn(20, 3).astype(np.float32)
        self.label = list(np.random.choice([0, 1]).astype(np.float32) for i in range(20))
        self.length = list(np.random.choice([0, 1]).astype(np.float32) for i in range(20))

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.length[index]

    def __len__(self):
        return len(self.data)

class MyModel(nn.Cell):
    """Model"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 1)

    def construct(self, data):
        output = self.fc(data)
        return output

class MyModel2(nn.Cell):
    """Model2"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 1)

    def construct(self, data, label, length):
        output = self.fc(data)
        label = label + label + length
        return output


class MyModelWithLoss(nn.Cell):
    """Model"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 1)
        self.loss = nn.MSELoss()
    def construct(self, data, label) -> Tensor:
        output = self.fc(data)
        loss = self.loss(output, label)
        return loss

@ddt
class TestTrainerRun(unittest.TestCase):
    r"""
    Test Trainer Run
    """
    def setUp(self):
        self.input = None
        # 1. define dataset
        self.dataset_generator = MyDataset()
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        self.train_dataset = train_dataset.batch(4)
        self.eval_dataset = eval_dataset.batch(4)
        # 2. define Models & Loss & Optimizer
        self.net = MyModel()
        self.net_2 = MyModel2()
        self.net.update_parameters_name('net.')
        self.net_2.update_parameters_name('net_2.')

        self.loss_fn = nn.MSELoss()
        self.optimizer = nn.Adam(self.net.trainable_params(), learning_rate=0.01)
        # 3. define callbacks
        self.timer_callback_epochs = TimerCallback(print_steps=2)
        self.earlystop_callback = EarlyStopCallback(patience=2)
        self.bestmodel_callback = BestModelCallback(save_path='save/callback/best_model', auto_load=True)
        self.checkpoint_callback = CheckpointCallback(save_path='save/callback/ckpt_files', epochs=2,\
                                                      keep_checkpoint_max=2)
        self.callbacks = [self.timer_callback_epochs, self.earlystop_callback, self.bestmodel_callback]
        # 4. define metrics
        self.metric = Accuracy()
        # 5. define trainer


    def test_pure_trainer(self):
        """test_pure_trainer"""
        # 6. trainer run
        pure_trainer = Trainer(network=self.net, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                                metrics=self.metric, epochs=2, optimizer=self.optimizer,
                                loss_fn=self.loss_fn)
        pure_trainer.run(tgt_columns='label')


    def test_trainer_timer(self):
        """test_trainer_timer"""
        trainer = Trainer(network=self.net, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                          metrics=self.metric, epochs=2, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.timer_callback_epochs)
        trainer.run(tgt_columns='label')


    def test_trainer_earlystop(self):
        """test_trainer_earlystop"""
        trainer = Trainer(network=self.net, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                          metrics=self.metric, epochs=6, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.earlystop_callback)
        trainer.run(tgt_columns='label')


    def test_trainer_bestmodel(self):
        """test_trainer_bestmodel"""
        trainer = Trainer(network=self.net, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                          metrics=self.metric, epochs=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.bestmodel_callback)
        trainer.run(tgt_columns='label')


    def test_trainer_checkpoint(self):
        """test_trainer_checkpoint"""
        trainer = Trainer(network=self.net, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                          metrics=self.metric, epochs=7, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.checkpoint_callback)
        trainer.run(tgt_columns='label')


    def test_different_model(self):
        """test_different_model"""
        trainer = Trainer(network=self.net_2, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                          metrics=self.metric, epochs=2, optimizer=self.optimizer,
                          loss_fn=self.loss_fn)
        trainer.run(tgt_columns='length')


    def test_no_eval_in_trainer(self):
        """test_eval_in_trainer"""
        trainer = Trainer(network=self.net, train_dataset=self.train_dataset, epochs=2,
                          optimizer=self.optimizer, loss_fn=self.loss_fn)
        trainer.run(tgt_columns='length')


    def test_train_object_netword(self):
        """test_eval_in_trainer"""
        net = MyModelWithLoss()
        trainer = Trainer(network=net, train_dataset=self.train_dataset, epochs=2,
                          optimizer=self.optimizer)
        trainer.run()
