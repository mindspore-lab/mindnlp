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
"""Test Callback function."""
import unittest

from mindnlp.engine.callbacks.timer_callback import TimerCallback
from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
from mindnlp.engine.callbacks.best_model_callback import BestModelCallback
from mindnlp.engine.callbacks.checkpoint_callback import CheckpointCallback

class TestCallbackRun(unittest.TestCase):
    r"""
    Test Callback.
    """
    def setUp(self):
        self.input = None

    def test_timer_callback_init(self):
        """Test Timer Callback Initialization."""
        try:
            timer_callback = TimerCallback(print_steps=-1)
        except Exception as exception:
            raise exception
        print(timer_callback)

    def test_earlystop_callback_init(self):
        """Test Early Stop Callback Initialization."""
        try:
            earlystop_callback = EarlyStopCallback(patience=2)
        except Exception as exception:
            raise exception
        print(earlystop_callback)

    def test_bestmodel_callback_init(self):
        """Test Best Model Callback Initialization."""
        try:
            bestmodel_callback = BestModelCallback(save_path='save')
        except Exception as exception:
            raise exception
        print(bestmodel_callback)

    def test_checkpoint_callback_init(self):
        """Test Checkpoint Callback Initialization."""
        try:
            checkpoint_callback = CheckpointCallback(save_path='save', epochs=1)
        except Exception as exception:
            raise exception
        print(checkpoint_callback)
