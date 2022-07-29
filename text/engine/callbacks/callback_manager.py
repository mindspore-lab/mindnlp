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
"""
Callback Manager
"""
from .callback import Callback

def _transfer(func):
    """
    Forward the call to the callback
    manager to each callback subclass.

    Args:
        func: callback function.
    """

    def wrapper(manager, *arg):
        returns = []
        for callback in manager.callbacks:
            returns.append(getattr(callback, func.__name__)(*arg))
        return returns

    return wrapper

class CallbackManager(Callback):
    """Callback Manager."""

    def __init__(self, callbacks=None):
        self.callbacks = []
        if callbacks:
            self.prepare_callbacks(callbacks)

    def prepare_callbacks(self, callbacks):
        """Check callback type."""
        if isinstance(callbacks, Callback):
            self.callbacks.append(callbacks)
        elif isinstance(callbacks, list):
            if all([isinstance(cb, Callback) for cb in callbacks]) is True:
                self.callbacks = callbacks
            else:
                obj = [not isinstance(cb, Callback) for cb in callbacks][0]
                raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
        else:
            raise TypeError(f"Expect callbacks in CallbackManager to be list or Callback. Got {type(callbacks)}.")

    @_transfer
    def train_begin(self):
        """Called once before the network executing."""

    @_transfer
    def train_end(self):
        """Called once after network training."""

    @_transfer
    def train_epoch_begin(self):
        """Called before each epoch beginning."""

    @_transfer
    def train_epoch_end(self, run_context):
        """Called after each epoch finished."""

    @_transfer
    def fetch_data_begin(self):
        """Called before fetch each batch/ds_sink_size data."""

    @_transfer
    def fetch_data_end(self):
        """Called after fetch each batch/ds_sink_size data."""

    @_transfer
    def train_step_begin(self, run_context):
        """Called before each step beginning."""

    @_transfer
    def train_step_end(self, run_context):
        """Called after each step finished."""

    @_transfer
    def ds_sink_begin(self):
        """Called before each data_sink beginning."""

    @_transfer
    def ds_sink_end(self):
        """Called after each data_sink finished."""

    @_transfer
    def load_model(self):
        """Called before loading model."""

    @_transfer
    def save_model(self):
        """Called before saving model."""

    @_transfer
    def load_checkpoint(self):
        """Called before loading checkpoint."""

    @_transfer
    def save_checkpoint(self):
        """Called before saving checkpoint."""

    @_transfer
    def evaluate_begin(self):
        """Called before evaluating epoch/steps/ds_size."""

    @_transfer
    def evaluate_end(self):
        """Called after evaluating epoch/steps/ds_size."""

    @_transfer
    def before_optimizer_step(self):
        """Called before optimizing."""

    @_transfer
    def after_optimizer_step(self):
        """Called after optimizing."""

    @_transfer
    def exception(self):
        """Called if having exceptions."""

class RunContext:
    """
        Provide information about the model.

        Provide information about original request to model function.
        Callback objects can stop the loop by calling request_stop() of run_context.
        This class needs to be used with :class:`mindspore.train.callback.Callback`.

        Args:
            trainer_args (dict): Holding the related information of model.
    """
    def __init__(self, trainer_args):
        if not isinstance(trainer_args, dict):
            raise TypeError("The argument 'original_args' of RunContext should be dict type, "
                            "but got {}.".format(type(trainer_args)))
        self.trainer_args = trainer_args
        # self._stop_requested = False

    def __getattr__(self, att):
        return self.trainer_args[att]
