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
Callback Manager.
"""
from mindnlp.abc import Callback

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
    """
    Callback Manager.

    Args:
        callbacks (Optional[list[Callback], Callback]): List of callback objects which should be executed
            while training. Default: None.

    """

    def __init__(self, callbacks):
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []

    @_transfer
    def train_begin(self, run_context):
        """Called once before the network executing."""

    @_transfer
    def train_end(self, run_context):
        """Called once after network training."""

    @_transfer
    def train_epoch_begin(self, run_context):
        """Called before each epoch beginning."""

    @_transfer
    def train_epoch_end(self, run_context):
        """Called after each epoch finished."""

    @_transfer
    def fetch_data_begin(self, run_context):
        """Called before fetch each batch/ds_sink_size data."""

    @_transfer
    def fetch_data_end(self, run_context):
        """Called after fetch each batch/ds_sink_size data."""

    @_transfer
    def train_step_begin(self, run_context):
        """Called before each train step beginning."""

    @_transfer
    def train_step_end(self, run_context):
        """Called after each train step finished."""

    @_transfer
    def forward_begin(self, run_context):
        """Called before each forward beginning."""

    @_transfer
    def forward_end(self, run_context):
        """Called after each step finished."""

    @_transfer
    def backward_begin(self, run_context):
        """Called before each forward beginning."""

    @_transfer
    def backward_end(self, run_context):
        """Called after each backward finished."""

    @_transfer
    def ds_sink_begin(self, run_context):
        """Called before each data_sink beginning."""

    @_transfer
    def ds_sink_end(self, run_context):
        """Called after each data_sink finished."""

    @_transfer
    def load_model(self, run_context):
        """Called before loading model."""

    @_transfer
    def save_model(self, run_context):
        """Called before saving model."""

    @_transfer
    def evaluate_begin(self, run_context):
        """Called before evaluating."""

    @_transfer
    def evaluate_end(self, run_context):
        """Called after evaluating."""

    @_transfer
    def exception(self, run_context):
        """Called if having exceptions."""

class RunContext:
    """
        Provide information about the model.
        This class needs to be used with :class:`mindspore.train.callback.Callback`.

        Args:
            engine_args (dict): Holding the related information of model.
    """
    def __init__(self, engine_args):
        if not isinstance(engine_args, dict):
            raise TypeError(f"The argument 'original_args' of RunContext should be dict type, "
                            f"but got {type(engine_args)}.")
        for arg, value in engine_args.items():
            setattr(self, arg, value)
