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
Abstract class for Callbacks
"""

class Callback():
    """
    Abstract base class used to build a callback class. Callbacks are context managers
    which will be entered and exited when passing into the Model.
    You can use this mechanism to do some custom operations.

    Callback function can perform some operations before and after step or epoch.
    To create a custom callback, subclass Callback and override the method associated
    with the stage of interest.
    """

    def train_begin(self, run_context):
        """Called once before network training."""

    def train_end(self, run_context):
        """Called once after network training."""

    def train_epoch_begin(self, run_context):
        """Called before each train epoch beginning."""

    def train_epoch_end(self, run_context):
        """Called after each train epoch finished."""

    def fetch_data_begin(self, run_context):
        """Called before fetch each batch/ds_sink_size data."""

    def fetch_data_end(self, run_context):
        """Called after fetch each batch/ds_sink_size data."""

    def train_step_begin(self, run_context):
        """Called before each train step beginning."""

    def train_step_end(self, run_context):
        """Called after each train step finished."""

    def forward_begin(self, run_context):
        """Called before each forward beginning."""

    def forward_end(self, run_context):
        """Called after each step finished."""

    def backward_begin(self, run_context):
        """Called before each forward beginning."""

    def backward_end(self, run_context):
        """Called after each backward finished."""

    def ds_sink_begin(self, run_context):
        """Called before each data_sink beginning."""

    def ds_sink_end(self, run_context):
        """Called after each data_sink finished."""

    def load_model(self, run_context):
        """Called before loading model."""

    def save_model(self, run_context):
        """Called before saving model."""

    def evaluate_begin(self, run_context):
        """Called before evaluating."""

    def evaluate_end(self, run_context):
        """Called after evaluating."""

    def exception(self, run_context):
        """Called if having exceptions."""
