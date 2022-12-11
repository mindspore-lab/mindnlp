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
Callback for saving and loading best model
"""
import os
import mindspore
from mindnlp.abc import Callback

class BestModelCallback(Callback):
    r"""
    Save the model with the best `metrics` value and reload the model at the end of the training.
    The best model can only be loaded at the end of the training.

    Args:
        save_path (str): Folder for saving.
        larger_better (bool): Whether the larger `metrics`, the better `metrics`. Default: True.
        auto_load (bool): Whether load the best model at the end of the training.
        save_on_exception (bool): Whether save the model on exception.

    """
    def __init__(self, save_path=None, larger_better=True, auto_load=False, save_on_exception=False):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        else:
            os.makedirs(os.path.expanduser('~'), exist_ok=True)

        self.save_path = save_path
        self.larger_better = larger_better
        self.auto_load = auto_load
        self.best_metrics_values = []
        self.save_on_exception = save_on_exception

    def evaluate_end(self, run_context):
        r"""
        Called after evaluating.

        Args:
            run_context (RunContext): Information about the model.

        """
        metrics_values = run_context.metrics_values
        if metrics_values is None:
            return
        if self.is_better_metric_value(metrics_values):
            self.best_metrics_values = metrics_values
            self._save_model(run_context=run_context)

    def train_end(self, run_context):
        r"""
        Called once after network training and load the best model params.

        Args:
            run_context (RunContext): Information about the model.

        """
        if self.auto_load:
            print(f"Loading best model from {self.save_path} with {run_context.metrics_names}: "
                  f"{self.best_metrics_values}...")
            self._load_model(run_context)

    # to do

    # def exception(self, run_context):
    #     """Called if having exceptions."""
    #     pass
        # if self.save_on_exception:
        #     self._load_model(load_folder=self.real_save_path, run_context=run_context, \
        #         only_state_dict=self.only_state_dict)

    def is_better_metric_value(self, metrics_values):
        r"""
        Compare each metrics values with the best metrics values.

        Args:
            metrics_values (float): metrics values used to compared with the best metrics values so far.

        """
        if self.best_metrics_values == []:
            return True
        values_larger = (metrics_values > self.best_metrics_values)
        is_better = values_larger & self.larger_better
        return is_better

    def _save_model(self, run_context):
        r"""
        Function to save model.

        Args:
            save_path (str): Folder for saving.
            run_context (RunContext): Information about the model.
            only_state_dict (bool): Whether to save only the parameters of the model. Default: False.

        """
        model = run_context.network
        ckpt_name = "best_so_far.ckpt"
        mindspore.save_checkpoint(model, self.save_path + '/' + ckpt_name)
        print(f"---------------Best Model: {ckpt_name} "
              f"has been saved in epoch: {run_context.cur_epoch_nums - 1}.---------------")

    def _load_model(self, run_context):
        r"""
        Function to load model.

        Args:
            load_path (str): Folder for loading.
            run_context (RunContext): Information about the model.

        """
        ckpt_name = "best_so_far.ckpt"
        param_dict = mindspore.load_checkpoint(self.save_path + '/' + ckpt_name)
        mindspore.load_param_into_net(run_context.network, param_dict)
        run_context.callbacks = []
        print("---------------The model is already load the best model from best_so_far.ckpt.---------------")
