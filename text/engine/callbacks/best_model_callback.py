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
Callback for loading best model
"""

import os
import shutil
from typing import Union
import mindspore
from mindspore import log
from ...abc import Callback

class BestModelCallback(Callback):
    """
    Save the model with the best `metrics` value and reload the model at the end of the training.
    The best model can only be loaded at the end of the training.

    Args:
        save_folder (str): Folder for saving.
        metrics (Union[Callback, str]): Metric value. Default: None.
        larger_better (bool): Whether the larger `metrics`, the better `metrics`. Default: True.
        only_state_dict (bool): Whether to save only the parameters of the model.
        When `model_save_fn` is not None, the value is invalid. Default: True.

    """
    def __init__(self, save_folder=None, metrics=None, larger_better=True, \
            only_state_dict=True):
        self._set_metric_name(metrics, larger_better)
        self._log_name = self.__class__.__name__

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
        else:
            os.makedirs(os.path.expanduser('~'), exist_ok=True)

        self.save_folder = save_folder
        self.only_state_dict = only_state_dict
        self.encounter_exception = False

    def _set_metric_name(self, metric_name, larger_better):
        self.metric_name = str(metric_name) if metric_name is not None else None
        if self.metric_name is not None:
            self.larger_better = bool(larger_better)
        if larger_better:
            self.metric_value = float('-inf')
        else:
            self.metric_value = float('inf')
        self._real_metric_name = self.metric_name

    def _get_metric_value(self, metric_name, real_metric_name, metrics):
        """
        Find `metric_name` in `res` and return it. If `metric_name` is not found in `res`, try to find
        it in `_real_metric_name`.

        Args:
            metric_name (str): Metric name.
            real_metric_name (Union[str, None]): Metric name.
            metrics (dict): Results of evaluation.

        Raises:
            RuntimeError: If `metric_name` and `real_metric_name` are not found in `res`.

        """
        if metrics is None or metric_name is None:
            return metric_name, None

        if metric_name in metrics:
            return metric_name, metrics[metric_name]

        if real_metric_name in metrics:
            return real_metric_name, metrics[real_metric_name]
        raise RuntimeError("`metric_name` and `real_metric_name` are not found in `res`.")

    def get_metric_value(self, metrics)->Union[float, None]:
        """
        Get metric value.

        Args:
            metrics (Dict): Results of evaluation.

        Returns:
            - **metric_value** (Union[float, None]) - Metric value.

        """
        if metrics is None or self.metric_name is None:
            return None
        metric_name, metric_value = self._get_metric_value(metric_name=self.metric_name,
                                                           real_metric_name=self._real_metric_name,
                                                           metrics=metrics)
        if metric_value is None:
            return metric_value

        if isinstance(self.metric_name, str) and self._real_metric_name == self.metric_name \
                and metric_name != self.metric_name:
            log.warning(f'We can not find metric:`{self.metric_name}` for `{self._log_name}` in the evaluation '
                        f'result (with keys as {list(metrics.keys())}), we use the `{metric_name}` as the '
                        f'metric.', once=True)
        elif isinstance(self.metric_name, str) and self._real_metric_name != self.metric_name \
                and metric_name != self._real_metric_name:
            log.warning(f'Change of metric detected for `{self._log_name}`. The expected metric is:'
                        f'`{self.metric_name}`, last used metric is:`{self._real_metric_name}` and '
                        f'current metric is:`{metric_name}`. Please consider using a customized metric '
                        f'function when the evaluation results are varying between validation.')

        self._real_metric_name = metric_name
        return metric_value

    def _is_former_metric_value_better(self, metric_value1, metric_value2):
        """
        Check whether `metric_value1` is better than `metric_value2`.

        Args:
            metric_value1 (float): Metric value 1 to compare.
            metric_value2 (float): Metric value 2 to compare.

        Returns:
            - **is_better** (bool) - `metric_value1` is better than `metric_value2`.

        """
        if metric_value1 is None and metric_value2 is None:
            return True
        if metric_value1 is None:
            return False
        if metric_value2 is None:
            return True
        is_better = False
        if (self.larger_better and metric_value1 > metric_value2) or \
                (not self.larger_better and metric_value1 < metric_value2):
            is_better = True
        return is_better

    def _is_better_metric_value(self, metric_value, keep_if_better=True):
        """
        Check whether `metric_value` is better. If `metric_value` is None, return False.

        Args:
            metric_value (float): Metric value to compare.
            keep_if_better (bool): If `metric_value` is better, keep it. Default: True.

        Returns:
            - **is_better** (bool) - Whether `metric_value` is better.

        """
        if metric_value is None:
            return False
        is_better = self._is_former_metric_value_better(metric_value, self.metric_value)
        if keep_if_better and is_better:
            self.metric_value = metric_value
        return is_better

    def is_better_metrics(self, metrics, keep_if_better=True):
        """
        Check whether the given `metrics` is better than the previous one. If no relevant metric is found
        in `metrics`, return False.

        Args:
            metrics (Dict): Results of evaluation.
            keep_if_better (bool): If the metric value is better, keep it. Default: True.

        Returns:
            - **is_better** (bool) - Whether the metric value is better.
        """
        metric_value = self.get_metric_value(metrics)
        if metric_value is None:
            return False
        is_better = self._is_better_metric_value(metric_value, keep_if_better=keep_if_better)
        return is_better

    def _prepare_save_folder(self):
        if not hasattr(self, 'real_save_folder'):
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder, exist_ok=True)
            self.real_save_folder = self.save_folder

    def _save_model(self, save_folder, run_context, only_state_dict=False):
        """
        Function to save model.

        Args:
            save_folder (str): Folder for saving.
            run_context (RunContext): Information about the model.
            only_state_dict (bool): Whether to save only the parameters of the model. Default: False.

        """
        ckpt_name = "best_so_far.ckpt"
        model_path = os.path.join(save_folder, ckpt_name)

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        if only_state_dict:
            state_dict = run_context.evaluator.network
            mindspore.save_checkpoint(state_dict, model_path)
        else:
            mindspore.save_checkpoint(run_context, model_path)

    def _load_model(self, load_folder, run_context, only_state_dict):
        """
        Function to load model.

        Args:
            save_folder (str): Folder for loading.
            run_context (RunContext): Information about the model.

        """
        ckpt_name = "best_so_far.ckpt"
        model_path = os.path.join(load_folder, ckpt_name)
        param_dict = mindspore.load_checkpoint(model_path)

        if only_state_dict:
            mindspore.load_param_into_net(run_context.network, param_dict)
        else:
            mindspore.load_param_into_net(run_context.network, param_dict)
            mindspore.load_param_into_net(run_context.optimizer, param_dict)

    def evaluate_end(self, run_context):
        """
        Called after evaluating epoch/steps/ds_size.

        Args:
            run_context (RunContext): Information about the model.

        Raises:
            TypeError: If `save_folder` is not a directory.

        """
        metrics = run_context.metrics_result
        if self.is_better_metrics(metrics, keep_if_better=True):
            self._prepare_save_folder()
            if self.real_save_folder:
                self._save_model(save_folder=self.real_save_folder, run_context=run_context, \
                    only_state_dict=self.only_state_dict)

    def train_end(self, run_context):
        '''
        Called once after network training.

        Args:
            run_context (RunContext): Information about the model.

        '''
        if abs(self.metric_value) != float('inf'):
            if not self.encounter_exception:
                log.info(f"Loading best model from {self.real_save_folder} with {self._real_metric_name}: "
                         f"{self.metric_value}...")
                self._load_model(load_folder=self.real_save_folder, run_context=run_context, \
                    only_state_dict=self.only_state_dict)

            if self.delete_after_after:
                self._delete_folder()

    def exception(self, run_context):
        """Called if having exceptions."""
        earlystop = run_context.earlystop
        if not earlystop:
            self.encounter_exception = True

    def _delete_folder(self):
        if getattr(self, 'real_save_folder', None):
            log.info(f"Deleting {self.real_save_folder}...")
            shutil.rmtree(self.real_save_folder, ignore_errors=True)
            os.rmdir(self.save_folder)
            log.debug(f"Since {self.save_folder} is an empty folder, it has been removed.")
        elif hasattr(self, 'buffer'):
            self.buffer.close()
            del self.buffer
