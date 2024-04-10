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
Callback for Early Stop.
"""
from mindspore import log
from mindnlp.abc import Callback

class EarlyStopCallback(Callback):
    """
    Stop training without getting better after n epochs.

    Args:
        patience (int): Numbers of epochs evaluations without raising. Default:10.
        larger_better (bool): Whether the larger value of the metric is better. Default:True.
    """
    def __init__(self, patience=10, larger_better=True):
        self.wait = 0
        self.patience = patience
        self.best_metrics_values = []
        self.larger_better = larger_better

    def evaluate_end(self, run_context):
        """
        Called after evaluating.

        Args:
            run_context (RunContext): Information about the model.

        """
        metrics_values = run_context.metrics_values
        if metrics_values is None:
            return
        if self.is_better_metric_value(metrics_values):
            self.wait = 0
            self.best_metrics_values = metrics_values
        else:
            self.wait += 1
        if self.wait >= self.patience:
            run_context.earlystop = True
            log.warning(f"After {self.wait} Evaluations, no improvement for "
                        f"metric `{run_context.metrics_names}`(best value: {self.best_metrics_values})")

    def is_better_metric_value(self, metrics_values):
        """
        Compare each metrics values with the best metrics values.

        Args:
            metrics_values (float): metrics values used to compared with the best metrics values so far.

        """
        if self.best_metrics_values == {}:
            return True
        values_larger = metrics_values > self.best_metrics_values
        better_or_not = values_larger & self.larger_better
        return better_or_not
