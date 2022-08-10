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
Evaluator for testing.
"""
from tqdm import tqdm

from mindspore import ms_function

from text.engine.callbacks.callback_manager import CallbackManager, RunContext
from ..abc import Metric

class Evaluator:
    r"""
    Evaluator to test the model.


    Args:
        network (Cell): A network for evaluating.
        eval_dataset (Dataset): A evaluating dataset iterator.
        batc_size (int): numbers of samples in each batch.
        metrcis (Optional[list[Metric], Metric]): List of metric objects which should be used
            while evaluating. Default:None.
        device (str): List of devices used for evaluating.
        callbacks (Optional[list[Callback], Callback]): List of callback objects which should be executed
            while training. Default: None.
         (str): Option for argument `level` in :func:`mindspore.build_train_network`, level for mixed
            precision training. Supports ["O0", "O2", "O3", "auto"]. Default: "O0".

            - "O0": Do not change.
            - "O2": Cast network to float16, keep BatchNorm run in float32, using dynamic loss scale.
            - "O3": Cast network to float16, the BatchNorm is also cast to float16, loss scale will not be used.
            - auto: Set level to recommended level in different devices. Set level to "O2" on GPU, set
              level to "O3" on Ascend. The recommended level is chosen by the expert experience, not applicable to all
              scenarios. User should specify the level for special network.

            "O2" is recommended on GPU, "O3" is recommended on Ascend.
            The BatchNorm strategy can be changed by `keep_batchnorm_fp32` settings in `kwargs`. `keep_batchnorm_fp32`
            must be a bool. The loss scale strategy can be changed by `loss_scale_manager` setting in `kwargs`.
            `loss_scale_manager` should be a subclass of :class:`mindspore.LossScaleManager`.
            The more detailed explanation of `amp_level` setting can be found at `mindspore.build_train_network`.
        boost_level (str):Option for argument `level` in `mindspore.boost`, level for boost mode
            training. Supports ["O0", "O1", "O2"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Enable the boost mode, the performance is improved by about 20%, and
              the accuracy is the same as the original accuracy.
            - "O2": Enable the boost mode, the performance is improved by about 30%, and
              the accuracy is reduced by less than 3%.

            If you want to config boost mode by yourself, you can set boost_config_dict as `boost.py`.

        dataset_sink_mode (bool): Determine whether the data should be passed through the dataset channel.
            Default: True.
            Configure pynative mode or CPU, the training process will be performed with dataset not sink.
    """

    def __init__(self, network, eval_dataset=None, batch_size=2, metrics=None, device=None, callbacks=None,
                 amp_level='O0', boost_level='O0', dataset_sink_mode=True):
        self.network = network
        self.batch_size = batch_size
        self.device = device
        self.callbacks = callbacks
        self.amp_level = amp_level
        self.boost_level = boost_level
        self.dataset_sink_mode = dataset_sink_mode
        self.earlystop = False

        self.check_metric_type(metrics)
        self.total = eval_dataset.get_dataset_size()
        self.eval_dataset = eval_dataset.batch(batch_size)

    def check_metric_type(self, metrics):
        """Check metrics type."""
        self.metrics = []
        if not metrics:
            raise ValueError("For Evaluator, the model argument 'metrics' can not be None or empty, "
                             "you should set the argument 'metrics' for model.")
        if isinstance(metrics, Metric):
            self.metrics.append(metrics)
        elif isinstance(metrics, list):
            if all([isinstance(mc, Metric) for mc in metrics]) is True:
                self.metrics = metrics
            else:
                obj = [not isinstance(mc, Metric) for mc in metrics][0]
                raise TypeError(f"Expect sub-classes of Metrics. Got {type(obj)}")
        else:
            raise TypeError(f"Expect metrics to be list or Metrics. Got {type(metrics)}.")

    def check_amp_level_arg(self, optimizer, amp_level):
        """Check mixed-precision argument rules."""
        raise NotImplementedError

    def check_for_graph_cell(self, kwargs):
        """Check network rules of GraphCell."""
        raise NotImplementedError

    def build_boost_network(self, *kwargs):
        """Build boost network."""
        raise NotImplementedError

    def check_reuse_dataset(self, dataset):
        """Check if dataset is being used by other models under the data sink mode."""
        if not hasattr(dataset, '__model_hash__'):
            dataset.__model_hash__ = hash(self)
        if hasattr(dataset, '__model_hash__') and dataset.__model_hash__ != hash(self):
            raise RuntimeError("The dataset object had been used in other model by model.train(...), "
                               "please create a new dataset.")

    def run(self, mode='pynative'):
        """Evaluating function entry."""
        args_dict = vars(self)
        run_context = RunContext(args_dict)
        self.callback_manager = CallbackManager(callbacks=self.callbacks)
        self.callback_manager.evaluate_begin(run_context)
        self.clear_metrics()
        self._run(mode)
        self.callback_manager.evaluate_end(run_context)
        self.earlystop = run_context.earlystop

    def _run(self, mode):
        """Evaluating process for non-data sinking mode. The data would be passed to network directly."""
        with tqdm(total=self.total) as t:
            t.set_description('Evaluate')
            for data in self.eval_dataset.create_tuple_iterator():
                inputs = data[0]
                labels = data[1]
                if mode == 'pynative':
                    outputs = self._run_step(inputs)
                    self.update_metrics(outputs, labels)
                elif mode == 'graph':
                    outputs = ms_function(self._run_step)(inputs)
                    self.update_metrics(outputs, labels)
                t.update(self.batch_size)
        t.close()
        self.metrics_result, self.metrics_names, self.metrics_values = self.get_metrics()
        print(f'Evaluate Score: {self.metrics_result}')

    def _run_ds_sink(self):
        """Evaluating process for data sinking mode."""
        raise NotImplementedError

    def _run_step(self, inputs):
        """Core process of each step."""
        outputs = self.network(inputs)
        return outputs

    def get_metrics(self):
        """Get all metrics values."""
        metrics = {}
        metrics_names = []
        metrics_values = []
        for metric in self.metrics:
            key = metric.get_metric_name()
            metrics_names.append(key)
            metrics[key] = metric.eval()
            metrics_values.append(metrics[key])
        return metrics, metrics_names, metrics_values

    def clear_metrics(self):
        """Clear metrics values."""
        for metric in self.metrics:
            metric.clear()

    def update_metrics(self, outputs, labels):
        """Update metrics values."""
        for metric in self.metrics:
            metric.updates(outputs, labels)
