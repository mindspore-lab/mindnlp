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
from inspect import signature
from tqdm.autonotebook import tqdm
from mindspore import log, mutable
from mindnlp import ms_jit
from mindnlp.abc import Metric
from mindnlp.engine.callbacks.callback_manager import CallbackManager, RunContext


class Evaluator:
    r"""
    Evaluator to test the model.


    Args:
        network (Cell): A network for evaluating.
        eval_dataset (Dataset): A evaluating dataset iterator.
        batc_size (int): numbers of samples in each batch.
        metrcis (Optional[list[Metric], Metric]): List of metric objects which should be used
            while evaluating. Default:None.
        callbacks (Optional[list[Callback], Callback]): List of callback objects which should be executed
            while training. Default: None.
        jit (bool): Whether use Just-In-Time compile.
    """

    def __init__(self, network, eval_dataset=None, metrics=None, callbacks=None, jit=False):
        self.network = network
        self.callbacks = callbacks
        self.earlystop = False

        self._check_metric_type(metrics)
        self.eval_dataset = eval_dataset
        self.total = eval_dataset.get_dataset_size()

        self.callback_manager = CallbackManager(callbacks=self.callbacks)
        self.eval_func = self._prepare_eval_func(network, jit)

    def _prepare_eval_func(self, network, jit):
        def _run_step(inputs):
            """Core process of each step."""
            outputs = network(*inputs)
            return outputs
        if jit:
            return ms_jit(_run_step)
        return _run_step

    def _check_metric_type(self, metrics):
        """Check metrics type."""
        self.metrics = []
        if not metrics:
            raise ValueError("For Evaluator, the model argument 'metrics' can not be None or empty, "
                             "you should set the argument 'metrics' for model.")
        if isinstance(metrics, Metric):
            self.metrics.append(metrics)
        elif isinstance(metrics, list):
            if all(isinstance(mc, Metric) for mc in metrics) is True:
                self.metrics = metrics
            else:
                obj = [not isinstance(mc, Metric) for mc in metrics][0]
                raise TypeError(f"Expect sub-classes of Metrics. Got {type(obj)}")
        else:
            raise TypeError(f"Expect metrics to be list or Metrics. Got {type(metrics)}.")

    def _check_amp_level_arg(self, optimizer, amp_level):
        """Check mixed-precision argument rules."""
        raise NotImplementedError

    def _check_for_graph_cell(self, kwargs):
        """Check network rules of GraphCell."""
        raise NotImplementedError

    def _build_boost_network(self, *kwargs):
        """Build boost network."""
        raise NotImplementedError

    def _check_reuse_dataset(self, dataset):
        """Check if dataset is being used by other models under the data sink mode."""
        if not hasattr(dataset, '__model_hash__'):
            dataset.__model_hash__ = hash(self)
        if hasattr(dataset, '__model_hash__') and dataset.__model_hash__ != hash(self):
            raise RuntimeError("The dataset object had been used in other model by model.train(...), "
                               "please create a new dataset.")

    def run(self, tgt_columns=None):
        """
        Evaluating function entry.

        Args:
            tgt_columns (Optional[list[str], str]): Target label column names for loss function.

        """
        args_dict = vars(self)
        run_context = RunContext(args_dict)
        self.callback_manager.evaluate_begin(run_context)
        self.clear_metrics()
        _ = self._run(tgt_columns)
        self.callback_manager.evaluate_end(run_context)
        self.earlystop = getattr(run_context, 'earlystop', False)

    def _run(self, tgt_columns=None):
        """Evaluating process for non-data sinking mode. The data would be passed to network directly."""
        self.network.set_train(False)
        with tqdm(total=self.total) as progress:
            progress.set_description('Evaluate')
            for data in self.eval_dataset.create_dict_iterator():
                inputs, tgts = self._data_process(data, tgt_columns)
                outputs = self.eval_func(inputs)
                self._update_metrics(outputs, *tgts)
                progress.update(1)

        progress.close()
        metrics_result, metrics_names, metrics_values = self._get_metrics()

        print(f'Evaluate Score: {metrics_result}')
        return metrics_result, metrics_names, metrics_values

    def _run_ds_sink(self):
        """Evaluating process for data sinking mode."""
        raise NotImplementedError

    def _get_metrics(self):
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

    def _update_metrics(self, outputs, *tgts):
        """Update metrics values."""
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        for metric in self.metrics:
            metric.update(logits, *tgts)
        return True

    def _data_process(self, data, tgt_columns):
        """Process data match the network construct"""
        # prepare input dataset.
        sig = signature(self.network.construct)
        net_args = sig.parameters
        inputs = ()
        for arg in net_args:
            if arg == 'self':
                continue
            if arg not in data.keys():
                if str(net_args[arg])[-4:] == 'None':
                    continue
            inputs = inputs + (data[arg],)
        # process target dataset.
        tgt_columns = self._prepare_tgt_columns(tgt_columns)
        tgts = ()
        for tgt_column in tgt_columns:
            tgts = tgts + (data[tgt_column],)
        return mutable(inputs), mutable(tgts)

    def _prepare_tgt_columns(self, tgt_columns):
        """Check and prepare target columns for training."""
        out_columns = []
        if tgt_columns is None:
            log.warning("In the process of training model, tgt_column can not be None.")
            return []
        if isinstance(tgt_columns, str):
            out_columns.append(tgt_columns)
        elif isinstance(tgt_columns, list):
            if all(isinstance(tgt_column, str) for tgt_column in tgt_columns) is True:
                out_columns = tgt_columns
            else:
                obj = [not isinstance(tgt_column, str) for tgt_column in tgt_columns][0]
                raise TypeError(f"Expect str of tgt_column. Got {type(obj)}")
        else:
            raise TypeError(f"Expect tgt_columns to be list or str. Got {type(tgt_columns)}.")
        return out_columns
