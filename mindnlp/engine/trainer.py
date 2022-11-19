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
# pylint: disable=W0212
# pylint: disable=no-name-in-module, ungrouped-imports
"""
Trainer for training.
"""
from inspect import signature
from tqdm import tqdm
from mindspore import ops
from mindspore import log, mutable
from mindspore.ops import value_and_grad
from mindnlp import ms_jit
from mindnlp.abc.callback import Callback
from mindnlp.engine.callbacks.callback_manager import CallbackManager, RunContext
from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
from mindnlp.engine.callbacks.best_model_callback import BestModelCallback
from mindnlp.engine.evaluator import Evaluator

class Trainer:
    r"""
    Trainer to train the model.


    Args:
        network (Cell): A training network.
        train_dataset (Dataset): A training dataset iterator. If `loss_fn` is defined, the data and label will be
            passed to the `network` and the `loss_fn` respectively, so a tuple (data, label)
            should be returned from dataset. If there is multiple data or labels, set `loss_fn`
            to None and implement calculation of loss in `network`,
            then a tuple (data1, data2, data3, ...) with all data returned from dataset will be
            passed to the `network`.
        eval_dataset (Dataset): A evaluating dataset iterator. If `loss_fn` is defined, the data and label will be
            passed to the `network` and the `loss_fn` respectively, so a tuple (data, label)
            should be returned from dataset. If there is multiple data or labels, set `loss_fn`
            to None and implement calculation of loss in `network`,
            then a tuple (data1, data2, data3, ...) with all data returned from dataset will be
            passed to the `network`.
        metrcis (Optional[list[Metrics], Metrics]): List of metrics objects which should be used
            while evaluating. Default:None.
        epochs (int): Total number of iterations on the data. Default: 10.
        optimizer (Cell): Optimizer for updating the weights. If `optimizer` is None, the `network` needs to
            do backpropagation and update weights. Default value: None.
        loss_fn (Cell): Objective function. If `loss_fn` is None, the `network` should contain the calculation of loss
            and parallel if needed. Default: None.
        callbacks (Optional[list[Callback], Callback]): List of callback objects which should be executed
            while training. Default: None.

    """

    def __init__(self, network=None, train_dataset=None, eval_dataset=None, metrics=None, epochs=10,
                 loss_fn=None, optimizer=None, callbacks=None):
        self.network = network
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics = metrics
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.cur_epoch_nums = 0
        self.cur_step_nums = 0
        self.earlystop = False
        self.grad_fn = None
        if callbacks:
            self._prepare_callbacks(callbacks)
        self._prepare_eval()
        self.callback_manager = CallbackManager(callbacks=self.callbacks)

    def _prepare_callbacks(self, callbacks):
        self.callbacks = []
        if isinstance(callbacks, Callback):
            self.callbacks.append(callbacks)
        elif isinstance(callbacks, list):
            if all(isinstance(cb, Callback) for cb in callbacks) is True:
                self.callbacks = callbacks
            else:
                obj = [not isinstance(cb, Callback) for cb in callbacks][0]
                raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
        else:
            raise TypeError(f"Expect callbacks to be list or Callback. Got {type(callbacks)}.")

    def _check_callbacks_type(self):
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopCallback):
                raise ValueError("EarlyStopCallback is not effective when eval_dataset is None.")
            if isinstance(callback, BestModelCallback):
                raise ValueError("BestModelCallback is not effective when eval_dataset is None.")

    def _prepare_eval(self):
        if self.eval_dataset is not None and self.metrics is not None:
            self.evaluator = Evaluator(network=self.network, eval_dataset=self.eval_dataset, metrics=self.metrics,
                                        callbacks=self.callbacks)
        elif self.eval_dataset is None and self.metrics is None:
            if self.callbacks:
                self._check_callbacks_type()
            self.evaluator = None
        else:
            raise ValueError("For evaluation in training process, both eval dataset and metrics should be not None.")

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

    def run(self, tgt_columns=None, jit=False):
        """
        Training process entry.

        Args:
            tgt_columns (Optional[list[str], str]): Target label column names for loss function.
            jit (bool): Whether use Just-In-Time compile.

        """

        args_dict = vars(self)
        run_context = RunContext(args_dict)
        self.callback_manager.train_begin(run_context)
        self._run(run_context, tgt_columns, jit)
        self.callback_manager.train_end(run_context)

    def _run(self, run_context, tgt_columns=None, jit=False):
        """
        Training process for non-data sinking mode. The data would be passed to network directly.
        """
        # forward function
        net = self.network

        loss_fn = self.loss_fn
        optimizer = self.optimizer
        def forward_fn(inputs, labels):
            logits_list = ()
            logits = net(*inputs)
            if isinstance(logits, tuple):
                logits_list += logits
            else:
                logits_list += (logits,)

            loss = loss_fn(*logits_list, *labels)
            return_list = (loss,) + logits_list
            return return_list

        def forward_without_loss_fn(inputs, labels):
            loss_and_logits = net(*inputs, *labels)
            return loss_and_logits

        if self.loss_fn is None:
            grad_fn = value_and_grad(forward_without_loss_fn, None, optimizer.parameters, has_aux=True)
        else:
            grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        def _run_step(inputs, labels):
            """Core process of each step, including the forward propagation process and back propagation of data."""
            (loss, *_), grads = grad_fn(inputs, labels)
            optimizer(grads)
            return loss

        @ms_jit
        def _run_step_graph(inputs, labels):
            """Core process of each step, including the forward propagation process and back propagation of data."""
            (loss, _), grads = grad_fn(inputs, labels)
            loss = ops.depend(loss, optimizer(grads))
            return loss

        total = self.train_dataset.get_dataset_size()
        # train epoch begin
        for epoch in range(0, self.epochs):
            net.set_train()
            self.cur_epoch_nums = epoch + 1
            self.cur_step_nums = 0
            run_context.cur_epoch_nums = self.cur_epoch_nums
            run_context.cur_step_nums = 0
            if self.earlystop is True:
                break
            self.callback_manager.train_epoch_begin(run_context)
            with tqdm(total=total) as progress:
                progress.set_description(f'Epoch {epoch}')
                loss_total = 0
                # step begin
                for data in self.train_dataset.create_dict_iterator():
                    inputs, tgts = self._data_process(data, tgt_columns)
                    run_context.cur_step_nums += 1
                    self.cur_step_nums += 1
                    self.callback_manager.train_step_begin(run_context)
                    if jit:
                        loss = _run_step_graph(inputs, tgts)
                    else:
                        loss = _run_step(inputs, tgts)
                    loss_total += loss
                    progress.set_postfix(loss=loss_total/self.cur_step_nums)
                    progress.update(1)
                    # step end
                    self.callback_manager.train_step_end(run_context)
            # train epoch end
            progress.close()
            self.callback_manager.train_epoch_end(run_context)
            # do epoch evaluation
            if self.evaluator is not None:
                self._do_eval_epoch(run_context, tgt_columns, jit)

    def _run_ds_sink(self, train_dataset, eval_dataset, list_callback,
                     cb_params, print_steps, eval_steps):
        """Training process for data sinking mode."""
        raise NotImplementedError

    def _load_checkpoint(self, path):
        """Load checkpoint."""
        raise NotImplementedError

    def _save_checkpoint(self, path):
        """Save checkpoint."""
        raise NotImplementedError

    def _do_eval_steps(self, steps, eval_dataset):
        """Evaluate the model after n steps."""
        raise NotImplementedError

    def _do_eval_epoch(self, run_context, tgt_columns=None, jit=False):
        """Evaluate the model after an epoch."""
        self.callback_manager.evaluate_begin(run_context)
        self.evaluator.clear_metrics()
        metrics_result, metrics_names, metrics_values = self.evaluator._run(tgt_columns, jit)
        setattr(run_context, "metrics_values", metrics_values)
        setattr(run_context, "metrics_result", metrics_result)
        setattr(run_context, "metrics_names", metrics_names)
        self.callback_manager.evaluate_end(run_context)
        self.earlystop = run_context.earlystop

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
