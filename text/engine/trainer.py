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
Trainer for training.
"""
from mindspore import ms_function
from tqdm import tqdm

from text.engine.callbacks.callback_manager import CallbackManager, RunContext
from ..common.grad import value_and_grad


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
        epochs (int): Total number of iterations on the data. Default: 10.
        optimizer (Cell): Optimizer for updating the weights. If `optimizer` is None, the `network` needs to
            do backpropagation and update weights. Default value: None.
        loss (Cell): Objective function. If `loss_fn` is None, the `network` should contain the calculation of loss
            and parallel if needed. Default: None.
        metrics (Union[dict, set]): A Dictionary or a set of metrics for model evaluation.
            eg: {'accuracy', 'recall'}. Default: None.
        save_path (str): File path to save the model.
        device (str): List of devices used for training.
        callbacks (Optional[list[Callback], Callback]): List of callback objects which should be executed
            while training. Default: None.
        amp_level (str): Option for argument `level` in :func:`mindspore.build_train_network`, level for mixed
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
        print_steps (int): Control the amount of data in each sink. Default: -1.

    """

    def __init__(self, network, train_dataset, eval_dataset=None, epochs=10, batch_size=2, loss_fn=None,
                 optimizer=None, metrics=None, save_path=None, device=None, callbacks=None, amp_level='O0',
                 boost_level='O0', dataset_sink_mode=True, print_steps=-1):
        self.network = network
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.save_path = save_path
        self.device = device
        self.callbacks = callbacks
        self.amp_level = amp_level
        self.boost_level = boost_level
        self.dataset_sink_mode = dataset_sink_mode
        self.print_steps = print_steps
        self.cur_epoch_nums = 0
        self.cur_step_nums = 0

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
        """Training function entry."""
        args_dict = vars(self)
        run_context = RunContext(args_dict)
        self.callback_manager = CallbackManager(callbacks=self.callbacks)
        self.callback_manager.train_begin()
        self._run(mode, run_context)
        self.callback_manager.train_end()

    def _run(self, mode, run_context):
        """
        Training process for non-data sinking mode. The data would be passed to network directly.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        loss_fn = self.loss_fn
        network = self.network

        def net_forward(data, label):
            logits = network(data)
            loss = loss_fn(logits, label)
            return loss, logits

        self.grad_fn = value_and_grad(
            net_forward, self.optimizer.parameters, has_aux=True)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        total = self.train_dataset.get_dataset_size()
        for epoch in range(0, self.epochs):
            self.cur_epoch_nums = epoch + 1
            self.cur_step_nums = 0
            run_context.cur_epoch_nums = self.cur_epoch_nums
            run_context.cur_step_nums = 0
            self.callback_manager.train_epoch_begin()
            with tqdm(total=total) as t:
                t.set_description('Epoch %i' % epoch)
                loss_total = 0
                # self.callback_manager.fetch_data_begin()
                for data in self.train_dataset.create_tuple_iterator():
                    # self.callback_manager.fetch_data_end()
                    run_context.cur_step_nums += 1
                    self.cur_step_nums += 1
                    self.callback_manager.train_step_begin(run_context)
                    if mode == 'pynative':
                        loss = self._run_step(data)
                    elif mode == 'graph':
                        loss = ms_function(self._run_step)(data)
                    loss_total += loss
                    t.set_postfix(loss=loss_total/self.cur_step_nums)
                    t.update(1)
                    self.callback_manager.train_step_end(run_context)
            self.callback_manager.train_epoch_end(run_context)

    def _run_ds_sink(self, train_dataset, eval_dataset, list_callback,
                     cb_params, print_steps, eval_steps):
        """Training process for data sinking mode."""
        raise NotImplementedError

    def _run_step(self, batch_data):
        """Core process of each step, including the forward propagation process and back propagation of data."""
        data = batch_data[0]
        label = batch_data[1]
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss

    def load_checkpoint(self, path):
        """Load checkpoint."""
        raise NotImplementedError

    def save_checkpoint(self, path):
        """Save checkpoint."""
        raise NotImplementedError

    def eval_steps(self, steps, eval_dataset):
        """Evaluate the model after n steps."""
        raise NotImplementedError

    def eval_epoch(self, eval_dataset):
        """Evaluate the model after an epoch."""
        raise NotImplementedError
