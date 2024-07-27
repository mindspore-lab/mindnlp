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
import os
from typing import Optional, List, Union
from inspect import signature
from tqdm.autonotebook import tqdm
from mindspore import nn, Tensor, context, mutable
from mindspore import save_checkpoint
from mindspore.context import K_CONTEXT
from mindspore.dataset.engine import Dataset, TakeDataset

from mindnlp._legacy.abc import Callback, Metric
from mindnlp.transformers.configuration_utils import PretrainedConfig
from mindnlp.utils import logging
from ..callbacks.callback_manager import CallbackManager, RunContext
from ..callbacks.earlystop_callback import EarlyStopCallback
from ..callbacks.best_model_callback import BestModelCallback
from ..evaluator import Evaluator
from ...amp import auto_mixed_precision, StaticLossScaler, NoLossScaler
from ..trainer.utils import get_default_forward_fn_with_loss_fn, \
    get_default_forward_fn_without_loss_fn, get_default_train_step_fn

logger = logging.get_logger(__name__)

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
        metrics (Optional[list[Metrics], Metrics]): List of metrics objects which should be used
            while evaluating. Default:None.
        epochs (int): Total number of iterations on the data. Default: 10.
        optimizer (Cell): Optimizer for updating the weights. If `optimizer` is None, the `network` needs to
            do backpropagation and update weights. Default value: None.
        loss_fn (Cell): Objective function. If `loss_fn` is None, the `network` should contain the calculation of loss
            and parallel if needed. Default: None.
        callbacks (Optional[list[Callback], Callback]): List of callback objects which should be executed
            while training. Default: None.
        jit (bool): Whether use Just-In-Time compile.

    """
    def __init__(self,
                 network,
                 args=None,
                 loss_fn: Optional[nn.Module] = None,
                 optimizer: Optional[nn.Module] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 metrics: Optional[Metric] = None,
                 callbacks: Optional[Union[Callback, List]] = None,
                 **kwargs):
        r"""
        Initializes an instance of the 'Trainer' class.
        
        Args:
            self: The instance of the 'Trainer' class.
            network: An object of type 'nn.Module' representing the neural network model.
            args: An optional argument of type 'None' or any other type. It is used for additional configuration purposes.
            loss_fn: An optional argument of type 'nn.Module' representing the loss function used for training the model.
            optimizer: An optional argument of type 'nn.Module' representing the optimizer used for updating the model's parameters.
            train_dataset: An optional argument of type 'Dataset' representing the training dataset.
            eval_dataset: An optional argument of type 'Dataset' representing the evaluation dataset.
            metrics: An optional argument of type 'Metric' representing the evaluation metrics used for monitoring the model's performance.
            callbacks: An optional argument of type 'Union[Callback, List]' representing the callbacks used during training.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If 'jit' is set to 'True' and the 'network' does not support static graph via jit compile.
            Warning: If the 'train_dataset' is of type 'TakeDataset', which may slow down the training speed and require recompiling the network.
        
        Note:
            The method initializes various attributes and sets the initial values for training-related parameters.
            It also prepares the callbacks, evaluation settings, and callback manager for the 'Trainer' instance.
        """
        self.args = args
        epochs = kwargs.pop('epochs', None)
        jit = kwargs.pop('jit', False)
        check_gradients = kwargs.pop('check_gradients', False)

        if jit and 'MS' not in str(network.__class__.__name__):
            raise ValueError(f'{network.__class__.__name__} do not support static graph via jit compile, '
                             f'please check the supported model list and use MS{network.__class__.__name__} instead.')
        # deprecated args
        self.jit = jit
        self.check_gradients = check_gradients

        if isinstance(train_dataset, TakeDataset):
            logger.warning("The `train_dataset` is split after the 'batch' operation, "
                        "which will slow down the training speed and recompile the neural network"
                        "please split it first, and then use 'map' operation.")

        # model components
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.forward_fn = None
        self.train_fn = None

        if loss_fn is None:
            self.obj_network = True
        else:
            self.obj_network = False

        # dataset
        self.train_dataset = train_dataset
        self.epochs = epochs

        # amp settings
        self.amp_level = 'O0'
        self.loss_scaler = NoLossScaler()

        self.cur_epoch_nums = 0
        self.cur_step_nums = 0
        self.earlystop = False
        if callbacks:
            callbacks = self._prepare_callbacks(callbacks)
        self._prepare_eval(eval_dataset, metrics, callbacks, jit)

        self.callback_manager = CallbackManager(callbacks)

    def _prepare_train_func(self):
        r"""
        Prepares the training function for the Trainer class.
        
        Args:
            self (Trainer): An instance of the Trainer class.
        
        Returns:
            None
        
        Raises:
            None
        
        This method initializes the training function for the Trainer class. It sets the forward function and the train function if they are not already set. The forward function is responsible for performing
the forward pass of the neural network and calculating the loss. The train function is responsible for executing a single training step, which includes the forward pass, backpropagation, and optimizer update.
        
        If the forward function is None, it is determined based on whether the Trainer has an objective network or not. If an objective network is present, the default forward function without a loss function
is used. Otherwise, the default forward function with a loss function and loss scaler is used.
        
        If the train function is None, the default train step function is used. The default train step function utilizes the forward function, the optimizer, the loss scaler, and other optional parameters such
as gradient checking and automatic mixed precision (AMP) level.
        
        Note:
            The forward function, train function, and other parameters are initialized during the Trainer's initialization process.
        """
        if self.forward_fn is None:
            self.forward_fn = get_default_forward_fn_with_loss_fn(self.network, self.loss_fn, self.loss_scaler) \
                if not self.obj_network else get_default_forward_fn_without_loss_fn(self.network, self.loss_scaler)

        if self.train_fn is None:
            self.train_fn = get_default_train_step_fn(self.forward_fn, self.optimizer, self.loss_scaler,
                                                      self.check_gradients or self.amp_level != 'O0', self.jit, self.obj_network)

    def _prepare_callbacks(self, callbacks):
        r"""
        Prepare the list of callbacks for the Trainer.
        
        Args:
            self (Trainer): The instance of the Trainer class.
            callbacks (Callback or list): The callbacks to be prepared. 
                Can be a single Callback instance or a list of Callback instances.
                
        Returns:
            None. This method modifies the internal state of the Trainer instance.
        
        Raises:
            TypeError: If the callbacks parameter is not of type Callback or list.
            TypeError: If the elements in the list of callbacks are not instances of Callback.
        """
        if isinstance(callbacks, Callback):
            return [callbacks]
        if isinstance(callbacks, list):
            if all(isinstance(cb, Callback) for cb in callbacks) is True:
                return callbacks
            obj = [not isinstance(cb, Callback) for cb in callbacks][0]
            raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
        raise TypeError(f"Expect callbacks to be list or Callback. Got {type(callbacks)}.")

    def _check_callbacks_type(self, callbacks):
        r"""
        This method '_check_callbacks_type' in the class 'Trainer' is used to verify the types of callbacks provided.
        
        Args:
            self: The instance of the Trainer class. It is used to access the attributes and methods of the class instance.
            
            callbacks: A list of callback objects to be checked for their types. Each callback object in the list is iterated over to ensure it meets specific criteria.
            
        Returns:
            None: This method does not return any value. It performs the type checking of the callbacks and raises exceptions if necessary.
        
        Raises:
            ValueError: This exception is raised if any of the callback objects in the 'callbacks' list is an instance of EarlyStopCallback or BestModelCallback. 
            If an EarlyStopCallback instance is found, the error message 'EarlyStopCallback is not effective when eval_dataset is None.' is raised.
            If a BestModelCallback instance is found, the error message 'BestModelCallback is not effective when eval_dataset is None.' is raised.
        """
        for callback in callbacks:
            if isinstance(callback, EarlyStopCallback):
                raise ValueError("EarlyStopCallback is not effective when eval_dataset is None.")
            if isinstance(callback, BestModelCallback):
                raise ValueError("BestModelCallback is not effective when eval_dataset is None.")

    def _prepare_eval(self, eval_dataset, metrics, callbacks, jit):
        r"""
        Prepares the evaluation process during training in the 'Trainer' class.
        
        Args:
            self (Trainer): The instance of the 'Trainer' class.
            eval_dataset (Dataset): The dataset to be used for evaluation during training.
            metrics (list): The list of metrics to be used for evaluation.
            callbacks (list): The list of callbacks to be used during evaluation.
            jit (bool): Flag indicating whether just-in-time (JIT) compilation should be applied.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If both 'eval_dataset' and 'metrics' parameters are None.
                This exception is raised when both 'eval_dataset' and 'metrics' are None,
                and the 'callbacks' parameter is not empty.
        
        This method prepares the evaluation process during training by creating an 'Evaluator' instance
        and assigning it to the 'evaluator' attribute of the 'Trainer' class instance. The 'Evaluator' instance
        is created with the specified 'network', 'eval_dataset', 'metrics', 'callbacks', and 'jit' parameters.
        If both 'eval_dataset' and 'metrics' parameters are None, the 'callbacks' parameter is checked to ensure
        it is of the correct type. If 'callbacks' is not empty, the 'Trainer' instance's 'evaluator' attribute is set to None.
        """
        if eval_dataset is not None and metrics is not None:
            self.evaluator = Evaluator(network=self.network, eval_dataset=eval_dataset, metrics=metrics,
                                       callbacks=callbacks, jit=jit)
        elif eval_dataset is None and metrics is None:
            if callbacks:
                self._check_callbacks_type(callbacks)
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

    def run(self, tgt_columns=None):
        """
        Training process entry.

        Args:
            tgt_columns (Optional[list[str], str]): Target label column names for loss function.

        """
        self._prepare_train_func()

        args_dict = vars(self)
        run_context = RunContext(args_dict)
        self.callback_manager.train_begin(run_context)

        self._run(run_context, tgt_columns)
        self.callback_manager.train_end(run_context)

    def _run(self, run_context, tgt_columns=None):
        """
        Training process for non-data sinking mode. The data would be passed to network directly.
        """
        # set mindspore mode to GRAPH_MODE, since jit mode with
        # control flow will slow down the training speed.
        if self.jit:
            context.set_context(mode=context.GRAPH_MODE)
        else:
            os.environ['MS_DEV_FORCE_ACL'] = '1'
            K_CONTEXT.set_backend_policy('ge')
            K_CONTEXT.set_backend_policy('ms')

        total = self.train_dataset.get_dataset_size()
        # train epoch begin
        for epoch in range(0, self.epochs):
            self.network.set_train()
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
                    data, tgts = self._data_process(data, tgt_columns)
                    run_context.cur_step_nums += 1
                    self.cur_step_nums += 1
                    self.callback_manager.train_step_begin(run_context)
                    if self.obj_network:
                        loss = self.train_fn(**data)
                    else:
                        loss = self.train_fn(tgts, **data)
                    loss_total += loss
                    run_context.loss = loss_total/self.cur_step_nums
                    progress.set_postfix(loss=loss_total/self.cur_step_nums)
                    progress.update(1)
                    # step end
                    self.callback_manager.train_step_end(run_context)
            # train epoch end
            progress.close()
            self.callback_manager.train_epoch_end(run_context)
            # do epoch evaluation
            if self.evaluator is not None:
                self._do_eval_epoch(run_context, tgt_columns)

        # restore PYNATIVE_MODE after training.
        if self.jit:
            context.set_context(mode=context.PYNATIVE_MODE)
            os.environ['MS_DEV_FORCE_ACL'] = '1'
            K_CONTEXT.set_backend_policy('ge')
            K_CONTEXT.set_backend_policy('ms')

    def _run_ds_sink(self, train_dataset, eval_dataset, list_callback,
                     cb_params, print_steps, eval_steps):
        """Training process for data sinking mode."""
        raise NotImplementedError

    def _load_checkpoint(self, path):
        """Load checkpoint."""
        raise NotImplementedError

    def _do_eval_steps(self, steps, eval_dataset):
        """Evaluate the model after n steps."""
        raise NotImplementedError

    def _do_eval_epoch(self, run_context, tgt_columns=None):
        """Evaluate the model after an epoch."""
        self.callback_manager.evaluate_begin(run_context)
        self.evaluator.clear_metrics()
        metrics_result, metrics_names, metrics_values = self.evaluator._run(tgt_columns)
        setattr(run_context, "metrics_values", metrics_values)
        setattr(run_context, "metrics_result", metrics_result)
        setattr(run_context, "metrics_names", metrics_names)
        self.callback_manager.evaluate_end(run_context)
        self.earlystop = run_context.earlystop

    def _data_process(self, data, tgt_columns):
        """Process data match the network forward"""
        # prepare input dataset.
        sig = signature(self.network.forward)
        net_args = sig.parameters

        inputs = {}
        used_col = set()
        for arg in net_args:
            if arg in ('self', 'kwargs'):
                continue
            if arg not in data.keys():
                if str(net_args[arg])[-4:] != 'None':
                    raise ValueError(f'network inputs need `{arg}`, but not found in dataset columns.')
            else:
                inputs[arg] = data.get(arg)
                used_col.add(arg)

        if self.obj_network:
            return inputs, None

        # process target dataset.
        tgts = ()
        tgt_columns = self._prepare_tgt_columns(tgt_columns)
        for tgt_column in tgt_columns:
            if tgt_column in data:
                tgts += (data.get(tgt_column),)
                used_col.add(tgt_column)
            else:
                raise ValueError(f'Not found `{tgt_column}` in dataset, please check dataset column names.')

        remain_data_keys = set(data.keys()) - used_col

        if remain_data_keys:
            logger.warning(f'{remain_data_keys} is not match inputs arguments of network or function.')

        return inputs, mutable(tgts)

    def _prepare_tgt_columns(self, tgt_columns):
        """Check and prepare target columns for training."""
        out_columns = []
        if tgt_columns is None:
            logger.warning("In the process of training model, tgt_column can not be None.")
            return []
        if not isinstance(tgt_columns, (str, list)):
            raise TypeError(f"Expect tgt_columns to be list or str. Got {type(tgt_columns)}.")

        if isinstance(tgt_columns, str):
            tgt_columns = [tgt_columns]

        for col in tgt_columns:
            if isinstance(col, str):
                out_columns.append(col)
            else:
                raise TypeError(f"Expect str of tgt_column. Got {type(col)}")

        return out_columns

    def add_callback(self):
        """add callback"""
    def remove_callback(self, name_or_type):
        """remove callback"""
    def set_forward_fn(self, forward_fn):
        """set forward function"""
        self.forward_fn = forward_fn

    def set_step_fn(self, step_fn):
        """set step function"""
        self.train_fn = step_fn

    def set_amp(self, level='O1', loss_scaler=None):
        """set amp"""
        self.amp_level = level
        self.network = auto_mixed_precision(self.network, level)
        if loss_scaler is None:
            logger.warning("Trainer will use 'StaticLossScaler' with `scale_value=2 ** 10` when `loss_scaler` is None.")
            self.loss_scaler = StaticLossScaler(2 ** 10)
        else:
            self.loss_scaler = loss_scaler

    def set_optimizer(self, optimizer):
        """set optimizer"""
        self.optimizer = optimizer

    def train(self, target_columns):
        """train"""
        return self.run(target_columns)

    def train_step(self, inputs):
        """train step"""
        if isinstance(inputs, Tensor):
            inputs = (inputs,)
        return self.train_fn(*inputs)

    def train_loop(self, train_dataset):
        """train loop"""
    def evaluate(self):
        """evalute"""
    def evaluate_loop(self):
        """evaluate loop"""
    def predict(self, test_dataset):
        """predict"""
    def predict_step(self, inputs, return_loss_only=False):
        """predict step"""
    def predict_loop(self):
        """predict loop"""
    def save_model(self, output_dir, model_name=None):
        """save model to specify dir."""
        assert output_dir, "`output_dir` is None, please input a real path."
        if hasattr(self.network, 'config') and isinstance(self.network.config, PretrainedConfig):
            self.network.config.to_file(output_dir)
        if model_name:
            model_path = os.path.join(output_dir, f'{model_name}.ckpt')
        else:
            model_path = os.path.join(output_dir, 'model.ckpt')
        save_checkpoint(self.network, model_path)
