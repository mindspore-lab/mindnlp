# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Callbacks to use with the Trainer class and customize the training loop.
"""
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from mindnlp.utils import logging
from .utils import IntervalStrategy, has_length
from .train_args import TrainingArguments


logger = logging.get_logger(__name__)


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.

    </Tip>

    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        logging_steps (`int`, *optional*, defaults to 500):
            Log every X updates steps
        eval_steps (`int`, *optional*):
            Run an evaluation every X steps.
        save_steps (`int`, *optional*, defaults to 500):
            Save checkpoint every X updates steps.
        train_batch_size (`int`, *optional*):
            The batch size for the training dataloader. Only needed when
            `auto_find_batch_size` has been used.
        num_input_tokens_seen (`int`, *optional*, defaults to 0):
            The number of tokens seen during training (number of input tokens, not the number of prediction tokens).
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (`List[Dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be `True` for one process).
        is_hyper_param_search (`bool`, *optional*, defaults to `False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
    """
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    train_batch_size: int = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str = None
    trial_params: Dict[str, Union[str, float, int, bool]] = None

    def __post_init__(self):
        r"""
        Method __post_init__ in the class TrainerState initializes the log_history attribute if it is None.
        
        Args:
            self: TrainerState object - The instance of the TrainerState class on which this method is called.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None
        """
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class TrainerControl:
    """
    A class that handles the [`Trainer`] control flow. This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    """
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    # no-format
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `mindspore.nn.cell`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data.
        optimizer (`mindspore.nn.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`mindspore.experimental.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataset (`mindspore.dataset.GeneratorDataset`, *optional*):
            The current dataloader used for training.
        eval_dataset (`mindspore.dataset.GeneratorDataset`, *optional*):
            The current dataloader used for training.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformers.PrinterCallback`].

    Example:

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```"""
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""
    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
        r"""
        Initializes a new instance of the CallbackHandler class.
        
        Args:
            self: The object instance.
            callbacks (list): A list of callback objects.
            model: The model object.
            tokenizer: The tokenizer object.
            optimizer: The optimizer object.
            lr_scheduler: The learning rate scheduler object.
        
        Returns:
            None
        
        Raises:
            None
        """
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = None
        self.eval_dataset = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        r"""
        Adds a callback to the list of callbacks in the CallbackHandler.
        
        Args:
            self (CallbackHandler): The instance of the CallbackHandler class.
            callback (callable or class): The callback to be added. It can be either a callable object or a class. If it is a class, an instance of it will be created.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        r"""
        pop_callback method in the CallbackHandler class removes a specified callback from the list of callbacks.
        
        Args:
            self (object): The instance of the CallbackHandler class.
            callback (object): The callback to be removed from the list. It can be either a function or a class type.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            TypeError: If the 'callback' parameter is not a valid type (function or class type).
        """
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        r"""
        Method to remove a callback from the list of callbacks in the CallbackHandler class.
        
        Args:
            self (CallbackHandler): The instance of the CallbackHandler class.
                It holds the list of callbacks from which the specified callback needs to be removed.
            callback (function or class): The callback to be removed from the list of callbacks.
                If the callback is a class type, then all callbacks of that specific class will be removed.
                If the callback is a function, that specific callback will be removed from the list.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            TypeError: If the callback is not a function or a class type.
            ValueError: If the specified callback is not found in the list of callbacks.
        """
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        r"""
        This method, callback_list, returns a string containing the names of the callback objects in the CallbackHandler.
        
        Args:
            self (CallbackHandler): The instance of the CallbackHandler class.
        
        Returns:
            str: A string containing the names of the callback objects separated by newline characters.
        
        Raises:
            None
        """
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        This method is called at the end of the initialization process.
        
        Args:
            self: The instance of the CallbackHandler class.
            args (TrainingArguments): An object containing training arguments.
            state (TrainerState): An object representing the state of the trainer.
            control (TrainerControl): An object providing control over the trainer.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        The 'on_train_begin' method is a callback function in the 'CallbackHandler' class. It is called at the beginning of the training process.
        
        Args:
            - self: The instance of the class. It is automatically passed and represents the current object.
            - args (TrainingArguments): An object containing various training arguments.
                - This parameter holds the training arguments that were passed to the Trainer.
                - It provides information such as the number of epochs, learning rate, batch size, etc.
            - state (TrainerState): An object representing the current state of the Trainer.
                - It encapsulates the training state, including information about the current training step, loss, and more.
            - control (TrainerControl): An object providing control over the training process.
                - It contains properties that can be modified to control the training behavior, such as stopping the training process.
        
        Returns:
            None
        
        Raises:
            This method does not raise any exceptions.
        """
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        This method is called at the end of the training process.
        
        Args:
            self: A reference to the current instance of the CallbackHandler class.
            args (TrainingArguments): An object containing training arguments.
                Purpose: Provides information about the training process configuration.
                Restrictions: Must be of type TrainingArguments.
            state (TrainerState): An object representing the state of the trainer during training.
                Purpose: Provides information about the current state of the trainer.
                Restrictions: Must be of type TrainerState.
            control (TrainerControl): An object providing control over the trainer during training.
                Purpose: Allows the callback to control the behavior of the trainer.
                Restrictions: Must be of type TrainerControl.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None
        """
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        CallbackHandler.on_epoch_begin method is called at the beginning of each epoch during training.
        
        Args:
        - self: The instance of the CallbackHandler class.
        - args (TrainingArguments): The training arguments containing configuration settings for the training process.
        - state (TrainerState): The current state of the trainer during training.
        - control (TrainerControl): An object that allows control over the training process.
        
        Returns:
        None. This method does not return any value.
        
        Raises:
        This method does not raise any exceptions.
        """
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        on_epoch_end method in CallbackHandler class.
        
        Args:
            self (CallbackHandler): The instance of the CallbackHandler class.
            args (TrainingArguments): The training arguments for the model.
            state (TrainerState): The state of the trainer during training.
            control (TrainerControl): The control object for the trainer.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        This method is called at the beginning of each training step.
        
        Args:
            self: CallbackHandler
                The instance of the CallbackHandler class invoking the method.
        
            args: TrainingArguments
                An object containing the training arguments for the current training session.
        
            state: TrainerState
                An object containing the current state of the Trainer.
        
            control: TrainerControl
                An object providing control over the behavior of the Trainer.
        
        Returns:
            None
            This method does not return any value.
        
        Raises:
            None
            This method does not raise any exceptions.
        """
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        CallbackHandler.on_substep_end(self, args, state, control)
        
        This method is called at the end of each substep of the training process.
        
        Args:
            self: An instance of the CallbackHandler class.
            args (TrainingArguments): An object that contains the training arguments.
                This parameter provides access to the training arguments such as 
                the number of epochs, learning rate, and batch size.
            state (TrainerState): An object that represents the current state of the trainer.
                This parameter provides information about the training progress, including
                the current step, epoch, and metrics.
            control (TrainerControl): An object that allows control over the training process.
                This parameter provides methods to pause, resume, or stop the training.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return self.call_event("on_substep_end", args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        on_step_end method in the CallbackHandler class.
        
        This method is called at the end of each training step in the Trainer class.
        
        Args:
            self (CallbackHandler): An instance of the CallbackHandler class.
            args (TrainingArguments): An object that contains the training arguments.
                This includes various parameters such as the number of epochs, batch size, learning rate, etc.
            state (TrainerState): An object that represents the current state of the Trainer.
                This includes information about the training progress, such as the current step and the loss.
            control (TrainerControl): An object that allows the CallbackHandler to control the training process.
                It provides methods to pause, resume, or stop the training.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        """
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        r"""
        This method 'on_evaluate' is a callback function in the 'CallbackHandler' class that is triggered during the evaluation phase of the training process.
        
        Args:
        - self: Represents the current instance of the class.
        - args (TrainingArguments): An object containing training arguments such as batch size, learning rate, etc.
        - state (TrainerState): Represents the current state of the Trainer during training.
        - control (TrainerControl): Provides control options for the Trainer, such as whether to continue evaluation.
        - metrics (dict): A dictionary containing evaluation metrics to be used during the evaluation process.
        
        Returns:
        - None: This method does not return any value explicitly. It sets 'control.should_evaluate' to False and triggers the 'on_evaluate' event using 'self.call_event'.
        
        Raises:
        - No specific exceptions are documented to be raised by this method. However, exceptions could potentially be raised within the 'call_event' method if any issues arise during the event triggering
process.
        """
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        r"""
        Method: on_predict
        
        Description:
        This method is called during the prediction phase of the training process. It is part of the CallbackHandler class and is responsible for handling the 'on_predict' event.
        
        Args:
        - self: The instance of the CallbackHandler class.
        - args (TrainingArguments): The arguments for training, containing various configuration settings and hyperparameters.
        - state (TrainerState): The current state of the Trainer, including the model, optimizer, and other training-related information.
        - control (TrainerControl): The control object that allows interaction with the Trainer during the training process.
        - metrics (dict): A dictionary containing the evaluation metrics calculated during the prediction phase.
        
        Returns:
        None.
        
        Raises:
        This method does not raise any exceptions.
        """
        return self.call_event("on_predict", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        r"""
        This method 'on_save' in the 'CallbackHandler' class is triggered when the model is saved during training.
        
        Args:
            self (CallbackHandler): The instance of the CallbackHandler class.
            args (TrainingArguments): An object containing training arguments.
                This parameter provides information such as hyperparameters and settings for the training process.
            state (TrainerState): An object representing the current state of the Trainer during training.
                It contains information about the model, optimizer, scheduler, and other training state variables.
            control (TrainerControl): An object that controls the behavior of the Trainer during training.
                It allows modifying the default behavior by setting flags like 'should_save' to control saving.
        
        Returns:
            None: This method does not return any value explicitly.
        
        Raises:
            This method does not raise any exceptions.
        """
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        r"""
        Method 'on_log' in the class 'CallbackHandler'.
        
        Args:
        - self: An instance of the class CallbackHandler.
        - args (TrainingArguments): Object containing training arguments.
        - state (TrainerState): Object representing the current state of the trainer.
        - control (TrainerControl): Object providing control options for the trainer.
        - logs: Additional logs or information related to the training process.
        
        Returns:
        None. This method does not return any value.
        
        Raises:
        - No specific exceptions are raised within this method.
        """
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        """
        Callback method called on each prediction step during training.
        
        Args:
            self (CallbackHandler): The instance of the CallbackHandler class.
            args (TrainingArguments): The training arguments for the current training session.
            state (TrainerState): The current state of the trainer during training.
            control (TrainerControl): The control object for the trainer.
        
        Returns:
            None. This method does not return any value explicitly.
        
        Raises:
            No specific exceptions are raised within this method.
        """
        return self.call_event("on_prediction_step", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        r"""
        call_event method in CallbackHandler class.
        
        This method is responsible for calling the specified event on each callback and handling the control flow based on the results.
        
        Args:
            self (object): The instance of the CallbackHandler class.
            event (str): The name of the event to be called on each callback.
            args (object): The arguments to be passed to the event callback.
            state (object): The current state of the system.
            control (object): The initial control parameter for the event handling.
        
        Returns:
            None. The method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
            However, the callbacks' event methods may raise exceptions depending on their implementation.
        """
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        r"""
        This method is called at the end of each training step in the 'DefaultFlowCallback' class.
        
        Args:
            self: An instance of the 'DefaultFlowCallback' class.
            args (TrainingArguments): An object containing training arguments.
                - args.logging_first_step (bool): Specifies whether to log the first step.
                - args.logging_strategy (IntervalStrategy): Specifies the logging strategy.
                - args.evaluation_strategy (IntervalStrategy): Specifies the evaluation strategy.
                - args.eval_delay (int): Specifies the delay in evaluation steps.
                - args.save_strategy (IntervalStrategy): Specifies the saving strategy.
            state (TrainerState): An object containing the current state of the trainer.
                - state.global_step (int): The current global step of the training.
                - state.logging_steps (int): The number of steps between each logging.
                - state.eval_steps (int): The number of steps between each evaluation.
                - state.save_steps (int): The number of steps between each saving.
                - state.max_steps (int): The maximum number of steps for the training.
            control (TrainerControl): An object controlling the behavior of the trainer.
                - control.should_log (bool): Specifies whether to log at the current step.
                - control.should_evaluate (bool): Specifies whether to evaluate at the current step.
                - control.should_save (bool): Specifies whether to save at the current step.
                - control.should_training_stop (bool): Specifies whether to stop training.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        r"""
        This method is called at the end of each epoch during the training process.
        
        Args:
            self (DefaultFlowCallback): The instance of the DefaultFlowCallback class.
            args (TrainingArguments): The training arguments containing various settings for training.
                This parameter is mandatory and must be of type TrainingArguments.
            state (TrainerState): The current state of the Trainer.
                This parameter is mandatory and must be of type TrainerState.
            control (TrainerControl): The control object that determines the actions to be taken during training.
                This parameter is mandatory and must be of type TrainerControl.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """
    def __init__(self):
        r"""
        Initializes a ProgressCallback object.
        
        Args:
            self: The ProgressCallback object itself.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No exceptions are raised by this method.
        """
        self.training_bar = None
        self.prediction_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        r"""
        This method is called at the beginning of the training process.
        
        Args:
            self (ProgressCallback): The instance of the ProgressCallback class invoking the method.
            args: Additional arguments passed to the method.
            state: The state of the training process.
            control: The control parameters for the training process.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        r"""
        Executes actions at the end of each step during training progress.
        
        Args:
            self (ProgressCallback): An instance of the ProgressCallback class.
            args: Additional arguments passed to the method.
            state: The state of the training process.
            control: Control parameters for the callback.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataset=None, **kwargs):
        r"""
        Method: on_prediction_step
        
        Description:
        This method is a callback function that is executed on each prediction step during model training. It updates the progress bar for prediction progress.
        
        Args:
        - self: The instance of the ProgressCallback class.
        - args: Additional arguments that may be passed to the method.
        - state: The current state of the training process.
          - Type: Any
          - Purpose: Provides information about the current state of the training process.
          - Restrictions: None
        - control: The control parameters for the training process.
          - Type: Any
          - Purpose: Provides control parameters for the training process.
          - Restrictions: None
        - eval_dataset: The evaluation dataset used for predictions.
          - Type: Any
          - Purpose: Stores the evaluation dataset used for predictions.
          - Restrictions: None
        
        Returns:
        - None: This method does not return any value.
        
        Raises:
        - None: This method does not raise any exceptions.
        """
        if state.is_world_process_zero and has_length(eval_dataset):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataset), leave=self.training_bar is None, dynamic_ncols=True
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        r"""
        This method 'on_evaluate' is defined in the 'ProgressCallback' class and is used to evaluate a given state based on certain conditions.
        
        Args:
        - self: Represents the instance of the class.
        - args: Additional arguments passed to the method.
        - state: Represents the current state being evaluated.
        - control: Additional control parameter.
        
        Returns:
        - None: This method does not return any value.
        
        Raises:
        - None.
        """
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        r"""
        Performs the prediction process and updates the progress bar accordingly.
        
        Args:
            self (ProgressCallback): An instance of the ProgressCallback class.
            args: Additional arguments passed to the method.
            state: The current state of the prediction process.
            control: The control object used to manage the prediction process.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        r"""
        Method 'on_log' in the class 'ProgressCallback' handles logging during training progress.
        
        Args:
        - self: The instance of the class.
        - args: Additional arguments passed to the method.
        - state: Represents the current state of the training process.
        - control: Control parameters for the logging behavior.
        - logs: A dictionary containing various log values. Default value is None.
        
        Returns:
        None: This method does not return any value.
        
        Raises:
        - None explicitly raised in this method.
        """
        if state.is_world_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        r"""
        This method 'on_train_end' is defined within the class 'ProgressCallback' and is called when the training process ends.
        
        Args:
        - self: Represents the instance of the class.
        - args: Additional arguments provided to the method.
        - state: Represents the current state of the training process.
        - control: Provides control options for the method.
        
        Returns:
        This method does not return any value, as it performs operations within the method itself and does not produce an output explicitly.
        
        Raises:
        This method does not explicitly raise any exceptions. However, it may raise exceptions indirectly depending on the operations performed within the method.
        """
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        r"""
        This method 'on_log' is defined within the class 'PrinterCallback' and is used to handle logging events.
        
        Args:
            self: Represents the instance of the class.
            args: A parameter that holds additional arguments.
            state: Represents the current state of the system.
            control: Indicates the control status.
            logs: A dictionary containing log information. Default is None.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            No specific exceptions are raised within this method.
        """
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


class EarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        r"""
        Initializes an instance of the EarlyStoppingCallback class.
        
        Args:
            self: The instance of the EarlyStoppingCallback class.
            early_stopping_patience (int, optional): The number of epochs to wait for improvement in the monitored metric before stopping the training process. Defaults to 1.
            early_stopping_threshold (float, optional): The minimum improvement required in the monitored metric to be considered as improvement. Defaults to 0.0.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        r"""
        This method 'check_metric_value' is part of the 'EarlyStoppingCallback' class and is used to evaluate a metric value and determine if early stopping criteria are met.
        
        Args:
        - self: Represents the instance of the class.
        - args: A set of arguments that influence the evaluation process.
            Type: Any.
            Purpose: Contains settings that affect the metric evaluation logic.
        - state: Represents the current state of the evaluation process.
            Type: Any.
            Purpose: Stores information about the best metric value encountered so far.
        - control: A parameter that controls the behavior of the evaluation process.
            Type: Any.
            Purpose: Allows for external control over the evaluation process.
        - metric_value: The value of the metric to be evaluated.
            Type: Any numerical type.
            Purpose: Represents the metric value to be checked against the current best metric.
        
        Returns:
        - None: This method does not return any value explicitly.
            Type: None.
            Purpose: The method updates internal state variables but does not provide any return value.
        
        Raises:
        - None: This method does not explicitly raise any exceptions. However, it may indirectly raise exceptions related to the operations performed within the method.
        """
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        r"""
        This method is called at the beginning of the training process within the 'EarlyStoppingCallback' class.
        
        Args:
            - self: The instance of the class.
            - args: A dictionary containing various settings for the training process.
                Type: dict
                Purpose: Contains configuration settings for the training process.
                Restrictions: None
            - state: Represents the current state of the training process.
                Type: any
                Purpose: Provides information about the current state of the training.
                Restrictions: None
            - control: A control variable to manage the training process.
                Type: any
                Purpose: Helps in controlling the flow of the training process.
                Restrictions: None
        
        Returns:
            - None: This method does not return any value.
                Type: None
                Purpose: N/A
        
        Raises:
            - AssertionError: Raised if 'load_best_model_at_end' is not set to True in the 'args' dictionary.
            - AssertionError: Raised if 'metric_for_best_model' is not defined in the 'args' dictionary.
            - AssertionError: Raised if 'evaluation_strategy' is set to IntervalStrategy.NO, as it should be steps or epoch.
        """
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        r"""
        This method 'on_evaluate' is a part of the 'EarlyStoppingCallback' class and is responsible for evaluating the metrics and performing early stopping if the specified metric value does not meet the
criteria.
        
        Args:
        - self: (object) The instance of the class.
        - args: (object) A collection of arguments containing the metric_for_best_model attribute, used to specify the metric for evaluating the model.
        - state: (object) The current state of the training process.
        - control: (object) A control object that manages the training process, including early stopping.
        - metrics: (object) A container for storing and retrieving the evaluated metrics during the training process.
        
        Returns:
        None: This method does not return any value.
        
        Raises:
        - Warning: If the specified metric_for_best_model is not found in the evaluated metrics, early stopping is disabled and a warning is logged.
        - Exception: If the early_stopping_patience_counter exceeds the early_stopping_patience, the training process is stopped by setting the should_training_stop flag in the control object to True.
        """
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
