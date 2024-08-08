# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
# pylint: disable=expression-not-assigned
# pylint: disable=assignment-from-no-return
# pylint: disable=used-before-assignment
"""Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""
import os
import re
import gc
import sys
import math
import time
import inspect
import shutil
import json
import copy
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import mindspore
from mindspore.dataset import Dataset, BatchDataset, PaddedBatchDataset

from mindnlp.core import nn, ops, optim
from ...core.serialization import safe_load_file, safe_save_file, save, save_checkpoint
from ...peft import PeftModel
from ...configs import WEIGHTS_NAME, CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME, \
    WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from ...dataset import BaseMapFunction
from ...utils import logging, find_labels, can_return_loss
from ...utils.import_utils import is_safetensors_available
from ...transformers.modeling_utils import PreTrainedModel
from ...transformers.configuration_utils import PretrainedConfig
from ...transformers.tokenization_utils_base import PreTrainedTokenizerBase
from ...transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from ...transformers.ms_utils import ALL_LAYERNORM_LAYERS

from ..utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    LabelSmoother,
    TrainOutput,
    PredictionOutput,
    EvalLoopOutput,
    find_batch_size,
    denumpify_detensorize,
    set_seed,
    enable_full_determinism,
    has_length,
    number_of_arguments,
    get_last_checkpoint,
    find_executable_batch_size,
    get_parameter_names,
    get_model_param_count,
    speed_metrics,
    nested_concat,
    nested_numpify,
    neftune_post_forward_hook,
    mismatch_dataset_col_names,
    get_function_args,
    args_only_in_map_fn,
    check_input_output_count
)
from ..train_args import TrainingArguments, OptimizerNames
from ..callbacks import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


logger = logging.get_logger(__name__)

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.ckpt"
SCHEDULER_NAME = "scheduler.json"
SCALER_NAME = "scaler.json"

def _is_peft_model(model):
    r"""
    Checks if the input model is an instance of the PeftModel class.
    
    Args:
        model (object): The model to be checked for being an instance of PeftModel class.
    
    Returns:
        None: Returns None if the input model is an instance of PeftModel class, otherwise returns False.
    
    Raises:
        None
    """
    classes_to_check = (PeftModel,)
    # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    return isinstance(model, classes_to_check)


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for MindSpore, optimized for 🤗 Transformers.
    """
    from ..utils import _get_learning_rate
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        map_fn: Optional[Union[Callable, BaseMapFunction]] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[mindspore.Tensor, mindspore.Tensor], mindspore.Tensor]] = None,
    ):
        """
        Initializes the Trainer class.
        
        Args:
            self (Trainer): The Trainer object itself.
            model (Union[PreTrainedModel, nn.Module]): The pre-trained model or neural network cell to be trained.
            args (TrainingArguments): The training arguments including hyperparameters and output directory.
            map_fn (Optional[Union[Callable, BaseMapFunction]]): Optional map function for data preprocessing.
            train_dataset (Optional[Dataset]): The training dataset.
            eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]]): The evaluation dataset.
            tokenizer (Optional[PreTrainedTokenizerBase]): The pre-trained tokenizer for tokenizing inputs.
            model_init (Optional[Callable[[], PreTrainedModel]]): Optional model initialization function.
            compute_metrics (Optional[Callable[[EvalPrediction], Dict]]): Optional function to compute evaluation metrics.
            callbacks (Optional[List[TrainerCallback]]): Optional list of trainer callbacks.
            optimizers (Tuple[nn.Optimizer, LearningRateSchedule]): Tuple of optimizer and learning rate scheduler.
            preprocess_logits_for_metrics (Optional[Callable[[mindspore.Tensor, mindspore.Tensor], mindspore.Tensor]]): Optional function to preprocess logits for metrics.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            RuntimeError: If `model` or `model_init` is not provided.
            ValueError: If the provided model cannot be used for training, or if there is an issue with the map function.
            ValueError: If `train_dataset` does not implement __len__ and `max_steps` is not specified.
            RuntimeError: If there is a conflict between `model_init` and `optimizers` arguments.
        """
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)

        self.args = args
        # Seed must be set before instantiating the model when using model
        # mindspore do not support full determinisim on 2.2
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.is_in_train = False

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                raise RuntimeError(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method"
                )
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        # if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
        #     self.is_model_parallel = True
        # else:
        self.is_model_parallel = False

        # TODO: support quantized model

        self.data_map_fn = map_fn
        if map_fn is not None and not (hasattr(map_fn, 'input_columns') and hasattr(map_fn, 'output_columns')) and \
            not check_input_output_count(map_fn):
            raise ValueError('`map_fn` must have same number of inputs and outputs when it is callable function'
                             ' without attributes `input_columns` and `output_columns`')

        self.train_dataset = copy.deepcopy(train_dataset)
        self.eval_dataset = copy.deepcopy(eval_dataset)
        self.tokenizer = tokenizer

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model = model
        self.model.set_train()

        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )

        default_callbacks = DEFAULT_CALLBACKS
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        self._signature_columns = None

        # Mixed precision setup
        self.use_amp = False

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged

        self.current_flos = 0
        self.hp_search_backend = None
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

    def _activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper:
        https://arxiv.org/abs/2310.05914
        """
        # TODO: support neftune
        unwrapped_model = model

        if _is_peft_model(unwrapped_model):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        del unwrapped_model

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model

    def _deactivate_neftune(self, model):
        """
        Deactivates the neftune method. Make sure to call `_activate_neftune` first.
        """
        # TODO: support neftune
        # if not hasattr(self, "neftune_hook_handle"):
        #     raise ValueError("Neftune is not activated make sure to call `trainer._activate_neftune()` first")

        unwrapped_model = model

        if _is_peft_model(unwrapped_model):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        self.neftune_hook_handle.remove()
        del embeddings.neftune_noise_alpha, unwrapped_model

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def _set_signature_columns_if_needed(self):
        r"""
        Method to set signature columns if they are not already set.
        
        Args:
            self (Trainer): The instance of the Trainer class.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            NotImplementedError: If the model does not have a 'forward' method.
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    # PeftMixedModel do not provide a `get_base_model` method
                    model_to_inspect = self.model.base_model.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())

            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "mindspore.dataset.Dataset", description: Optional[str] = None):
        r""" 
        Method _remove_unused_columns in the class Trainer removes unused columns from the input dataset if the corresponding argument is set to True.
        
        Args:
            self: The instance of the Trainer class.
            dataset (mindspore.dataset.Dataset): The input dataset from which the unused columns need to be removed.
            description (Optional[str]): An optional description of the dataset. Defaults to None.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None
        """
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.get_col_names()) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in dataset.get_col_names() if k not in ignored_columns]

        return dataset.project(columns)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for p in opt_model.parameters() if (p.name in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for p in opt_model.parameters() if (p.name not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """
        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        # TODO: support Adafactor
        # if args.optim == OptimizerNames.ADAFACTOR:
        #     optimizer_cls = Adafactor
        #     optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        # TODO: support AdamW huggingface version
        # elif args.optim == OptimizerNames.ADAMW_HF:
        #     from .optimization import AdamW

        #     optimizer_cls = AdamW
        #     optimizer_kwargs.update(adam_kwargs)
        if args.optim == OptimizerNames.ADAMW:
            optimizer_cls = optim.AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = optim.SGD
        # TODO: support Adagrad and Rmsporp
        # elif args.optim == OptimizerNames.ADAGRAD:
        #     optimizer_cls = mindspore.nn.Adagrad
        # elif args.optim == OptimizerNames.RMSPROP:
        #     optimizer_cls = mindspore.nn.RMSprop
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int, optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            from ...transformers.optimization import get_scheduler
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def num_examples(self, dataset: 'mindspore.dataset.Dataset') -> int:
        """
        Helper to get number of samples in a [`~mindspore.dataset.GeneratorDataset`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        return dataset.get_dataset_size() * dataset.get_batch_size()

    def num_tokens(self, train_ds: 'mindspore.dataset.Dataset', max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~mindspore.dataset.GeneratorDataset`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            for step, batch in train_ds.create_dict_iterator():
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens

    def call_model_init(self):
        r"""
        Method to call the model initialization function and validate its output.
        
        Args:
            self (Trainer): Instance of the Trainer class.
                This parameter represents the current instance of the Trainer class.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            RuntimeError: If the model_init method does not have 0 or 1 arguments.
                This exception is raised when the number of arguments in the model_init method is not 0 or 1.
            RuntimeError: If the model_init method returns None.
                This exception is raised when the model_init method returns None.
        """
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    @lru_cache(128)
    def get_train_dataset(self) -> Dataset:
        """
        Returns the training [`~mindspore.dataset.GeneratorDataset`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        map_fn = self.data_map_fn
        if isinstance(train_dataset, (BatchDataset, PaddedBatchDataset)):
            if self.data_map_fn is not None:
                logger.warning("The trainer has been passed a `map_fn` and found `BatchDataset` at same time, "
                               "the `map_fn` will be ignored.")
            return train_dataset

        if map_fn is not None:
            if mismatch_dataset_col_names(get_function_args(map_fn), train_dataset.get_col_names()):
                raise ValueError(f'The arguments of `map_fn` must be subset of useful dataset columns, '
                                f'but found {args_only_in_map_fn(get_function_args(map_fn), train_dataset.get_col_names())}')
            train_dataset = train_dataset.map(map_fn, map_fn.input_columns, map_fn.output_columns)

        train_dataset = self._remove_unused_columns(train_dataset, description="training")

        train_dataset = train_dataset.batch(self._train_batch_size, self.args.dataset_drop_last, self.args.dataset_num_workers)
        return train_dataset

    @lru_cache(128)
    def get_test_dataset(self, test_dataset: Dataset) -> Dataset:
        """
        Returns the test [`~mindspore.dataset.GeneratorDataset`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`mindspore.dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        # data_collator = self.data_collator
        if isinstance(test_dataset, (BatchDataset, PaddedBatchDataset)):
            return test_dataset
        test_dataset = self._remove_unused_columns(test_dataset, description="test")
        test_dataset = test_dataset.batch(self.args.eval_batch_size, self.args.dataset_drop_last, self.args.dataset_num_workers)

        # We use the same batch_size as for eval.
        return test_dataset

    @lru_cache(128)
    def get_eval_dataset(self, eval_dataset: Dataset = None) -> Dataset:
        """
        Returns the test [`~mindspore.dataset.GeneratorDataset`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`mindspore.dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        # data_collator = self.data_collator
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if isinstance(eval_dataset, (BatchDataset, PaddedBatchDataset)):
            return eval_dataset

        eval_dataset = self._remove_unused_columns(eval_dataset, description="test")
        eval_dataset = eval_dataset.batch(self.args.eval_batch_size, self.args.dataset_drop_last, self.args.dataset_num_workers)

        # We use the same batch_size as for eval.
        return eval_dataset

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._train_batch_size = self.args.train_batch_size
        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init()
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        self.model_reload = model_reloaded

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        if model_reloaded:
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )

        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _load_best_model(self):
        r"""Load the best model checkpoint.
        
        This method loads the best model checkpoint based on the provided state. The best model checkpoint is determined by the highest score achieved during training.
        
        Args:
            self: An instance of the Trainer class.
        
        Returns:
            None. The method only loads the best model checkpoint.
        
        Raises:
            None.
        
        Note:
            This method assumes that the best model checkpoint is saved in the specified checkpoint directory. If the best model checkpoint is not found, a warning message will be logged.
        
            If the model is a PEFT model, the method checks if the active adapter is available and loads the adapter model if it exists. Otherwise, a warning message is logged.
        
            If `save_safetensors` flag is enabled and the best safe model checkpoint is available, the method loads the safe model state dictionary. Otherwise, it loads the model state dictionary using
MindSpore's `load_checkpoint` function.
        
            If the best model checkpoint is not found, but the weights index file exists, the method attempts to load the sharded checkpoint using the `load_sharded_checkpoint` function.
        
            If the best model checkpoint is not found and `save_on_each_node` is not activated during distributed training, a warning message is logged.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model

        if (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True

            if _is_peft_model(model):
                # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                    if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                        model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
                        # Load_adapter has no return value present, modify it when appropriate.
                        load_result = [], []
                    else:
                        logger.warning(
                            "The intermediate checkpoints of PEFT may not be saved correctly, "
                            f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                            "Check some examples here: https://github.com/huggingface/peft/issues/96"
                        )
                        has_been_loaded = False
                else:
                    logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                    has_been_loaded = False
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                    state_dict = safe_load_file(best_safe_model_path)
                else:
                    state_dict = mindspore.load_checkpoint(
                        best_model_path,
                    )

                # If the model is on the GPU, it still works!
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
            if has_been_loaded:
                self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=False
            )
            self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _get_output_dir(self):
        r"""
        This method retrieves the output directory from the Trainer object.
        
        Args:
            self: Trainer object instance.
            
        Returns:
            None. Returns the output directory path specified in the Trainer object.
        
        Raises:
            This method does not raise any exceptions.
        """
        run_dir = self.args.output_dir
        return run_dir

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, ignore_keys_for_eval=None
    ):
        r"""
        Method _inner_training_loop in the class Trainer.
        
        Args:
            self (Trainer): The Trainer object.
            batch_size (int): The batch size for training.
            args (dict): Additional arguments for training configuration.
            resume_from_checkpoint (str): Path to a checkpoint to resume training from.
            ignore_keys_for_eval (list): List of keys to ignore during evaluation.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If args.max_steps must be set to a positive value if dataloader does not have a length.
            Warning: If there are no samples in the epoch_iterator during training.
            Exception: Any unexpected exceptions raised during the training process.
        """
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Dataset and number of training steps
        train_dataset = self.get_train_dataset()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataset = None
        num_train_tokens = None

        if has_length(train_dataset):
            len_dataset = len(train_dataset)
            num_update_steps_per_epoch = len_dataset // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataset)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataset, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataset) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataset) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataset, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        delay_optimizer_creation = False # do not delay now

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.recompute:
            if args.recompute_kwargs is None:
                recompute_kwargs = {}
            else:
                recompute_kwargs = args.recompute_kwargs

            self.model.recompute_enable(recompute_kwargs=recompute_kwargs)

        model = self.model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataset = train_dataset
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = mindspore.tensor(0.0)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataset
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataset is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                # self._load_rng_state(resume_from_checkpoint)
                pass

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                # epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator.create_dict_iterator()):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    # self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step, grads = self.training_step(model, inputs)
                if (
                    args.logging_nan_inf_filter
                    and (ops.isnan(tr_loss_step) or ops.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )
                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        _grad_norm = nn.utils.clip_grad_norm_(
                            grads,
                            args.max_grad_norm,
                        )
                    # Optimizer step
                    self.optimizer.step(grads)


                    optimizer_was_run = True
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self.log(metrics)

        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=self.args.output_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        r"""
        Loads the model from a checkpoint directory.
        
        Args:
            self (Trainer): The Trainer instance.
            resume_from_checkpoint (str): The path to the checkpoint directory.
            model (Optional[object]): The model to load. If not provided, the model specified in the Trainer instance will be used.
        
        Returns:
            None
        
        Raises:
            ValueError: If a valid checkpoint cannot be found at the given directory.
        
        '''
        
        The method `_load_from_checkpoint` is responsible for loading the model from a checkpoint directory. It takes three parameters: `self`, `resume_from_checkpoint`, and an optional parameter `model`. The
method does not return any value (`None`).
        
        - `self` (Trainer): The `Trainer` instance on which the method is called.
        - `resume_from_checkpoint` (str): The path to the checkpoint directory from which the model will be loaded.
        - `model` (Optional[object]): An optional parameter that specifies the model to be loaded. If not provided, the model specified in the `Trainer` instance will be used.
        
        The method first checks if a valid checkpoint can be found at the given `resume_from_checkpoint` directory. If no valid checkpoint is found, a `ValueError` is raised.
        
        If a valid checkpoint is found, the method proceeds to load the model. It first checks if a configuration file (`CONFIG_NAME`) is present in the checkpoint directory. If a configuration file is found,
it loads the configuration using the `PretrainedConfig.from_json_file` method.
        
        Next, the method checks if either the weights file (`weights_file`) or the safe weights file (`safe_weights_file`) is present in the checkpoint directory. If either of these files is found, the method
checks if the `save_safetensors` flag is enabled. If the flag is enabled and the safe weights file is present, it loads the model's state dictionary using the `safe_load_file` method. Otherwise, it uses the
`mindspore.load_checkpoint` method to load the model's state dictionary. The method then loads the state dictionary into the model using the `model.load_state_dict` method, with the `False` argument indicating
that strict loading should be disabled. After loading the state dictionary, any temporary variables are deleted and any warnings are issued using the `_issue_warnings_after_load` method.
        
        If neither the weights file nor the safe weights file is found, the method calls the `load_sharded_checkpoint` method to load the model from the checkpoint directory, with the `prefer_safe` parameter
indicating whether to prefer safe tensors.
        
        Note: The method assumes that the necessary imports and variables are already defined.
        """
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
            # If the model is on the GPU, it still works!
            # We load the model state dict on the CPU to avoid an OOM error.
            if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                state_dict = safe_load_file(safe_weights_file)
            else:
                state_dict = mindspore.load_checkpoint(
                    weights_file,
                )

            # workaround for FSDP bug
            # which takes *args instead of **kwargs
            load_result = model.load_state_dict(state_dict, False)
            # release memory
            del state_dict
            self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        # elif _is_peft_model(model):
        #     # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
        #     if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
        #         if os.path.exists(resume_from_checkpoint):
        #             model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
        #         else:
        #             logger.warning(
        #                 "The intermediate checkpoints of PEFT may not be saved correctly, "
        #                 f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
        #                 "Check some examples here: https://github.com/huggingface/peft/issues/96"
        #             )
        #     else:
        #         logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, prefer_safe=self.args.save_safetensors
            )
            self._issue_warnings_after_load(load_result)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        checkpoint_file_exists = (
            os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            mindspore.load_param_into_net(self.optimizer, mindspore.load_checkpoint(os.path.join(checkpoint, OPTIMIZER_NAME)))
            # with warnings.catch_warnings(record=True) as caught_warnings:
            with open(os.path.join(checkpoint, SCHEDULER_NAME), 'r') as fp:
                self.lr_scheduler.load_state_dict(json.load(fp))

            # reissue_pt_warnings(caught_warnings)

    def _prepare_inputs(self, inputs: Dict[str, Union[mindspore.Tensor, Any]]) -> Dict[str, Union[mindspore.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[mindspore.Tensor, Any]]) -> Tuple[List[mindspore.Tensor], mindspore.Tensor]:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[mindspore.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `mindspore.Tensor`: The tensor with training loss on this batch.
        """
        model.set_train()
        inputs = self._prepare_inputs(inputs)

        def forward(inputs):
            return self.compute_loss(model, inputs)

        if getattr(self, 'grad_fn', None) is None or self.model_reload:
            self.grad_fn = mindspore.value_and_grad(forward, None, tuple(model.parameters()))

        loss, grads = self.grad_fn(inputs)

        return loss / self.args.gradient_accumulation_steps, grads

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model = model
            if _is_peft_model(unwrapped_model):
                model_name = type(unwrapped_model.get_base_model()).__name__
            else:
                model_name = type(unwrapped_model).__name__
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        return self.args.process_index == 0

    def floating_point_ops(self, inputs: Dict[str, Union[mindspore.Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[mindspore.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def _issue_warnings_after_load(self, load_result):
        r"""Issues warnings after loading a checkpoint model.
        
        Args:
            self (Trainer): The current instance of the Trainer class.
            load_result (tuple): A tuple containing two lists. The first list represents the missing keys in the loaded checkpoint model, 
                while the second list represents the unexpected keys in the loaded checkpoint model.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        if len(load_result[0]) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result[0]) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result[0]}.")
        if len(load_result[1]) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result[1]}."
            )

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, epoch, ignore_keys_for_eval):
        r""" 
        This method '_maybe_log_save_evaluate' is a part of the 'Trainer' class. It takes 6 parameters:
        
        Args:
        - self (object): The instance of the Trainer class.
        - tr_loss (Tensor): The training loss value.
        - grad_norm (float or None): The gradient norm value, or None if not available.
        - model (object): The model being trained.
        - epoch (int): The current epoch number.
        - ignore_keys_for_eval (list or None): A list of keys to ignore during evaluation, or None if not applicable.
        
        Returns:
        None: This method does not return any value.
        
        Raises:
        - ValueError: If an invalid input is provided.
        - RuntimeError: If any runtime error occurs during execution.
        - KeyError: If a key error occurs while accessing dictionaries.
        """
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}

            # # all_gather + mean() to get average loss over all processes
            # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss_scalar = tr_loss.item()

            # reset tr_loss to zero
            ops.assign(tr_loss, 0)

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        metrics = None

        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save the model checkpoint to the specified output directory.
        
        Args:
            self (Trainer): The instance of the Trainer class.
            output_dir (Optional[str], optional): The directory path where the model checkpoint will be saved. If not provided, it defaults to self.args.output_dir. Defaults to None.
            state_dict (optional): The state dictionary of the model. Defaults to None.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the model is not an instance of PreTrainedModel.
            RuntimeError: If an error occurs during the saving process.
        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) #if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.parameters_dict()

            if isinstance(self.model, supported_classes):
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safe_save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "np"}
                    )
                else:
                    save_checkpoint(self.model, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # # Good practice: save your training arguments together with the trained model
        # mindspore.save_checkpoint(save_obj, ckpt_file_name, integrated_save=True, async_save=False, append_dict=None, enc_key=None, enc_mode='AES-GCM', choice_func=None, **kwargs)

    def _save_optimizer_and_scheduler(self, output_dir):
        r"""
        Save the optimizer and scheduler states to the specified output directory.
        
        Args:
            self (Trainer): The Trainer object instance.
            output_dir (str): The directory path where the optimizer and scheduler states will be saved.
            
        Returns:
            None. This method does not return any value.
        
        Raises:
            N/A
        """
        if self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        if self.args.should_save:
            # with warnings.catch_warnings(record=True) as caught_warnings:
            save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _save_checkpoint(self, model, metrics=None):
        r"""
        Save the model checkpoint along with relevant metrics and state information.
        
        Args:
            self (Trainer): The Trainer instance.
            model: The model to be saved.
            metrics (dict): A dictionary containing evaluation metrics. Defaults to None if metrics are not available.
            
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If the metric for the best model is not specified correctly.
            FileNotFoundError: If the specified output directory does not exist.
            OSError: If there are any issues with file operations while saving the checkpoint.
        """
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        # self.store_flos()

        run_dir = self._get_output_dir()
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            # self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        r"""
        Rotate the checkpoints to limit the total number of saved checkpoints.
        
        Args:
            self (Trainer): The instance of the Trainer class.
            use_mtime (bool, optional): If True, sorts the checkpoints based on modification time. Defaults to False.
            output_dir (str, optional): The directory where the checkpoints are stored. Defaults to None.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - OSError: If an error occurs while deleting the older checkpoints using shutil.rmtree.
            - TypeError: If the input parameters are of incorrect types.
        """
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    def store_flos(self):
        r"""
        Stores the current number of floating point operations (FLOs) and updates the total FLOs count.
        
        Args:
            self (Trainer): The Trainer object itself.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            N/A
        """
        # Storing the number of floating-point operations that went into the model
        # if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
        #     self.state.total_flos += (
        #         distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
        #     )
        #     self.current_flos = 0
        # else:
        self.state.total_flos += self.current_flos
        self.current_flos = 0

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        eval_dataset = self.get_eval_dataset(eval_dataset)

        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataset,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        # self._memory_tracker.start()

        test_dataloader = self.get_test_dataset(test_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        # if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        #     start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        dataset,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self.model
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=mindspore.float16)
            elif args.bf16_full_eval:
                model = model.to(dtype=mindspore.bfloat16)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataset):
            logger.info(f"  Num examples = {self.num_examples(dataset)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.set_train(False)

        self.callback_handler.eval_dataset = dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataset.create_dict_iterator()):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)

            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = loss.repeat(observed_batch_size)
                # losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            # if labels is not None:
            #     labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                # inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                # inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )

            if logits is not None:
                # logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                # logits = self.gather_function((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                # labels = self.gather_function((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(dataset):
            num_samples = len(dataset)
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        return tensors

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[mindspore.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[mindspore.Tensor], Optional[mindspore.Tensor], Optional[mindspore.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.cell`):
                The model to evaluate.
            inputs (`Dict[str, Union[mindspore.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[mindspore.Tensor], Optional[mindspore.Tensor], Optional[mindspore.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = tuple(inputs.get(name) for name in self.label_names)
            # labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        if has_labels or loss_without_labels:
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]
        else:
            loss = None
            outputs = model(**inputs)
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        # logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        r"""
        Method to retrieve and sort checkpoints based on specific criteria.
        
        Args:
            self: Trainer object, the instance invoking the method.
            output_dir (str, optional): Directory path to search for checkpoints. Defaults to None.
            checkpoint_prefix (str): Prefix for identifying checkpoint files.
            use_mtime (bool): Flag to indicate whether to use modification time for sorting.
        
        Returns:
            List[str]: A list of sorted checkpoint file paths.
        
        Raises:
            N/A
        """
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if (
            self.state.best_model_checkpoint is not None
            and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
        ):
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`mindspore.nn.cell`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and MindSpore save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, MindSpore files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.parameters_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    loader = safe_load_file if load_safe else mindspore.load_checkpoint

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return missing_keys, unexpected_keys
