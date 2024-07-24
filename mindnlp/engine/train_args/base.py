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
Training arguments for Trainer.
"""

import contextlib
# import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
# from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# from huggingface_hub import get_full_repo_name
# from packaging import version

from ..utils import (
    EvaluationStrategy,
    # HubStrategy,
    IntervalStrategy,
    SchedulerType,
)
from ...utils import (
    ExplicitEnum,
    # cached_property,
    is_safetensors_available,
    is_mindspore_available,
    logging,
    # requires_backends,
)
# from mindnlp.utils.generic import strtobool

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = {"passive": -1, **log_levels}

if is_mindspore_available():
    import mindspore


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    ADAMW = "adamw"
    SGD = "sgd"


# TODO: `TrainingArguments` users rely on it being fully mutable. In the future see if we can narrow this to a few keys: https://github.com/huggingface/transformers/pull/25903
@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.


    Parameters:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (`bool`, *optional*, defaults to `False`):
            If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
            points to a checkpoint directory.
        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `evaluation_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (`bool`, *optional*, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            <Tip warning={true}>

            When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
            evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.

            </Tip>

        eval_accumulation_steps (`int`, *optional*):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/NPU/TPU before being moved to the CPU (faster but
            requires more memory).
        eval_delay (`float`, *optional*):
            Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
            evaluation_strategy.
        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for [`AdamW`] optimizer.
        weight_decay (`float`, *optional*, defaults to 0):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            optimizer.
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The beta1 hyperparameter for the [`AdamW`] optimizer.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The beta2 hyperparameter for the [`AdamW`] optimizer.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            The epsilon hyperparameter for the [`AdamW`] optimizer.
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
            For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
            `max_steps` is reached.
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
        lr_scheduler_kwargs ('dict', *optional*, defaults to {}):
            The extra arguments for the lr_scheduler. See the documentation of each scheduler for possible values.
        warmup_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
        log_level (`str`, *optional*, defaults to `passive`):
            Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug',
            'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and keeps the
            current log level for the Transformers library (which will be `"warning"` by default).
        log_level_replica (`str`, *optional*, defaults to `"warning"`):
            Logger log level to use on replicas. Same choices as `log_level`"
        log_on_each_node (`bool`, *optional*, defaults to `True`):
            In multinode distributed training, whether to log using `log_level` once per node, or only on the main
            node.
        logging_dir (`str`, *optional*):
            [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
            *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:

                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.

        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log the first `global_step` or not.
        logging_steps (`int` or `float`, *optional*, defaults to 500):
            Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in
            range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        logging_nan_inf_filter (`bool`, *optional*, defaults to `True`):
            Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is `nan`
            or `inf` is filtered and the average loss of the current logging window is taken instead.

            <Tip>

            `logging_nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
            gradient is computed or applied to the model.

            </Tip>

        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.
        save_steps (`int` or `float`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a
            float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        save_total_limit (`int`, *optional*):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`. When `load_best_model_at_end` is enabled, the "best" checkpoint according to
            `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for
            `save_total_limit=5` and `load_best_model_at_end`, the four last checkpoints will always be retained
            alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two
            checkpoints are saved: the last one and the best one (if they are different).
        save_safetensors (`bool`, *optional*, defaults to `True`):
            Use [safetensors](https://huggingface.co/docs/safetensors) saving and loading for state dicts instead of
            default `mindspore.load_checkpoint` and `mindspore.save_checkpoint`.
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.

            This should not be activated when the different nodes use the same storage as the files will be saved with
            the same names for each node.
        save_only_model (`bool`, *optional*, defaults to `False`):
            When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.
            Note that when this is true, you won't be able to resume training from checkpoint.
            This enables you to save storage by not storing the optimizer, scheduler & rng state.
            You can only load the model using `from_pretrained` with this option set to `True`.
        use_cpu (`bool`, *optional*, defaults to `False`):
            Whether or not to use cpu. If set to False, we will use cuda or mps device if available.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
        data_seed (`int`, *optional*):
            Random seed to be used with data samplers. If not set, random generators for data sampling will use the
            same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model
            seed.
        jit_mode_eval (`bool`, *optional*, defaults to `False`):
            Whether or not to use MindSpore jit trace for inference.
        use_ipex (`bool`, *optional*, defaults to `False`):
            Use Intel extension for MindSpore when it is available.
        bf16 (`bool`, *optional*, defaults to `False`):
            Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher
            NVIDIA architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change.
        fp16 (`bool`, *optional*, defaults to `False`):
            Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        fp16_opt_level (`str`, *optional*, defaults to 'O1'):
            For `fp16` training, Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details on
            the [Apex documentation](https://nvidia.github.io/apex/amp).
        fp16_backend (`str`, *optional*, defaults to `"auto"`):
            This argument is deprecated. Use `half_precision_backend` instead.
        half_precision_backend (`str`, *optional*, defaults to `"auto"`):
            The backend to use for mixed precision training. Must be one of `"auto", "apex", "cpu_amp"`. `"auto"` will
            use CPU/CUDA AMP or APEX depending on the MindSpore version detected, while the other choices will force the
            requested backend.
        bf16_full_eval (`bool`, *optional*, defaults to `False`):
            Whether to use full bfloat16 evaluation instead of 32-bit. This will be faster and save memory but can harm
            metric values. This is an experimental API and it may change.
        fp16_full_eval (`bool`, *optional*, defaults to `False`):
            Whether to use full float16 evaluation instead of 32-bit. This will be faster and save memory but can harm
            metric values.
        local_rank (`int`, *optional*, defaults to -1):
            Rank of the process during distributed training.
        ddp_backend (`str`, *optional*):
            The backend to use for distributed training. Must be one of `"nccl"`, `"mpi"`, `"ccl"`, `"gloo"`, `"hccl"`.
        tpu_num_cores (`int`, *optional*):
            When training on TPU, the number of TPU cores (automatically passed by launcher script).
        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (`int` or `float`, *optional*):
            Number of update steps between two evaluations if `evaluation_strategy="steps"`. Will default to the same
            value as `logging_steps` if not set. Should be an integer or a float in range `[0,1)`. If smaller than 1,
            will be interpreted as ratio of total training steps.
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading (MindSpore only). 0 means that the data will be loaded in the
            main process.
        past_index (`int`, *optional*, defaults to -1):
            Some models like [TransformerXL](../model_doc/transformerxl) or [XLNet](../model_doc/xlnet) can make use of
            the past hidden states for their predictions. If this argument is set to a positive int, the `Trainer` will
            use the corresponding output (usually index 2) as the past state and feed it to the model at the next
            training step under the keyword argument `mems`.
        run_name (`str`, *optional*):
            A descriptor for the run. Typically used for [wandb](https://www.wandb.com/) and
            [mlflow](https://www.mlflow.org/) logging.
        disable_tqdm (`bool`, *optional*):
            Whether or not to disable the tqdm progress bars and table of metrics produced by
            [`~notebook.NotebookTrainingTracker`] in Jupyter Notebooks. Will default to `True` if the logging level is
            set to warn or lower (default), `False` otherwise.
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
            Whether or not to automatically remove the columns unused by the model forward method.
        label_names (`List[str]`, *optional*):
            The list of keys in your dictionary of inputs that correspond to the labels.

            Will eventually default to the list of argument names accepted by the model that contain the word "label",
            except if the model used is one of the `XxxForQuestionAnswering` in which case it will also include the
            `["start_positions", "end_positions"]` keys.
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Whether or not to load the best model found during training at the end of training. When this option is
            enabled, the best checkpoint will always be saved. See
            [`save_total_limit`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_total_limit)
            for more.

            <Tip>

            When set to `True`, the parameters `save_strategy` needs to be the same as `evaluation_strategy`, and in
            the case it is "steps", `save_steps` must be a round multiple of `eval_steps`.

            </Tip>

        metric_for_best_model (`str`, *optional*):
            Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`. Will
            default to `"loss"` if unspecified and `load_best_model_at_end=True` (to use the evaluation loss).

            If you set this value, `greater_is_better` will default to `True`. Don't forget to set it to `False` if
            your metric is better when lower.
        greater_is_better (`bool`, *optional*):
            Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models
            should have a greater metric or not. Will default to:

            - `True` if `metric_for_best_model` is set to a value that isn't `"loss"` or `"eval_loss"`.
            - `False` if `metric_for_best_model` is not set, or set to `"loss"` or `"eval_loss"`.
        ignore_data_skip (`bool`, *optional*, defaults to `False`):
            When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
            stage as in the previous training. If set to `True`, the training will begin faster (as that skipping step
            can take a long time) but will not yield the same results as the interrupted training would have.
        fsdp (`bool`, `str` or list of [`~trainer_utils.FSDPOption`], *optional*, defaults to `''`):
            Use MindSpore Distributed Parallel Training (in distributed training only).

            A list of options along the following:

            - `"full_shard"`: Shard parameters, gradients and optimizer states.
            - `"shard_grad_op"`: Shard optimizer states and gradients.
            - `"hybrid_shard"`: Apply `FULL_SHARD` within a node, and replicate parameters across nodes.
            - `"hybrid_shard_zero2"`: Apply `SHARD_GRAD_OP` within a node, and replicate parameters across nodes.
            - `"offload"`: Offload parameters and gradients to CPUs (only compatible with `"full_shard"` and
              `"shard_grad_op"`).
            - `"auto_wrap"`: Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.
        fsdp_config (`str` or `dict`, *optional*):
            Config to be used with fsdp (MindSpore Distributed Parallel Training). The value is either a location of
            fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`.

            A List of config and its options:
                - min_num_params (`int`, *optional*, defaults to `0`):
                    FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `fsdp` field is
                    passed).
                - transformer_layer_cls_to_wrap (`List[str]`, *optional*):
                    List of transformer layer class names (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`,
                    `T5Block` .... (useful only when `fsdp` flag is passed).
                - backward_prefetch (`str`, *optional*)
                    FSDP's backward prefetch mode. Controls when to prefetch next set of parameters (useful only when
                    `fsdp` field is passed).

                    A list of options along the following:

                    - `"backward_pre"` : Prefetches the next set of parameters before the current set of parameter's
                      gradient
                        computation.
                    - `"backward_post"` : This prefetches the next set of parameters after the current set of
                      parameter’s
                        gradient computation.
                - forward_prefetch (`bool`, *optional*, defaults to `False`)
                    FSDP's forward prefetch mode (useful only when `fsdp` field is passed).
                     If `"True"`, then FSDP explicitly prefetches the next upcoming all-gather while executing in the
                     forward pass.
                - limit_all_gathers (`bool`, *optional*, defaults to `False`)
                    FSDP's limit_all_gathers (useful only when `fsdp` field is passed).
                     If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight
                     all-gathers.
                - use_orig_params (`bool`, *optional*, defaults to `True`)
                    If `"True"`, allows non-uniform `requires_grad` during init, which means support for interspersed
                    frozen and trainable paramteres. Useful in cases such as parameter-efficient fine-tuning.
                - sync_module_states (`bool`, *optional*, defaults to `True`)
                    If `"True"`, each individually wrapped FSDP unit will broadcast module parameters from rank 0 to
                    ensure they are the same across all ranks after initialization
                - activation_checkpointing (`bool`, *optional*, defaults to `False`):
                    If `"True"`, activation checkpointing is a technique to reduce memory usage by clearing activations of
                    certain layers and recomputing them during a backward pass. Effectively, this trades extra
                    computation time for reduced memory usage.
                - xla (`bool`, *optional*, defaults to `False`):
                    Whether to use MindSpore/XLA Fully Sharded Data Parallel Training. This is an experimental feature
                    and its API may evolve in the future.
                - xla_fsdp_settings (`dict`, *optional*)
                    The value is a dictionary which stores the XLA FSDP wrapping parameters.
                - xla_fsdp_grad_ckpt (`bool`, *optional*, defaults to `False`):
                    Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be
                    used when the xla flag is set to true, and an auto wrapping policy is specified through
                    fsdp_min_num_params or fsdp_transformer_layer_cls_to_wrap.

        deepspeed (`str` or `dict`, *optional*):
            Use [Deepspeed](https://github.com/microsoft/deepspeed). This is an experimental feature and its API may
            evolve in the future. The value is either the location of DeepSpeed json config file (e.g.,
            `ds_config.json`) or an already loaded json file as a `dict`"

        accelerator_config (`str`, `dict`, or `AcceleratorConfig`, *optional*):
            Config to be used with the internal `Accelerator` implementation. The value is either a location of
            accelerator json config file (e.g., `accelerator_config.json`), an already loaded json file as `dict`,
            or an instance of [`~trainer_pt_utils.AcceleratorConfig`].

            A list of config and its options:
                - split_batches (`bool`, *optional*, defaults to `False`):
                    Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
                    `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
                    round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
                    in your script multiplied by the number of processes.
                - dispatch_batches (`bool`, *optional*):
                    If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
                    and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
                    underlying dataset is an `IterableDataset`, `False` otherwise.
                - even_batches (`bool`, *optional*, defaults to `True`):
                    If set to `True`, in cases where the total batch size across all processes does not exactly divide the
                    dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
                    all workers.
                - use_seedable_sampler (`bool`, *optional*, defaults to `True`):
                    Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`]). Ensures
                    training results are fully reproducable using a different sampling technique. While seed-to-seed results
                    may differ, on average the differences are neglible when using multiple different seeds to compare. Should
                    also be ran with [`~utils.set_seed`] for the best results.

        label_smoothing_factor (`float`, *optional*, defaults to 0.0):
            The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
            labels are changed from 0s and 1s to `label_smoothing_factor/num_labels` and `1 - label_smoothing_factor +
            label_smoothing_factor/num_labels` respectively.
        debug (`str` or list of [`~debug_utils.DebugOption`], *optional*, defaults to `""`):
            Enable one or more debug features. This is an experimental feature.

            Possible options are:

            - `"underflow_overflow"`: detects overflow in model's input/outputs and reports the last frames that led to
              the event
            - `"tpu_metrics_debug"`: print debug metrics on TPU

            The options should be separated by whitespaces.
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw"`):
            The optimizer to use: adamw, sgd.
        optim_args (`str`, *optional*):
            Optional arguments that are supplied to AnyPrecisionAdamW.
        group_by_length (`bool`, *optional*, defaults to `False`):
            Whether or not to group together samples of roughly the same length in the training dataset (to minimize
            padding applied and be more efficient). Only useful if applying dynamic padding.
        length_column_name (`str`, *optional*, defaults to `"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless `group_by_length` is `True` and the dataset is an
            instance of `Dataset`.
        report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
            The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
            `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"neptune"`,
            `"tensorboard"`, and `"wandb"`. Use `"all"` to report to all integrations installed, `"none"` for no
            integrations.
        ddp_find_unused_parameters (`bool`, *optional*):
            When using distributed training, the value of the flag `find_unused_parameters` passed to
            `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
        ddp_bucket_cap_mb (`int`, *optional*):
            When using distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`.
        ddp_broadcast_buffers (`bool`, *optional*):
            When using distributed training, the value of the flag `broadcast_buffers` passed to
            `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
        dataloader_persistent_workers (`bool`, *optional*, defaults to `False`):
            If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will
            increase RAM usage. Will default to `False`.
        dataloader_prefetch_factor (`int`, *optional*):
            Number of batches loaded in advance by each worker.
            2 means there will be a total of 2 * num_workers batches prefetched across all workers.
        skip_memory_metrics (`bool`, *optional*, defaults to `True`):
            Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows
            down the training and evaluation speed.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether or not to push the model to the Hub every time the model is saved. If this is activated,
            `output_dir` will begin a git directory synced with the repo (determined by `hub_model_id`) and the content
            will be pushed each time a save is triggered (depending on your `save_strategy`). Calling
            [`~Trainer.save_model`] will also trigger a push.

            <Tip warning={true}>

            If `output_dir` exists, it needs to be a local clone of the repository to which the [`Trainer`] will be
            pushed.

            </Tip>

        resume_from_checkpoint (`str`, *optional*):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`. Will default to `user_name/output_dir_name` with *output_dir_name* being the
            name of `output_dir`.

            Will default to the name of `output_dir`.
        hub_strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
            Defines the scope of what is pushed to the Hub and when. Possible values are:

            - `"end"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and a
              draft of a model card when the [`~Trainer.save_model`] method is called.
            - `"every_save"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and
              a draft of a model card each time there is a model save. The pushes are asynchronous to not block
              training, and in case the save are very frequent, a new push is only attempted if the previous one is
              finished. A last push is made with the final model at the end of training.
            - `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
              last-checkpoint, allowing you to resume training easily with
              `trainer.train(resume_from_checkpoint="last-checkpoint")`.
            - `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the output
              folder (so you will get one checkpoint folder per folder in your final repository)

        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `huggingface-cli login`.
        hub_private_repo (`bool`, *optional*, defaults to `False`):
            If True, the Hub repo will be set to private.
        hub_always_push (`bool`, *optional*, defaults to `False`):
            Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not finished.
        recompute (`bool`, *optional*, defaults to `False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        recompute_kwargs (`dict`, *optional*, defaults to `None`):
            Key word arguments to be passed to the `recompute_enable` method.
        include_inputs_for_metrics (`bool`, *optional*, defaults to `False`):
            Whether or not the inputs will be passed to the `compute_metrics` function. This is intended for metrics
            that need inputs, predictions and references for scoring calculation in Metric class.
        auto_find_batch_size (`bool`, *optional*, defaults to `False`)
            Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding
            CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
        full_determinism (`bool`, *optional*, defaults to `False`)
            If `True`, [`enable_full_determinism`] is called instead of [`set_seed`] to ensure reproducible results in
            distributed training. Important: this will negatively impact the performance, so only use it for debugging.
        ray_scope (`str`, *optional*, defaults to `"last"`):
            The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray will
            then use the last checkpoint of all trials, compare those, and select the best one. However, other options
            are also available. See the [Ray documentation](
            https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis.get_best_trial) for
            more options.
        ddp_timeout (`int`, *optional*, defaults to 1800):
            The timeout for `mindspore.communication.init` calls, used to avoid GPU socket timeouts when
            performing slow operations in distributed runnings.
        use_mps_device (`bool`, *optional*, defaults to `False`):
            This argument is deprecated.`mps` device will be used if it is available similar to `cuda` device.
        nn_compile (`bool`, *optional*, defaults to `False`):
            This will use the best defaults for the [`mindspore.nn.Module.compile`]
            You can customize the defaults with the argument `nn_compile_backend` and `nn_compile_mode` but we
            don't guarantee any of them will work as the support is progressively rolled in in MindSpore.

            This flag and the whole compile API is experimental and subject to change in future releases.
        nn_compile_backend (`str`, *optional*):
            The backend to use in `mindspore.nn.Module.compile`. If set to any value, `mindspore.nn.Module.compile` will be set to `True`.

            Refer to the MindSpore doc for possible values and note that they may change across MindSpore versions.

            This flag is experimental and subject to change in future releases.
        nn_compile_mode (`str`, *optional*):
            The mode to use in `mindspore.nn.Module.compile`. If set to any value, `mindspore.nn.Module.compile` will be set to `True`.

            Refer to the MindSpore doc for possible values and note that they may change across MindSpore versions.

            This flag is experimental and subject to change in future releases.
        split_batches (`bool`, *optional*):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices
            during distributed training. If

            set to `True`, the actual batch size used will be the same on any kind of distributed processes, but it
            must be a

            round multiple of the number of processes you are using (such as GPUs).
        include_tokens_per_second (`bool`, *optional*):
            Whether or not to compute the number of tokens per second per device for training speed metrics.

            This will iterate over the entire training dataloader once beforehand,

            and will slow down the entire process.

        include_num_input_tokens_seen (`bool`, *optional*):
            Whether or not to track the number of input tokens seen throughout training.

            May be slower in distributed training as gather operations must be called.

        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This can drastically improve model performance
            for instruction fine-tuning. Check out the [original paper](https://arxiv.org/abs/2310.05914) and the
            [original code](https://github.com/neelsjain/NEFTune). Support transformers `PreTrainedModel` and also
            `PeftModel` from peft.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs: Optional[Dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts"
            )
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: Optional[str] = field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    save_safetensors: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of default mindspore.load_checkpoint and mindspore.save_checkpoint."
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": " Whether or not to use cpu. If set to False, we will use cuda/tpu/mps/npu device if available."
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use MindSpore jit trace for inference"}
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    ddp_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "The backend to be used for distributed training",
            "choices": ["nccl", "gloo", "mpi", "ccl", "hccl"],
        },
    )

    dataset_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    dataset_num_workers: int = field(
        default=1,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (MindSpore only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    dataset_prefetch_factor: int = field(
        default=None,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
                "Default is unset"
            )
        },
    )
    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )

    default_optim = "adamw"
    optim: Union[OptimizerNames, str] = field(
        default=default_optim,
        metadata={"help": "The optimizer to use."},
    )
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    recompute: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    recompute_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `mindspore.nn.Module.recompute` through `model.recompute_enable`."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )

    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"},
    )

    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                " a CUDA Out-of-Memory was reached"
            )
        },
    )
    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )

    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Overrides the default timeout for distributed training (value should be given in seconds)."
        },
    )

    include_tokens_per_second: Optional[bool] = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )

    include_num_input_tokens_seen: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set to `True`, will track the number of input tokens seen throughout training. (May be slower in distributed training)"
        },
    )

    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "Activates neftune noise embeddings into the model."
                    "NEFTune has been proven to drastically improve model performances for instrcution fine-tuning. "
                    "Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune. "
                    "Only supported for `PreTrainedModel` and `PeftModel` classes."
        },
    )

    def __post_init__(self):
        r"""
        This method initializes the TrainingArguments class instance after its creation.
        
        Args:
            self: An instance of the TrainingArguments class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            - ValueError: If the evaluation strategy requires non-zero evaluation steps or logging steps are zero.
            - FutureWarning: If using `EvaluationStrategy` for `evaluation_strategy` is deprecated.
            - ValueError: If the logging strategy requires non-zero logging steps or steps are not an integer.
            - ValueError: If the saving steps are not an integer when required.
            - ValueError: If `load_best_model_at_end` is enabled but save and evaluation strategies do not match.
            - ValueError: If the saving steps are not a multiple of evaluation steps for `load_best_model_at_end`.
            - ValueError: If `save_safetensors` is enabled but safetensors are not installed.
            - ValueError: If both `fp16` and `bf16` are set to True.
            - ValueError: If both `fp16_full_eval` and `bf16_full_eval` are set to True.
            - ValueError: If lr_scheduler_type is reduce_lr_on_plateau but eval strategy or mindspore is not available.
            - ValueError: If warmup_ratio is not in the range [0,1] or if both warmup_ratio and warmup_steps are provided.
            - ValueError: If dataset_prefetch_factor is set without dataset_num_workers > 1.
        """
        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5"
                " of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.evaluation_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == IntervalStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps "
                            f"{self.save_steps} and eval_steps {self.eval_steps}."
                        )
                    # Work around floating point precision issues
                    LARGE_MULTIPLIER = 1_000_000
                    if (self.save_steps * LARGE_MULTIPLIER) % (self.eval_steps * LARGE_MULTIPLIER) != 0:
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            f"steps, but found {self.save_steps}, which is not a multiple of {self.eval_steps}."
                        )
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        safetensors_available = is_safetensors_available()
        if self.save_safetensors and not safetensors_available:
            raise ValueError(f"--save_safetensors={self.save_safetensors} requires safetensors to be installed!")
        if not self.save_safetensors and safetensors_available:
            logger.info(
                f"Found safetensors installation, but --save_safetensors={self.save_safetensors}. "
                f"Safetensors should be a preferred weights saving format due to security and performance reasons. "
                f"If your model cannot be saved by safetensors please feel free to open an issue at "
                f"https://github.com/huggingface/safetensors!"
            )

        if (
            self.load_best_model_at_end or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.evaluation_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires an eval strategy")
            if not is_mindspore_available():
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires mindspore")

        self.optim = OptimizerNames(self.optim)

        # if training args is specified, it will override the one specified in the accelerate config
        if self.fp16:
            mixed_precision_dtype = "fp16"
        elif self.bf16:
            mixed_precision_dtype = "bf16"

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training"
            )

        if self.dataset_num_workers == 0 and self.dataset_prefetch_factor is not None:
            raise ValueError(
                "--dataset_prefetch_factor can only be set when data is loaded in a different process, i.e."
                " when --dataset_num_workers > 1."
            )

    def __str__(self):
        r"""
        This method returns a string representation of the TrainingArguments object.
        
        Args:
            self (TrainingArguments): The instance of the TrainingArguments class.
            
        Returns:
            None: This method returns a string representation of the TrainingArguments object.
        
        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def n_device(self):
        r"""
        Returns the number of devices used for training.
        
        Args:
            self (TrainingArguments): The object instance.
            
        Returns:
            None: This method does not return a value.
            
        Raises:
            None: This method does not raise any exceptions.
        """
        return 1

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        per_device_batch_size = self.per_device_train_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_device)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        per_device_batch_size = self.per_device_eval_batch_size
        eval_batch_size = per_device_batch_size * max(1, self.n_device)
        return eval_batch_size

    @property
    def ddp_timeout_delta(self) -> timedelta:
        """
        The actual timeout for mindspore.communication.init since it expects a timedelta variable.
        """
        return timedelta(seconds=self.ddp_timeout)

    @property
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `nn.DistributedDataParallel`).
        - `ParallelMode.TPU`: several TPU cores.
        """
    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        return 1

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        if self.log_on_each_node:
            return self.local_process_index == 0
        else:
            return self.process_index == 0

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            return self.process_index == 0

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
        anything) unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """
        # convert to int
        log_level = trainer_log_levels[self.log_level]
        log_level_replica = trainer_log_levels[self.log_level_replica]

        log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
        log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
        return log_level_main_node if self.should_log else log_level_replica_node

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for MindSpore distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
        if is_mindspore_available() and self.world_size > 1:
            main_process_desc = "main local process" if local else "main process"
            is_main_process = mindspore.communication.get_rank() == 0

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")

                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
        else:
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_mindspore_available():
            valid_types.append(mindspore.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    # The following methods are there to simplify the instantiation of `TrainingArguments`
    def set_training(
        self,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        weight_decay: float = 0,
        num_epochs: float = 3,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        recompute: bool = False,
    ):
        """
        A method that regroups all basic arguments linked to the training.

        <Tip>

        Calling this method will automatically set `self.do_train` to `True`.

        </Tip>

        Args:
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate for the optimizer.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for training.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the
                optimizer.
            num_train_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            gradient_accumulation_steps (`int`, *optional*, defaults to 1):
                Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

                <Tip warning={true}>

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training
                examples.

                </Tip>

            seed (`int`, *optional*, defaults to 42):
                Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use
                the [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized
                parameters.
            recompute (`bool`, *optional*, defaults to `False`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_training(learning_rate=1e-4, batch_size=32)
        >>> args.learning_rate
        1e-4
        ```
        """
        self.do_train = True
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_train_epochs = num_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.recompute = recompute
        return self

    def set_evaluate(
        self,
        strategy: Union[str, IntervalStrategy] = "no",
        steps: int = 500,
        batch_size: int = 8,
        accumulation_steps: Optional[int] = None,
        delay: Optional[float] = None,
        loss_only: bool = False,
        jit_mode: bool = False,
    ):
        """
        A method that regroups all arguments linked to evaluation.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
                The evaluation strategy to adopt during training. Possible values are:

                    - `"no"`: No evaluation is done during training.
                    - `"steps"`: Evaluation is done (and logged) every `steps`.
                    - `"epoch"`: Evaluation is done at the end of each epoch.

                Setting a `strategy` different from `"no"` will set `self.do_eval` to `True`.
            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two evaluations if `strategy="steps"`.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for evaluation.
            accumulation_steps (`int`, *optional*):
                Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU.
                If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster
                but requires more memory).
            delay (`float`, *optional*):
                Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
                evaluation_strategy.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
            jit_mode (`bool`, *optional*):
                Whether or not to use MindSpore jit trace for inference.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_evaluate(strategy="steps", steps=100)
        >>> args.eval_steps
        100
        ```
        """
        self.evaluation_strategy = IntervalStrategy(strategy)
        if self.evaluation_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.do_eval = self.evaluation_strategy != IntervalStrategy.NO
        self.eval_steps = steps
        self.per_device_eval_batch_size = batch_size
        self.eval_accumulation_steps = accumulation_steps
        self.eval_delay = delay
        self.prediction_loss_only = loss_only
        self.jit_mode_eval = jit_mode
        return self

    def set_testing(
        self,
        batch_size: int = 8,
        loss_only: bool = False,
        jit_mode: bool = False,
    ):
        """
        A method that regroups all basic arguments linked to testing on a held-out dataset.

        <Tip>

        Calling this method will automatically set `self.do_predict` to `True`.

        </Tip>

        Args:
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for testing.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.
            jit_mode (`bool`, *optional*):
                Whether or not to use MindSpore jit trace for inference.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_testing(batch_size=32)
        >>> args.per_device_eval_batch_size
        32
        ```
        """
        self.do_predict = True
        self.per_device_eval_batch_size = batch_size
        self.prediction_loss_only = loss_only
        self.jit_mode_eval = jit_mode
        return self

    def set_save(
        self,
        strategy: Union[str, IntervalStrategy] = "steps",
        steps: int = 500,
        total_limit: Optional[int] = None,
        on_each_node: bool = False,
    ):
        """
        A method that regroups all arguments linked to checkpoint saving.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The checkpoint save strategy to adopt during training. Possible values are:

                    - `"no"`: No save is done during training.
                    - `"epoch"`: Save is done at the end of each epoch.
                    - `"steps"`: Save is done every `save_steps`.

            steps (`int`, *optional*, defaults to 500):
                Number of updates steps before two checkpoint saves if `strategy="steps"`.
            total_limit (`int`, *optional*):
                If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
                `output_dir`.
            on_each_node (`bool`, *optional*, defaults to `False`):
                When doing multi-node distributed training, whether to save models and checkpoints on each node, or
                only on the main one.

                This should not be activated when the different nodes use the same storage as the files will be saved
                with the same names for each node.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_save(strategy="steps", steps=100)
        >>> args.save_steps
        100
        ```
        """
        self.save_strategy = IntervalStrategy(strategy)
        if self.save_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.save_steps = steps
        self.save_total_limit = total_limit
        self.save_on_each_node = on_each_node
        return self

    def set_logging(
        self,
        strategy: Union[str, IntervalStrategy] = "steps",
        steps: int = 500,
        report_to: Union[str, List[str]] = "none",
        level: str = "passive",
        first_step: bool = False,
        nan_inf_filter: bool = False,
        on_each_node: bool = False,
        replica_level: str = "passive",
    ):
        """
        A method that regroups all arguments linked to logging.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The logging strategy to adopt during training. Possible values are:

                    - `"no"`: No logging is done during training.
                    - `"epoch"`: Logging is done at the end of each epoch.
                    - `"steps"`: Logging is done every `logging_steps`.

            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two logs if `strategy="steps"`.
            level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on the main process. Possible choices are the log levels as strings: `"debug"`,
                `"info"`, `"warning"`, `"error"` and `"critical"`, plus a `"passive"` level which doesn't set anything
                and lets the application set the level.
            report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
                The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
                `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`,
                `"neptune"`, `"tensorboard"`, and `"wandb"`. Use `"all"` to report to all integrations installed,
                `"none"` for no integrations.
            first_step (`bool`, *optional*, defaults to `False`):
                Whether to log and evaluate the first `global_step` or not.
            nan_inf_filter (`bool`, *optional*, defaults to `True`):
                Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is
                `nan` or `inf` is filtered and the average loss of the current logging window is taken instead.

                <Tip>

                `nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
                gradient is computed or applied to the model.

                </Tip>

            on_each_node (`bool`, *optional*, defaults to `True`):
                In multinode distributed training, whether to log using `log_level` once per node, or only on the main
                node.
            replica_level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on replicas. Same choices as `log_level`

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_logging(strategy="steps", steps=100)
        >>> args.logging_steps
        100
        ```
        """
        self.logging_strategy = IntervalStrategy(strategy)
        if self.logging_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.logging_steps = steps
        self.report_to = report_to
        self.log_level = level
        self.logging_first_step = first_step
        self.logging_nan_inf_filter = nan_inf_filter
        self.log_on_each_node = on_each_node
        self.log_level_replica = replica_level
        return self

    def set_optimizer(
        self,
        name: Union[str, OptimizerNames] = "adamw_torch",
        learning_rate: float = 5e-5,
        weight_decay: float = 0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        args: Optional[str] = None,
    ):
        """
        A method that regroups all arguments linked to the optimizer and its hyperparameters.

        Args:
            name (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw"`):
                The optimizer to use: `"adamw"`, `"sgd"`.
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
            beta1 (`float`, *optional*, defaults to 0.9):
                The beta1 hyperparameter for the adam optimizer or its variants.
            beta2 (`float`, *optional*, defaults to 0.999):
                The beta2 hyperparameter for the adam optimizer or its variants.
            epsilon (`float`, *optional*, defaults to 1e-8):
                The epsilon hyperparameter for the adam optimizer or its variants.
            args (`str`, *optional*):
                Optional arguments that are supplied to AnyPrecisionAdamW (only useful when
                `optim="adamw_anyprecision"`).

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_optimizer(name="adamw", beta1=0.8)
        >>> args.optim
        'adamw'
        ```
        """
        self.optim = OptimizerNames(name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = beta1
        self.adam_beta2 = beta2
        self.adam_epsilon = epsilon
        self.optim_args = args
        return self

    def set_lr_scheduler(
        self,
        name: Union[str, SchedulerType] = "linear",
        num_epochs: float = 3.0,
        max_steps: int = -1,
        warmup_ratio: float = 0,
        warmup_steps: int = 0,
    ):
        """
        A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

        Args:
            name (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
                The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
            num_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            warmup_ratio (`float`, *optional*, defaults to 0.0):
                Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
            warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of
                `warmup_ratio`.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
        >>> args.warmup_ratio
        0.05
        ```
        """
        self.lr_scheduler_type = SchedulerType(name)
        self.num_train_epochs = num_epochs
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        return self

    def set_dataloader(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        auto_find_batch_size: bool = False,
        ignore_data_skip: bool = False,
        sampler_seed: Optional[int] = None,
    ):
        """
        A method that regroups all arguments linked to the dataloaders creation.

        Args:
            drop_last (`bool`, *optional*, defaults to `False`):
                Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch
                size) or not.
            num_workers (`int`, *optional*, defaults to 0):
                Number of subprocesses to use for data loading (MindSpore only). 0 means that the data will be loaded in
                the main process.
            pin_memory (`bool`, *optional*, defaults to `True`):
                Whether you want to pin memory in data loaders or not. Will default to `True`.
            persistent_workers (`bool`, *optional*, defaults to `False`):
                If True, the data loader will not shut down the worker processes after a dataset has been consumed
                once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training,
                but will increase RAM usage. Will default to `False`.
            prefetch_factor (`int`, *optional*):
                Number of batches loaded in advance by each worker.
                2 means there will be a total of 2 * num_workers batches prefetched across all workers.
            auto_find_batch_size (`bool`, *optional*, defaults to `False`)
                Whether to find a batch size that will fit into memory automatically through exponential decay,
                avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
            ignore_data_skip (`bool`, *optional*, defaults to `False`):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True`, the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have.
            sampler_seed (`int`, *optional*):
                Random seed to be used with data samplers. If not set, random generators for data sampling will use the
                same seed as `self.seed`. This can be used to ensure reproducibility of data sampling, independent of
                the model seed.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
        >>> args.per_device_train_batch_size
        16
        ```
        """
        self.per_device_train_batch_size = train_batch_size
        self.per_device_eval_batch_size = eval_batch_size
        self.dataset_drop_last = drop_last
        self.dataset_num_workers = num_workers
        self.dataset_persistent_workers = persistent_workers
        self.dataset_prefetch_factor = prefetch_factor
        self.auto_find_batch_size = auto_find_batch_size
        self.ignore_data_skip = ignore_data_skip
        self.data_seed = sampler_seed
        return self


class ParallelMode(Enum):

    r"""
    Represents the different modes of parallel processing supported by the system.
    
    This class defines an enumeration for the various modes of parallel processing that can be utilized by the system. It inherits from the Enum class, providing a structured way to define and work with
parallel processing modes within the system.
    """
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
    SAGEMAKER_MODEL_PARALLEL = "sagemaker_model_parallel"
    SAGEMAKER_DATA_PARALLEL = "sagemaker_data_parallel"
    TPU = "tpu"
