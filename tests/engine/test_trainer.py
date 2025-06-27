# coding=utf-8
# Copyright 2018 the HuggingFace Inc. team.
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

import dataclasses
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch
import pytest

import numpy as np
from parameterized import parameterized
from requests.exceptions import HTTPError

import mindnlp
from mindnlp.core.nn import Parameter
from mindnlp.engine import (
    IntervalStrategy,
    TrainerCallback,
    TrainingArguments
)
from mindnlp.transformers import (
    AutoTokenizer,
    PretrainedConfig,
)
from mindnlp.common.optimization import get_polynomial_decay_schedule_with_warmup
from mindnlp.utils import is_mindspore_available, logging
from mindnlp.core.serialization import safe_load_file, safe_save_file, load_checkpoint
from mindnlp.utils.testing_utils import (
    # ENDPOINT_STAGING,
    # TOKEN,
    # USER,
    CaptureLogger,
    LoggingLevel,
    TestCasePlus,
    execute_subprocess_async,
    get_tests_dir,
    is_staging_test,
    require_safetensors,
    require_sentencepiece,
    require_tokenizers,
    require_mindspore,
    slow,
)
from mindnlp.transformers.tokenization_utils_base import PreTrainedTokenizerBase
from mindnlp.engine.utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from mindnlp.engine.train_args import OptimizerNames
from mindnlp.utils import (
    is_safetensors_available,
)
from mindnlp.configs import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

if is_mindspore_available():
    import mindspore
    from mindnlp.core import nn, ops, optim
    from mindnlp.core.serialization import load_checkpoint, save_checkpoint
    from mindnlp.core.nn import functional as F
    from mindspore.dataset import GeneratorDataset

    # import transformers.optimization
    from mindnlp.engine.callbacks import (
        EarlyStoppingCallback
    )
    from mindnlp.transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        # GlueDataset,
        # GlueDataTrainingArguments,
        GPT2Config,
        GPT2LMHeadModel,
        # LineByLineTextDataset,
        PreTrainedModel,
    )
    from mindnlp.engine import Trainer, TrainerState
    # from mindnlp.transformers.modeling_utils import unwrap_model

PATH_SAMPLE_TEXT = f"{get_tests_dir()}/fixtures/sample_text.txt"


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = [self.x[i]]
        for y in self.ys:
            result.append(y[i])
        return tuple(result)

@dataclasses.dataclass
class RegressionTrainingArguments(TrainingArguments):
    a: float = 0.0
    b: float = 0.0
    keep_report_to: bool = False

    def __post_init__(self):
        super().__post_init__()
        # save resources not dealing with reporting unless specified (also avoids the warning when it's not set)
        # can be explicitly disabled via `keep_report_to`
        if not self.keep_report_to:
            self.report_to = []


class RepeatDataset:
    def __init__(self, x, length=64):
        self.x = x
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.x, self.x


class DynamicShapesDataset:
    def __init__(self, length=64, seed=42, batch_size=8):
        self.length = length
        np.random.seed(seed)
        sizes = np.random.randint(1, 20, (length // batch_size,))
        # For easy batching, we make every batch_size consecutive samples the same size.
        self.xs = [np.random.normal(size=(s,)).astype(np.float32) for s in sizes.repeat(batch_size)]
        self.ys = [np.random.normal(size=(s,)).astype(np.float32) for s in sizes.repeat(batch_size)]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.xs[i], self.ys[i]


class Almost:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


class RegressionModelConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_ms=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_ms = random_ms
        self.hidden_size = 1


if is_mindspore_available():

    class SampleIterableDataset:
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            self.dataset = RegressionDataset(a=a, b=b, length=length, seed=seed, label_names=label_names)

        def __iter__(self):
            for i in range(len(self.dataset)):
                yield self.dataset[i]

    class FiniteIterableDataset(SampleIterableDataset):
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            super().__init__(a, b, length, seed, label_names)
            self.current_sample = 0

        def __iter__(self):
            while self.current_sample < len(self.dataset):
                yield self.dataset[self.current_sample]
                self.current_sample += 1

    class MultiLoader:
        def __init__(self, loaders):
            self.loaders = loaders

        def __len__(self):
            return sum(len(loader) for loader in self.loaders)

        def __iter__(self):
            for loader in self.loaders:
                yield from loader

    class RegressionModel(nn.Module):
        def __init__(self, a=0, b=0, double_output=False):
            super().__init__()
            self.a = Parameter(mindspore.tensor([a]).float())
            self.b = Parameter(mindspore.tensor([b]).float())
            self.double_output = double_output
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = F.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionDictModel(nn.Module):
        def __init__(self, a=0, b=0):
            super().__init__()
            self.a = Parameter(mindspore.tensor([a]).float())
            self.b = Parameter(mindspore.tensor([b]).float())
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            result = {"output": y}
            if labels is not None:
                result["loss"] = F.mse_loss(y, labels)
            return result

    class RegressionPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = Parameter(mindspore.tensor([config.a]).float())
            self.b = Parameter(mindspore.tensor([config.b]).float())
            self.double_output = config.double_output

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = F.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionRandomPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = Parameter(mindspore.tensor([config.a]).float())
            self.b = Parameter(mindspore.tensor([config.b]).float())
            self.random_ms = config.random_ms

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if self.random_ms:
                torch_rand = ops.randn(1).squeeze()
            np_rand = np.random.rand()
            rand_rand = random.random()

            if self.random_ms:
                y += 0.05 * torch_rand
            y += 0.05 * mindspore.tensor(np_rand + rand_rand)

            if labels is None:
                return (y,)
            loss = F.mse_loss(y, labels)
            return (loss, y)

    class TstLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.bias = Parameter(ops.zeros(hidden_size))

        def forward(self, x):
            h = self.ln1(F.relu(self.linear1(x)))
            h = F.relu(self.linear2(x))
            return self.ln2(x + h + self.bias)

    def get_regression_trainer(
        a=0, b=0, double_output=False, train_len=64, eval_len=64, pretrained=True, keep_report_to=False, **kwargs
    ):
        label_names = kwargs.get("label_names", None)

        column_names = ['input_x']
        if label_names is None:
            column_names.append('labels')
        else:
            column_names.extend(label_names)
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        print(column_names)
        train_dataset = GeneratorDataset(RegressionDataset(length=train_len, label_names=label_names), column_names=column_names)
        eval_dataset = GeneratorDataset(RegressionDataset(length=eval_len, label_names=label_names), column_names=column_names, shuffle=False)

        model_init = kwargs.pop("model_init", None)
        if model_init is not None:
            model = None
        else:
            if pretrained:
                config = RegressionModelConfig(a=a, b=b, double_output=double_output)
                # We infer the correct model class if one uses gradient_checkpointing or not
                target_cls = (
                    RegressionPreTrainedModel
                )
                model = target_cls(config)
            else:
                model = RegressionModel(a=a, b=b, double_output=double_output)

        compute_metrics = kwargs.pop("compute_metrics", None)
        optimizers = kwargs.pop("optimizers", (None, None))
        output_dir = kwargs.pop("output_dir", "./regression")
        preprocess_logits_for_metrics = kwargs.pop("preprocess_logits_for_metrics", None)

        args = RegressionTrainingArguments(output_dir, a=a, b=b, keep_report_to=keep_report_to, **kwargs)
        return Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            model_init=model_init,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


class TrainerIntegrationCommon:
    def check_saved_checkpoints(self, output_dir, freq, total, is_pretrained=True, safe_weights=True):
        weights_file = WEIGHTS_NAME if not safe_weights else SAFE_WEIGHTS_NAME
        file_list = [weights_file]#, "training_args.bin", "optimizer.ckpt", "scheduler.json", "trainer_state.json"]
        if is_pretrained:
            file_list.append("config.json")
        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint))
            for filename in file_list:
                self.assertTrue(os.path.isfile(os.path.join(checkpoint, filename)))

    def check_best_model_has_been_loaded(
        self, output_dir, freq, total, trainer, metric, greater_is_better=False, is_pretrained=True, safe_weights=True
    ):
        checkpoint = os.path.join(output_dir, f"checkpoint-{(total // freq) * freq}")
        log_history = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json")).log_history

        values = [d[metric] for d in log_history]
        best_value = max(values) if greater_is_better else min(values)
        best_checkpoint = (values.index(best_value) + 1) * freq
        checkpoint = os.path.join(output_dir, f"checkpoint-{best_checkpoint}")
        if is_pretrained:
            best_model = RegressionPreTrainedModel.from_pretrained(checkpoint)
        else:
            best_model = RegressionModel()
            if not safe_weights:
                state_dict = load_checkpoint(os.path.join(checkpoint, WEIGHTS_NAME))
            else:
                state_dict = safe_load_file(os.path.join(checkpoint, SAFE_WEIGHTS_NAME))
            best_model.load_state_dict(state_dict)
        self.assertTrue(np.allclose(best_model.a.asnumpy(), trainer.model.a.asnumpy()))
        self.assertTrue(np.allclose(best_model.b.asnumpy(), trainer.model.b.asnumpy()))

        metrics = trainer.evaluate()
        self.assertEqual(metrics[metric], best_value)

    def check_trainer_state_are_the_same(self, trainer_state, trainer_state1):
        # We'll pop things so operate on copies.
        state = trainer_state.copy()
        state1 = trainer_state1.copy()
        # Log history main contain different logs for the time metrics (after resuming a training).
        log_history = state.pop("log_history", None)
        log_history1 = state1.pop("log_history", None)
        self.assertEqual(state, state1)
        skip_log_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"]
        for log, log1 in zip(log_history, log_history1):
            for key in skip_log_keys:
                _ = log.pop(key, None)
                _ = log1.pop(key, None)
            self.assertEqual(log, log1)

    def convert_to_sharded_checkpoint(self, folder, save_safe=True, load_safe=True):
        # Converts a checkpoint of a regression model to a sharded checkpoint.
        if load_safe:
            loader = safe_load_file
            weights_file = os.path.join(folder, SAFE_WEIGHTS_NAME)
        else:
            loader = load_checkpoint
            weights_file = os.path.join(folder, WEIGHTS_NAME)

        if save_safe:
            extension = "safetensors"
            saver = safe_save_file
            index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
            shard_name = SAFE_WEIGHTS_NAME
        else:
            extension = "ckpt"
            saver = save_checkpoint
            index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
            shard_name = WEIGHTS_NAME

        state_dict = loader(weights_file)

        os.remove(weights_file)
        keys = list(state_dict.keys())

        shard_files = [
            shard_name.replace(f".{extension}", f"-{idx+1:05d}-of-{len(keys):05d}.{extension}")
            for idx in range(len(keys))
        ]
        index = {"metadata": {}, "weight_map": {key: shard_files[i] for i, key in enumerate(keys)}}

        with open(index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        for param_name, shard_file in zip(keys, shard_files):
            saver({param_name: state_dict[param_name]}, os.path.join(folder, shard_file))

@require_mindspore
@require_sentencepiece
@require_tokenizers
class TrainerIntegrationPrerunTest(TestCasePlus, TrainerIntegrationCommon):
    """
    Only tests that want to tap into the auto-pre-run 2 trainings:
    - self.default_trained_model
    - self.alternate_trained_model
    directly, or via check_trained_model
    """

    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.default_trained_model = (trainer.model.a, trainer.model.b)

        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.alternate_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, alternate_seed=False):
        # Checks a training seeded with learning_rate = 0.1
        (a, b) = self.alternate_trained_model if alternate_seed else self.default_trained_model
        self.assertTrue(np.allclose(model.a.asnumpy(), a.asnumpy()))
        self.assertTrue(np.allclose(model.b.asnumpy(), b.asnumpy()))

    def test_reproducible_training(self):
        # Checks that training worked, model trained and seed made a reproducible training.
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Checks that a different seed gets different (reproducible) results.
        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.check_trained_model(trainer.model, alternate_seed=True)

    def test_trainer_with_datasets(self):
        import datasets

        np.random.seed(42)
        class MyAccessible:
            def __init__(self):
                self._data = np.random.normal(size=(64,)).astype(np.float32)
                self._label = 2.0 * self._data + 3.0 + np.random.normal(scale=0.1, size=(64,)).astype(np.float32)

            def __getitem__(self, index):
                return self._data[index], self._label[index]

            def __len__(self):
                return len(self._data)

        train_dataset = GeneratorDataset(MyAccessible(), column_names=["input_x", "labels"])

        # Base training. Should have the same results as test_reproducible_training
        model = RegressionModel()
        args = TrainingArguments("./regression", learning_rate=0.1)
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

        # # Can return tensors.
        # model = RegressionModel()
        # trainer = Trainer(model, args, train_dataset=train_dataset)
        # trainer.train()
        # self.check_trained_model(trainer.model)

        # Adding one column not used by the model should have no impact
        
        class MyAccessible:
            def __init__(self):
                self._data = np.random.normal(size=(64,)).astype(np.float32)
                self._label = 2.0 * self._data + 3.0 + np.random.normal(scale=0.1, size=(64,)).astype(np.float32)
                self._extra = np.random.normal(size=(64,)).astype(np.float32)

            def __getitem__(self, index):
                return self._data[index], self._label[index], self._extra[index]

            def __len__(self):
                return len(self._data)

        train_dataset = GeneratorDataset(MyAccessible(), column_names=["input_x", "labels", "extra"])
        model = RegressionModel()
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

    def test_model_init(self):
        train_dataset = GeneratorDataset(RegressionDataset(), column_names=['input_x', 'labels'])

        args = TrainingArguments("./regression", learning_rate=0.1)
        trainer = Trainer(args=args, train_dataset=train_dataset, model_init=lambda: RegressionModel())
        trainer.train()
        self.check_trained_model(trainer.model)

        # Re-training should restart from scratch, thus lead the same results.
        trainer.train()
        self.check_trained_model(trainer.model)

        # Re-training should restart from scratch, thus lead the same results and new seed should be used.
        trainer.args.seed = 314
        trainer.train()
        self.check_trained_model(trainer.model, alternate_seed=True)

    # TODO: support gradient accumulation
    # def test_gradient_accumulation(self):
    #     # Training with half the batch size but accumulation steps as 2 should give the same results.
    #     trainer = get_regression_trainer(
    #         gradient_accumulation_steps=2, per_device_train_batch_size=4, learning_rate=0.1
    #     )
    #     trainer.train()
    #     self.check_trained_model(trainer.model)

    # def test_gradient_checkpointing(self):
    #     trainer = get_regression_trainer(
    #         per_device_train_batch_size=1,
    #         learning_rate=0.1,
    #         gradient_checkpointing=True,
    #         gradient_checkpointing_kwargs={"use_reentrant": False},
    #     )
    #     previous_params = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}

    #     trainer.train()

    #     # Check if model weights have been updated
    #     for k, v in trainer.model.named_parameters():
    #         self.assertFalse(
    #             np.allclose(previous_params[k], v, rtol=1e-4, atol=1e-4),
    #             f"Model weights for {k} have not been updated",
    #         )

    def test_training_loss(self):
        # With even logs
        trainer = get_regression_trainer(logging_steps=8)
        trainer.train()
        log_history = trainer.state.log_history

        losses = [log["loss"] for log in log_history if "loss" in log]
        train_loss = log_history[-1]["train_loss"]
        self.assertAlmostEqual(sum(losses) / len(losses), train_loss, places=4)

        # With uneven logs
        trainer = get_regression_trainer(logging_steps=5*8)
        trainer.train()
        log_history = trainer.state.log_history

        # Training loss should be the same as before
        new_train_loss = log_history[-1]["train_loss"]
        self.assertAlmostEqual(train_loss, new_train_loss, places=4)

    def test_custom_optimizer(self):
        train_dataset = GeneratorDataset(RegressionDataset(), column_names=['input_x', 'labels'])
        args = TrainingArguments("./regression")
        model = RegressionModel()

        optimizer = optim.SGD(model.trainable_params(), lr=1.0)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
        trainer = Trainer(model, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler))
        trainer.train()

        (a, b) = self.default_trained_model
        self.assertFalse(np.allclose(trainer.model.a.asnumpy(), a.asnumpy()))
        self.assertFalse(np.allclose(trainer.model.b.asnumpy(), b.asnumpy()))
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1.0)

    def test_lr_scheduler_kwargs(self):
        # test scheduler kwargs passed via TrainingArguments
        train_dataset = RegressionDataset()
        model = RegressionModel()
        num_steps, num_warmup_steps = 10, 2
        extra_kwargs = {"power": 5.0, "lr_end": 1e-5}  # Non-default arguments
        args = TrainingArguments(
            "./regression",
            lr_scheduler_type="polynomial",
            lr_scheduler_kwargs=extra_kwargs,
            learning_rate=0.2,
            warmup_steps=num_warmup_steps,
        )
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.create_optimizer_and_scheduler(num_training_steps=num_steps)

        # Checking that the scheduler was created
        self.assertIsNotNone(trainer.lr_scheduler)

        # Checking that the correct args were passed
        sched1 = trainer.lr_scheduler
        sched2 = get_polynomial_decay_schedule_with_warmup(
            trainer.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps, **extra_kwargs
        )
        self.assertEqual(sched1.lr_lambdas[0].args, sched2.lr_lambdas[0].args)
        self.assertEqual(sched1.lr_lambdas[0].keywords, sched2.lr_lambdas[0].keywords)

    def test_reduce_lr_on_plateau_args(self):
        # test passed arguments for a custom ReduceLROnPlateau scheduler
        train_dataset = GeneratorDataset(RegressionDataset(length=64), column_names=['input_x', 'labels'])
        eval_dataset = GeneratorDataset(RegressionDataset(length=64), column_names=['input_x', 'labels'])

        args = TrainingArguments(
            "./regression",
            evaluation_strategy="epoch",
            metric_for_best_model="eval_loss",
        )
        model = RegressionModel()
        optimizer = optim.SGD(model.trainable_params(), lr=1.0)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)
        trainer = Trainer(
            model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, optimizers=(optimizer, lr_scheduler)
        )
        trainer.train()

        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(trainer.lr_scheduler.factor, 0.2)
        self.assertEqual(trainer.lr_scheduler.patience, 5)
        self.assertEqual(trainer.lr_scheduler.cooldown, 2)

    def test_reduce_lr_on_plateau(self):
        # test the ReduceLROnPlateau scheduler

        class TrainerWithLRLogs(Trainer):
            def log(self, logs):
                # the LR is computed after metrics and does not exist for the first epoch
                if hasattr(self.lr_scheduler, "_last_lr"):
                    logs["learning_rate"] = self.lr_scheduler._last_lr[0]
                super().log(logs)

        train_dataset = GeneratorDataset(RegressionDataset(length=64), column_names=['input_x', 'labels'])
        eval_dataset = GeneratorDataset(RegressionDataset(length=64), column_names=['input_x', 'labels'])

        args = TrainingArguments(
            "./regression",
            lr_scheduler_type="reduce_lr_on_plateau",
            evaluation_strategy="epoch",
            metric_for_best_model="eval_loss",
            num_train_epochs=10,
            learning_rate=0.2,
        )
        model = RegressionModel()
        trainer = TrainerWithLRLogs(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()

        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau)
        patience = trainer.lr_scheduler.patience

        logs = trainer.state.log_history[1:]
        best_loss = logs[0]["eval_loss"]
        bad_epochs = 0
        for i, log in enumerate(logs[:-1]):  # Compare learning rate to next epoch's
            loss = log["eval_loss"]
            just_decreased = False
            if loss > best_loss:
                bad_epochs += 1
                if bad_epochs > patience:
                    self.assertLess(logs[i + 1]["learning_rate"], log["learning_rate"])
                    just_decreased = True
                    bad_epochs = 0
            else:
                best_loss = loss
                bad_epochs = 0
            if not just_decreased:
                self.assertEqual(logs[i + 1]["learning_rate"], log["learning_rate"])

    # def test_adafactor_lr_none(self):
    #     # test the special case where lr=None, since Trainer can't not have lr_scheduler
    #     from mindspore.nn import AdaFactor
    #     # from transformers.optimization import Adafactor, AdafactorSchedule

    #     train_dataset = RegressionDataset()
    #     args = TrainingArguments("./regression")
    #     model = RegressionModel()
    #     optimizer = AdaFactor(model.trainable_params(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    #     lr_scheduler = AdafactorSchedule(optimizer)
    #     trainer = Trainer(model, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler))
    #     trainer.train()

    #     (a, b) = self.default_trained_model
    #     self.assertFalse(np.allclose(trainer.model.a, a))
    #     self.assertFalse(np.allclose(trainer.model.b, b))
    #     self.assertGreater(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 0)

    # TODO: support bf16
    # def test_mixed_bf16(self):
    #     # very basic test
    #     trainer = get_regression_trainer(learning_rate=0.1, bf16=True)
    #     trainer.train()
    #     self.check_trained_model(trainer.model)

    #     # --bf16 --half_precision_backend apex can't be used together
    #     with self.assertRaises(ValueError):
    #         trainer = get_regression_trainer(learning_rate=0.1, bf16=True, half_precision_backend="apex")

    #     # will add more specific tests once there are some bugs to fix


@require_mindspore
@require_sentencepiece
@require_tokenizers
class TrainerIntegrationTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_trainer_works_with_dict(self):
        # Edge case because Apex with mode O2 will change our models to return dicts. This test checks it doesn't break
        # anything.
        train_dataset = GeneratorDataset(RegressionDataset(), column_names=['input_x', 'labels'])
        eval_dataset = GeneratorDataset(RegressionDataset(), column_names=['input_x', 'labels'])
        model = RegressionDictModel()
        args = TrainingArguments("./regression")
        trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
        _ = trainer.evaluate()
        _ = trainer.predict(GeneratorDataset(RegressionDataset(), column_names=['input_x', 'labels']))

    def test_evaluation_with_keys_to_drop(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = ops.randint(0, 100, (128,))
        eval_dataset = GeneratorDataset(RepeatDataset(x), column_names=['input_ids', 'labels'])
        args = TrainingArguments("./test")
        trainer = Trainer(tiny_gpt2, args, eval_dataset=eval_dataset)
        # By default the past_key_values are removed
        result = trainer.predict(eval_dataset)
        self.assertTrue(isinstance(result.predictions, np.ndarray))
        # We can still get them by setting ignore_keys to []
        result = trainer.predict(eval_dataset, ignore_keys=[])
        self.assertTrue(isinstance(result.predictions, tuple))
        self.assertEqual(len(result.predictions), 2)

    def test_training_arguments_are_left_untouched(self):
        trainer = get_regression_trainer()
        trainer.train()
        args = TrainingArguments("./regression")
        dict1, dict2 = args.to_dict(), trainer.args.to_dict()
        for key in dict1.keys():
            # Logging dir can be slightly different as they default to something with the time.
            if key != "logging_dir":
                self.assertEqual(dict1[key], dict2[key])

    def test_number_of_steps_in_training(self):
        # Regular training has n_epochs * len(train_dl) steps
        trainer = get_regression_trainer(learning_rate=0.1)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, self.n_epochs * 64 / self.batch_size)

        # Check passing num_train_epochs works (and a float version too):
        trainer = get_regression_trainer(learning_rate=0.1, num_train_epochs=1.5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(1.5 * 64 / self.batch_size))

        # If we pass a max_steps, num_train_epochs is ignored
        trainer = get_regression_trainer(learning_rate=0.1, max_steps=10)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 10)

    def test_neftune(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = np.random.randint(0, 100, (128,))
        train_dataset = GeneratorDataset(RepeatDataset(x), column_names=['input_ids', 'labels'])

        # Trainer without inf/nan filter
        args = TrainingArguments(
            "./test", learning_rate=1e-9, logging_steps=5, logging_nan_inf_filter=False, neftune_noise_alpha=0.4
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)

        trainer.model = trainer._activate_neftune(trainer.model)

        dummy_input = mindspore.Tensor([[1, 0, 1]])
        emb1 = trainer.model.get_input_embeddings()(dummy_input)
        emb2 = trainer.model.get_input_embeddings()(dummy_input)
        self.assertFalse(np.allclose(emb1.asnumpy(), emb2.asnumpy()), "Neftune noise is not applied!")

        # redefine the model
        tiny_gpt2 = GPT2LMHeadModel(config)
        # Trainer without inf/nan filter
        args = TrainingArguments(
            "./test", learning_rate=1e-9, logging_steps=5, logging_nan_inf_filter=False, neftune_noise_alpha=0.4
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        # Check that it trains without errors
        trainer.train()

        # Make sure forward pass works fine
        _ = trainer.model(dummy_input)
        self.assertTrue(len(trainer.model.get_input_embeddings()._forward_hooks) == 0)

        trainer.model.set_train(False)
        # Check that we get identical embeddings just in case
        emb1 = trainer.model.get_input_embeddings()(dummy_input)
        emb2 = trainer.model.get_input_embeddings()(dummy_input)

        self.assertTrue(np.allclose(emb1.asnumpy(), emb2.asnumpy()), "Neftune noise is still applied!")

    def test_logging_inf_nan_filter(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = ops.randint(0, 100, (128,))
        train_dataset = GeneratorDataset(RepeatDataset(x), column_names=['input_ids', 'labels'])

        # Trainer without inf/nan filter
        args = TrainingArguments("./test", learning_rate=1e9, logging_steps=5, logging_nan_inf_filter=False)
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_no_filter = trainer.state.log_history

        # Trainer with inf/nan filter
        args = TrainingArguments("./test", learning_rate=1e9, logging_steps=5, logging_nan_inf_filter=True)
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_filter = trainer.state.log_history

        def is_any_loss_nan_or_inf(log_history):
            losses = [l["loss"] for l in log_history[:-1]]
            return any(math.isnan(x) for x in losses) or any(math.isinf(x) for x in losses)

        self.assertTrue(is_any_loss_nan_or_inf(log_history_no_filter))
        self.assertFalse(is_any_loss_nan_or_inf(log_history_filter))

    def test_train_and_eval_dataloaders(self):
        n_gpu = 1
        trainer = get_regression_trainer(learning_rate=0.1, per_device_train_batch_size=16)
        self.assertEqual(trainer.get_train_dataset().get_batch_size(), 16 * n_gpu)
        trainer = get_regression_trainer(learning_rate=0.1, per_device_eval_batch_size=16)
        self.assertEqual(trainer.get_eval_dataset().get_batch_size(), 16 * n_gpu)

        # Check drop_last works
        trainer = get_regression_trainer(
            train_len=66, eval_len=74, learning_rate=0.1, per_device_train_batch_size=16, per_device_eval_batch_size=32
        )
        self.assertEqual(len(trainer.get_train_dataset()), 66 // (16 * n_gpu) + 1)
        self.assertEqual(len(trainer.get_eval_dataset()), 74 // (32 * n_gpu) + 1)

        trainer = get_regression_trainer(
            train_len=66,
            eval_len=74,
            learning_rate=0.1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            dataset_drop_last=True,
        )
        self.assertEqual(len(trainer.get_train_dataset()), 66 // (16 * n_gpu))
        self.assertEqual(len(trainer.get_eval_dataset()), 74 // (32 * n_gpu))

        # Check passing a new dataset for evaluation works
        new_eval_dataset = GeneratorDataset(RegressionDataset(length=128), column_names=['input_x', 'labels'])
        self.assertEqual(len(trainer.get_eval_dataset(new_eval_dataset)), 128 // (32 * n_gpu))

    # tests that we do not require mindspore.dataset.Dataset
    def test_data_without_dataset(self):
        pass

    def test_evaluate(self):
        trainer = get_regression_trainer(a=1.5, b=2.5, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.source.x, trainer.eval_dataset.source.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.source.x, trainer.eval_dataset.source.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With logits preprocess
        trainer = get_regression_trainer(
            a=1.5,
            b=2.5,
            compute_metrics=AlmostAccuracy(),
            preprocess_logits_for_metrics=lambda logits, labels: logits + 1,
        )
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.source.x, trainer.eval_dataset.source.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred + 1, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_evaluate_with_jit(self):
        trainer = get_regression_trainer(a=1.5, b=2.5, compute_metrics=AlmostAccuracy(), jit_mode_eval=True)
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.source.x, trainer.eval_dataset.source.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(
            a=1.5, b=2.5, eval_len=66, compute_metrics=AlmostAccuracy(), jit_mode_eval=True
        )
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.source.x, trainer.eval_dataset.source.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With logits preprocess
        trainer = get_regression_trainer(
            a=1.5,
            b=2.5,
            compute_metrics=AlmostAccuracy(),
            preprocess_logits_for_metrics=lambda logits, labels: logits + 1,
            jit_mode_eval=True,
        )
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.source.x, trainer.eval_dataset.source.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred + 1, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict(self):
        trainer = get_regression_trainer(a=1.5, b=2.5)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.source.x
        print(preds, 1.5 * x + 2.5)
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.source.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With more than one output of the model
        trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.source.x
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

        # With more than one output/label of the model
        trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True, label_names=["labels", "labels_2"])
        outputs = trainer.predict(trainer.eval_dataset)
        preds = outputs.predictions
        labels = outputs.label_ids
        x = trainer.eval_dataset.source.x
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
        self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.source.ys[0]))
        self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.source.ys[1]))

    def test_predict_with_jit(self):
        trainer = get_regression_trainer(a=1.5, b=2.5, jit_mode_eval=True)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.source.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, jit_mode_eval=True)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.source.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With more than one output of the model
        trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True, jit_mode_eval=True)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.source.x
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

        # With more than one output/label of the model
        trainer = get_regression_trainer(
            a=1.5, b=2.5, double_output=True, label_names=["labels", "labels_2"], jit_mode_eval=True
        )
        outputs = trainer.predict(trainer.eval_dataset)
        preds = outputs.predictions
        labels = outputs.label_ids
        x = trainer.eval_dataset.source.x
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
        self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.source.ys[0]))
        self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.source.ys[1]))

    @pytest.mark.skip('not support dynamic shape')
    def test_dynamic_shapes(self):
        eval_dataset = GeneratorDataset(DynamicShapesDataset(batch_size=self.batch_size), column_names=['input_x', 'labels'])
        model = RegressionModel(a=2, b=1)
        args = TrainingArguments("./regression")
        trainer = Trainer(model, args, eval_dataset=eval_dataset)

        # Check evaluation can run to completion
        _ = trainer.evaluate()

        # Check predictions
        preds = trainer.predict(eval_dataset)
        for expected, seen in zip(eval_dataset.ys, preds.label_ids):
            self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

        for expected, seen in zip(eval_dataset.xs, preds.predictions):
            self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

        # Same tests with eval accumulation
        args = TrainingArguments("./regression", eval_accumulation_steps=2)
        trainer = Trainer(model, args, eval_dataset=eval_dataset)

        # Check evaluation can run to completion
        _ = trainer.evaluate()

        # Check predictions
        preds = trainer.predict(eval_dataset)
        for expected, seen in zip(eval_dataset.ys, preds.label_ids):
            self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

        for expected, seen in zip(eval_dataset.xs, preds.predictions):
            self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
            self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    def test_log_level(self):
        # testing only --log_level (--log_level_replica requires multiple gpus and DDP and is tested elsewhere)
        logger = logging.get_logger()
        log_info_string = "Running training"

        # test with the default log_level - should be the same as before and thus we test depending on is_info
        is_info = logging.get_verbosity() <= 20
        with CaptureLogger(logger) as cl:
            trainer = get_regression_trainer()
            trainer.train()
        if is_info:
            self.assertIn(log_info_string, cl.out)
        else:
            self.assertNotIn(log_info_string, cl.out)

        with LoggingLevel(logging.INFO):
            # test with low log_level - lower than info
            with CaptureLogger(logger) as cl:
                trainer = get_regression_trainer(log_level="debug")
                trainer.train()
            self.assertIn(log_info_string, cl.out)

        with LoggingLevel(logging.INFO):
            # test with high log_level - should be quiet
            with CaptureLogger(logger) as cl:
                trainer = get_regression_trainer(log_level="error")
                trainer.train()
            self.assertNotIn(log_info_string, cl.out)

    def test_save_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, int(self.n_epochs * 64 / self.batch_size))

        # With a regular model that is not a PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5, pretrained=False)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, int(self.n_epochs * 64 / self.batch_size), False)

    def test_save_checkpoints_is_atomic(self):
        class UnsaveableTokenizer(PreTrainedTokenizerBase):
            def save_pretrained(self, *args, **kwargs):
                raise OSError("simulated file write error")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5)
            # Attach unsaveable tokenizer to partially fail checkpointing
            trainer.tokenizer = UnsaveableTokenizer()
            with self.assertRaises(OSError) as _context:
                trainer.train()
            # assert get_last_checkpoint(tmpdir) is None

    @require_safetensors
    def test_safe_checkpoints(self):
        for save_safetensors in [True, False]:
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5, save_safetensors=save_safetensors)
                trainer.train()
                self.check_saved_checkpoints(
                    tmpdir, 5, int(self.n_epochs * 64 / self.batch_size), safe_weights=save_safetensors
                )

            # With a regular model that is not a PreTrainedModel
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = get_regression_trainer(
                    output_dir=tmpdir, save_steps=5, pretrained=False, save_safetensors=save_safetensors
                )
                trainer.train()
                self.check_saved_checkpoints(
                    tmpdir, 5, int(self.n_epochs * 64 / self.batch_size), False, safe_weights=save_safetensors
                )

    def test_run_seq2seq_double_train_wrap_once(self):
        # test that we don't wrap the model more than once
        # since wrapping primarily happens on multi-gpu setup we want multiple gpus to test for
        # example DataParallel(DataParallel(model))

        trainer = get_regression_trainer()
        trainer.train()
        model_wrapped_before = trainer.model
        trainer.train()
        model_wrapped_after = trainer.model
        self.assertIs(model_wrapped_before, model_wrapped_after, "should be not wrapped twice")

    @pytest.mark.skipif(sys.platform == 'darwin', reason="MacOS cannot get same loss after resume.")
    def test_can_resume_training(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {
                "output_dir": tmpdir,
                "train_len": 128,
                "save_steps": 16,
                "learning_rate": 0.1,
                "logging_steps": 16,
            }
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-16")

            # Reinitialize trainer
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertAlmostEqual(a, a1, 4)
            self.assertAlmostEqual(b, b1, 4)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(tmpdir, "checkpoint-48")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertAlmostEqual(a, a1, 4)
            self.assertAlmostEqual(b, b1, 4)
            self.check_trainer_state_are_the_same(state, state1)

        # With a regular model that is not a PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {
                "output_dir": tmpdir,
                "train_len": 128,
                "save_steps": 16,
                "learning_rate": 0.1,
                "pretrained": False,
            }

            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-16")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertAlmostEqual(a, a1, 4)
            self.assertAlmostEqual(b, b1, 4)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(tmpdir, "checkpoint-48")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertAlmostEqual(a, a1, 4)
            self.assertAlmostEqual(b, b1, 4)
            self.check_trainer_state_are_the_same(state, state1)

        # Now check failures

        # 1. fail to find a bogus checkpoint
        trainer = get_regression_trainer()
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
        self.assertTrue("Can't find a valid checkpoint at" in str(context.exception))

        # 2. fail to find any checkpoint - due a fresh output_dir
        output_dir2 = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=output_dir2)
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=True)
        self.assertTrue("No valid checkpoint found in output directory" in str(context.exception))

    @unittest.skip(
        reason="@muellerzr: Fix once Trainer can take an accelerate configuration. Need to set `seedable_sampler=True`."
    )
    def test_resume_training_with_randomness(self):
        # For more than 1 GPUs, since the randomness is introduced in the model and with DataParallel (which is used
        # in this test for more than 2 GPUs), the calls to the torch RNG will happen in a random order (sometimes
        # GPU 0 will call first and sometimes GPU 1).
        train_dataset = RegressionDataset(length=128)
        eval_dataset = RegressionDataset()

        with self.subTest("Test every step"):
            config = RegressionModelConfig(a=0, b=2, random_ms=random_ms)
            model = RegressionRandomPreTrainedModel(config)

            tmp_dir = self.get_auto_remove_tmp_dir()
            args = RegressionTrainingArguments(tmp_dir, save_steps=5, learning_rate=0.1)
            trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()

            model = RegressionRandomPreTrainedModel(config)
            trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
            trainer.train(resume_from_checkpoint=os.path.join(tmp_dir, "checkpoint-15"))
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()

            self.assertAlmostEqual(a, a1, delta=1e-5)
            self.assertAlmostEqual(b, b1, delta=1e-5)

        with self.subTest("Test every epoch"):
            config = RegressionModelConfig(a=0, b=2)
            model = RegressionRandomPreTrainedModel(config)

            tmp_dir = self.get_auto_remove_tmp_dir()
            args = RegressionTrainingArguments(tmp_dir, save_strategy="epoch", learning_rate=0.1)
            trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()

            model = RegressionRandomPreTrainedModel(config)
            trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

            checkpoints = [d for d in os.listdir(tmp_dir) if d.startswith("checkpoint-")]
            # There should be one checkpoint per epoch.
            self.assertEqual(len(checkpoints), 3)
            checkpoint_dir = sorted(checkpoints, key=lambda x: int(x.replace("checkpoint-", "")))[0]

            trainer.train(resume_from_checkpoint=os.path.join(tmp_dir, checkpoint_dir))
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()

            self.assertAlmostEqual(a, a1, delta=1e-5)
            self.assertAlmostEqual(b, b1, delta=1e-5)

    @pytest.mark.skip('not support auto batch size now')
    def test_auto_batch_size_with_resume_from_checkpoint(self):
        train_dataset = GeneratorDataset(RegressionDataset(length=128), column_names=['input_x', 'labels'])

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()

        class MockCudaOOMCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                # simulate OOM on the first step
                if state.train_batch_size >= 16:
                    raise RuntimeError("CUDA out of memory.")

        args = RegressionTrainingArguments(
            tmp_dir,
            do_train=True,
            max_steps=2,
            save_steps=1,
            per_device_train_batch_size=16,
            auto_find_batch_size=True,
        )
        trainer = Trainer(model, args, train_dataset=train_dataset, callbacks=[MockCudaOOMCallback()])
        trainer.train()
        # After `auto_find_batch_size` is ran we should now be at 8
        self.assertEqual(trainer._train_batch_size, 8)

        # We can then make a new Trainer
        trainer = Trainer(model, args, train_dataset=train_dataset)
        # Check we are at 16 to start
        self.assertEqual(trainer._train_batch_size, 16 * max(trainer.args.n_gpu, 1))
        trainer.train(resume_from_checkpoint=True)
        # We should be back to 8 again, picking up based upon the last ran Trainer
        self.assertEqual(trainer._train_batch_size, 8)

    # regression for this issue: https://github.com/huggingface/transformers/issues/12970
    def test_training_with_resume_from_checkpoint_false(self):
        train_dataset = GeneratorDataset(RegressionDataset(length=128), column_names=['input_x', 'labels'])
        eval_dataset = GeneratorDataset(RegressionDataset(), column_names=['input_x', 'labels'])

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()
        args = RegressionTrainingArguments(tmp_dir, save_steps=5, learning_rate=0.1)
        trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

        trainer.train(resume_from_checkpoint=False)

    def test_resume_training_with_shard_checkpoint(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, train_len=128, save_steps=16, learning_rate=0.1)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-16")
            self.convert_to_sharded_checkpoint(checkpoint)

            # Reinitialize trainer
            trainer = get_regression_trainer(output_dir=tmpdir, train_len=128, save_steps=5, learning_rate=0.1)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertAlmostEqual(a, a1, 4)
            self.assertAlmostEqual(b, b1, 4)
            self.check_trainer_state_are_the_same(state, state1)

    @require_safetensors
    def test_resume_training_with_safe_checkpoint(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        for initial_safe in [False, True]:
            for loaded_safe in [False, True]:
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainer = get_regression_trainer(
                        output_dir=tmpdir,
                        train_len=128,
                        save_steps=16,
                        learning_rate=0.1,
                        save_safetensors=initial_safe,
                    )
                    trainer.train()
                    (a, b) = trainer.model.a.item(), trainer.model.b.item()
                    state = dataclasses.asdict(trainer.state)

                    checkpoint = os.path.join(tmpdir, "checkpoint-16")
                    self.convert_to_sharded_checkpoint(checkpoint, load_safe=initial_safe, save_safe=loaded_safe)

                    # Reinitialize trainer
                    trainer = get_regression_trainer(
                        output_dir=tmpdir, train_len=128, save_steps=16, learning_rate=0.1, save_safetensors=loaded_safe
                    )

                    trainer.train(resume_from_checkpoint=checkpoint)
                    (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
                    state1 = dataclasses.asdict(trainer.state)
                    self.assertAlmostEqual(a, a1, 4)
                    self.assertAlmostEqual(b, b1, 4)
                    self.check_trainer_state_are_the_same(state, state1)

    def test_resume_training_with_gradient_accumulation(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                save_steps=16,
                learning_rate=0.1,
            )
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-16")

            # Reinitialize trainer
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                save_steps=16,
                learning_rate=0.1,
            )

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertAlmostEqual(a, a1, 4)
            self.assertAlmostEqual(b, b1, 4)
            self.check_trainer_state_are_the_same(state, state1)

    def test_resume_training_with_frozen_params(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                per_device_train_batch_size=4,
                save_steps=32,
                learning_rate=0.1,
            )
            trainer.model.a.requires_grad = False
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-32")

            # Reinitialize trainer
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                per_device_train_batch_size=4,
                save_steps=32,
                learning_rate=0.1,
            )
            trainer.model.a.requires_grad = False

            trainer.train(resume_from_checkpoint=checkpoint)

            self.assertFalse(trainer.model.a.requires_grad)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertAlmostEqual(a, a1, 4)
            self.assertAlmostEqual(b, b1, 4)
            self.check_trainer_state_are_the_same(state, state1)

    def test_load_best_model_at_end(self):
        total = int(self.n_epochs * 64 / self.batch_size)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
            )
            self.assertFalse(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_accuracy", greater_is_better=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 64 // self.batch_size, total)
            self.check_best_model_has_been_loaded(
                tmpdir, 64 // self.batch_size, total, trainer, "eval_accuracy", greater_is_better=True
            )

        # Test this works with a non PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
                pretrained=False,
            )
            self.assertFalse(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total, is_pretrained=False)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss", is_pretrained=False)

    @require_safetensors
    def test_load_best_model_from_safetensors(self):
        total = int(self.n_epochs * 64 / self.batch_size)
        for save_safetensors, pretrained in product([False, True], [False, True]):
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = get_regression_trainer(
                    a=1.5,
                    b=2.5,
                    output_dir=tmpdir,
                    learning_rate=0.1,
                    eval_steps=5,
                    evaluation_strategy="steps",
                    save_steps=5,
                    load_best_model_at_end=True,
                    save_safetensors=save_safetensors,
                    pretrained=pretrained,
                )
                self.assertFalse(trainer.args.greater_is_better)
                trainer.train()
                self.check_saved_checkpoints(tmpdir, 5, total, is_pretrained=pretrained, safe_weights=save_safetensors)
                self.check_best_model_has_been_loaded(
                    tmpdir, 5, total, trainer, "eval_loss", is_pretrained=pretrained, safe_weights=save_safetensors
                )

    @slow
    def test_trainer_eval_mrpc(self):
        MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir=f"{get_tests_dir()}/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        training_args = TrainingArguments(output_dir="./examples", use_cpu=True)
        trainer = Trainer(model=model, args=training_args, eval_dataset=eval_dataset)
        result = trainer.evaluate()
        self.assertLess(result["eval_loss"], 0.2)

    @slow
    def test_trainer_eval_multiple(self):
        MODEL_ID = "openai-community/gpt2"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=PATH_SAMPLE_TEXT,
            block_size=tokenizer.max_len_single_sentence,
        )
        for example in dataset.examples:
            example["labels"] = example["input_ids"]
        training_args = TrainingArguments(
            output_dir="./examples",
            use_cpu=True,
            per_device_eval_batch_size=1,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset={
                "data1": dataset,
                "data2": dataset,
            },
        )
        result = trainer.evaluate()
        self.assertIn("eval_data1_loss", result)
        self.assertIn("eval_data2_loss", result)

    @slow
    def test_trainer_eval_lm(self):
        MODEL_ID = "distilbert/distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=PATH_SAMPLE_TEXT,
            block_size=tokenizer.max_len_single_sentence,
        )
        self.assertEqual(len(dataset), 31)

    def test_training_iterable_dataset(self):
        config = RegressionModelConfig()
        model = RegressionPreTrainedModel(config)
        # Adding one column not used by the model should have no impact
        train_dataset = GeneratorDataset(SampleIterableDataset(label_names=["labels", "extra"]), column_names=['input_x', "labels", "extra"])

        args = RegressionTrainingArguments(output_dir="./examples", max_steps=4)
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
        trainer.train()
        self.assertEqual(trainer.state.global_step, 4)

        loader = trainer.get_train_dataset()
        self.assertIsInstance(loader, mindspore.dataset.Dataset)

    def test_evaluation_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        # Adding one column not used by the model should have no impact
        eval_dataset = GeneratorDataset(SampleIterableDataset(label_names=["labels", "extra"]),
                                        column_names=['input_x', "labels", "extra"], shuffle=False)

        args = RegressionTrainingArguments(output_dir="./examples", per_device_eval_batch_size=64)
        trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.source.dataset.x, trainer.eval_dataset.source.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        eval_dataset = GeneratorDataset(SampleIterableDataset(length=66),
                                        column_names=['input_x', "labels"], shuffle=False)
        results = trainer.evaluate(eval_dataset)

        x, y = eval_dataset.source.dataset.x, eval_dataset.source.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = GeneratorDataset(SampleIterableDataset(), column_names=['input_x', 'labels'])

        args = RegressionTrainingArguments(output_dir="./examples")
        trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, compute_metrics=AlmostAccuracy())

        preds = trainer.predict(trainer.eval_dataset).predictions
        x = eval_dataset.source.dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With a number of elements not a round multiple of the batch size
        # Adding one column not used by the model should have no impact
        test_dataset = GeneratorDataset(SampleIterableDataset(length=66, label_names=["labels", "extra"]),
                                        column_names=['input_x', 'labels', "extra"])
        preds = trainer.predict(test_dataset).predictions
        x = test_dataset.source.dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

    def test_num_train_epochs_in_training(self):
        # len(train_dl) < gradient_accumulation_steps shouldn't give ``ZeroDivisionError`` when ``max_steps`` is given.
        # It should give 1 update step for each epoch.
        trainer = get_regression_trainer(
            max_steps=3, train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5
        )
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 3)

        # Even ``max_steps`` is not specified, we still expect 1 update step for each epoch if
        # len(train_dl) < gradient_accumulation_steps.
        trainer = get_regression_trainer(train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(self.n_epochs))

    def test_early_stopping_callback(self):
        # early stopping stops training before num_training_epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=True,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
            train_output = trainer.train()
            self.assertLess(train_output.global_step, 20 * 64 / 16)

        # Invalid inputs to trainer with early stopping callback result in assertion error
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                evaluation_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1))
            self.assertEqual(trainer.state.global_step, 0)
            try:
                trainer.train()
            except AssertionError:
                self.assertEqual(trainer.state.global_step, 0)

    def test_flos_extraction(self):
        trainer = get_regression_trainer(learning_rate=0.1)

        def assert_flos_extraction(trainer, wrapped_model_to_check):
            self.assertEqual(trainer.model, wrapped_model_to_check)
            self.assertGreaterEqual(getattr(wrapped_model_to_check.config, "total_flos", 0), 0)

        # with plain model
        assert_flos_extraction(trainer, trainer.model)

        # # with enforced DataParallel
        # assert_flos_extraction(trainer, nn.DataParallel(trainer.model))

        trainer.train()
        self.assertTrue(isinstance(trainer.state.total_flos, float))

    def check_checkpoint_deletion(self, trainer, output_dir, expected):
        # Make fake checkpoints
        for n in [5, 10, 15, 20, 25]:
            os.makedirs(os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{n}"), exist_ok=True)
        trainer._rotate_checkpoints(output_dir=output_dir)
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")]
        values = [int(re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", d).groups()[0]) for d in glob_checkpoints]
        self.assertSetEqual(set(values), set(expected))

    def test_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Without best model at end
            trainer = get_regression_trainer(output_dir=tmp_dir, save_total_limit=2)
            self.check_checkpoint_deletion(trainer, tmp_dir, [20, 25])

            # With best model at end
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=2
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

            # Edge case: we don't always honor save_total_limit=1 if load_best_model_at_end=True to be able to resume
            # from checkpoint
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=1
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-25")
            self.check_checkpoint_deletion(trainer, tmp_dir, [25])

            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

    def check_mem_metrics(self, trainer, check_func):
        metrics = trainer.train().metrics
        check_func("init_mem_cpu_alloc_delta", metrics)
        check_func("train_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("init_mem_gpu_alloc_delta", metrics)
            check_func("train_mem_gpu_alloc_delta", metrics)

        metrics = trainer.evaluate()
        check_func("eval_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("eval_mem_gpu_alloc_delta", metrics)

        metrics = trainer.predict(RegressionDataset()).metrics
        check_func("test_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("test_mem_gpu_alloc_delta", metrics)

    # def test_mem_metrics(self):
    #     # with mem metrics enabled
    #     trainer = get_regression_trainer(skip_memory_metrics=False)
    #     self.check_mem_metrics(trainer, self.assertIn)

    #     # with mem metrics disabled
    #     trainer = get_regression_trainer(skip_memory_metrics=True)
    #     self.check_mem_metrics(trainer, self.assertNotIn)

    @pytest.mark.skip('skip_memory_metrics not support')
    def test_fp16_full_eval(self):
        # this is a sensitive test so let's keep debugging printouts in place for quick diagnosis.
        # it's using pretty large safety margins, but small enough to detect broken functionality.
        debug = 0
        n_gpus = 1

        bs = 8
        eval_len = 16 * n_gpus
        # make the params somewhat big so that there will be enough RAM consumed to be able to
        # measure things. We should get about 64KB for a+b in fp32
        a = np.ones((1000, bs), np.float32) + 0.001
        b = np.ones((1000, bs), np.float32) - 0.001

        # 1. with fp16_full_eval disabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        del trainer
        gc.collect()

        fp32_init = metrics["init_mem_gpu_alloc_delta"]
        fp32_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"fp32_init {fp32_init}")
            print(f"fp32_eval {fp32_eval}")

        # here we expect the model to be preloaded in trainer.__init__ and consume around 64K gpu ram.
        # perfect world: fp32_init == 64<<10
        self.assertGreater(fp32_init, 59_000)
        # after eval should be no extra memory allocated - with a small margin (other than the peak
        # memory consumption for the forward calculation that gets recovered)
        # perfect world: fp32_eval == close to zero
        self.assertLess(fp32_eval, 5_000)

        # 2. with fp16_full_eval enabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, fp16_full_eval=True, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        fp16_init = metrics["init_mem_gpu_alloc_delta"]
        fp16_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"fp16_init {fp16_init}")
            print(f"fp16_eval {fp16_eval}")

        # here we expect the model to not be preloaded in trainer.__init__, so with a small margin it should be close to 0
        # perfect world: fp16_init == close to zero
        self.assertLess(fp16_init, 5_000)
        # here we put the model on device in eval and only `half()` of it, i.e. about 32K,(again we ignore the peak margin which gets returned back)
        # perfect world: fp32_init == 32<<10
        self.assertGreater(fp16_eval, 27_000)

        # 3. relative comparison fp32 vs full fp16
        # should be about half of fp16_init
        # perfect world: fp32_init/2 == fp16_eval
        self.assertAlmostEqual(fp16_eval, fp32_init / 2, delta=5_000)


    @pytest.mark.skip('not support bf16')
    @require_mindspore
    def test_bf16_full_eval(self):
        # note: most of the logic is the same as test_fp16_full_eval

        # this is a sensitive test so let's keep debugging printouts in place for quick diagnosis.
        # it's using pretty large safety margins, but small enough to detect broken functionality.
        debug = 0
        n_gpus = 1

        bs = 8
        eval_len = 16 * n_gpus
        # make the params somewhat big so that there will be enough RAM consumed to be able to
        # measure things. We should get about 64KB for a+b in fp32
        a = ops.ones(1000, bs) + 0.001
        b = ops.ones(1000, bs) - 0.001

        # 1. with bf16_full_eval disabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        del trainer
        gc.collect()

        fp32_init = metrics["init_mem_gpu_alloc_delta"]
        fp32_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"fp32_init {fp32_init}")
            print(f"fp32_eval {fp32_eval}")

        # here we expect the model to be preloaded in trainer.__init__ and consume around 64K gpu ram.
        # perfect world: fp32_init == 64<<10
        self.assertGreater(fp32_init, 59_000)
        # after eval should be no extra memory allocated - with a small margin (other than the peak
        # memory consumption for the forward calculation that gets recovered)
        # perfect world: fp32_eval == close to zero
        self.assertLess(fp32_eval, 5_000)

        # 2. with bf16_full_eval enabled
        trainer = get_regression_trainer(a=a, b=b, eval_len=eval_len, bf16_full_eval=True, skip_memory_metrics=False)
        metrics = trainer.evaluate()
        bf16_init = metrics["init_mem_gpu_alloc_delta"]
        bf16_eval = metrics["eval_mem_gpu_alloc_delta"]

        if debug:
            print(f"bf16_init {bf16_init}")
            print(f"bf16_eval {bf16_eval}")

        # here we expect the model to not be preloaded in trainer.__init__, so with a small margin it should be close to 0
        # perfect world: bf16_init == close to zero
        self.assertLess(bf16_init, 5_000)
        # here we put the model on device in eval and only `half()` of it, i.e. about 32K,(again we ignore the peak margin which gets returned back)
        # perfect world: fp32_init == 32<<10
        self.assertGreater(bf16_eval, 27_000)

        # 3. relative comparison fp32 vs full bf16
        # should be about half of bf16_init
        # perfect world: fp32_init/2 == bf16_eval
        self.assertAlmostEqual(bf16_eval, fp32_init / 2, delta=5_000)

    def test_no_wd_param_group(self):
        model = nn.Sequential(TstLayer(128), nn.ModuleList([TstLayer(128), TstLayer(128)]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=tmp_dir))
            trainer.create_optimizer_and_scheduler(10)
            wd_names = ['0.linear1.weight', '0.linear2.weight', '1.0.linear1.weight', '1.0.linear2.weight', '1.1.linear1.weight', '1.1.linear2.weight']  # fmt: skip
            wd_params = [p for n, p in model.named_parameters() if n in wd_names]
            no_wd_params = [p for n, p in model.named_parameters() if n not in wd_names]
            self.assertListEqual(trainer.optimizer.param_groups[0]["params"], wd_params)
            self.assertListEqual(trainer.optimizer.param_groups[1]["params"], no_wd_params)

optim_test_params = []
if is_mindspore_available():
    default_adam_kwargs = {
        "betas": (TrainingArguments.adam_beta1, TrainingArguments.adam_beta2),
        "eps": TrainingArguments.adam_epsilon,
        "lr": TrainingArguments.learning_rate,
    }

    default_lion_kwargs = {
        "betas": (TrainingArguments.adam_beta1, TrainingArguments.adam_beta2),
        "lr": TrainingArguments.learning_rate,
    }

    default_anyprecision_kwargs = {
        "use_kahan_summation": False,
        "momentum_dtype": mindspore.float32,
        "variance_dtype": mindspore.float32,
        "compensation_buffer_dtype": mindspore.bfloat16,
    }

    optim_test_params = [
        (
            TrainingArguments(optim=OptimizerNames.ADAMW, output_dir="None"),
            mindnlp.core.optim.AdamW,
            default_adam_kwargs,
        ),
        # (
        #     TrainingArguments(optim=OptimizerNames.ADAFACTOR, output_dir="None"),
        #     transformers.optimization.Adafactor,
        #     {
        #         "scale_parameter": False,
        #         "relative_step": False,
        #         "lr": TrainingArguments.learning_rate,
        #     },
        # ),
    ]


@require_mindspore
class TrainerOptimizerChoiceTest(unittest.TestCase):
    def check_optim_and_kwargs(self, training_args: TrainingArguments, expected_cls, expected_kwargs):
        actual_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
        self.assertEqual(expected_cls, actual_cls)
        self.assertIsNotNone(optim_kwargs)

        for p, v in expected_kwargs.items():
            self.assertTrue(p in optim_kwargs)
            actual_v = optim_kwargs[p]
            self.assertTrue(actual_v == v, f"Failed check for {p}. Expected {v}, but got {actual_v}.")

    @parameterized.expand(optim_test_params, skip_on_empty=True)
    def test_optim_supported(self, training_args: TrainingArguments, expected_cls, expected_kwargs):
        # exercises all the valid --optim options
        self.check_optim_and_kwargs(training_args, expected_cls, expected_kwargs)

        trainer = get_regression_trainer(**training_args.to_dict())
        trainer.train()
