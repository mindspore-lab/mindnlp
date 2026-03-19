# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.

"""Public exports for wizard merge package."""

from .config import MergeConfiguration, ConfigReader
from .common import ModelReference, ModelPath, dtype_from_name, ImmutableMap
from .options import MergeOptions
from .graph import Task, Executor, TaskUniverse, TaskHandle
from .merge import run_merge

__all__ = [
    "MergeConfiguration",
    "ConfigReader",
    "ModelReference",
    "ModelPath",
    "MergeOptions",
    "Task",
    "Executor",
    "TaskUniverse",
    "TaskHandle",
    "run_merge",
    "dtype_from_name",
    "ImmutableMap",
]
