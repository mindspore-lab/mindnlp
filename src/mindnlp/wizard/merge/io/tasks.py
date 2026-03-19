# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.

"""IO-related tasks for the DAG execution engine."""

import os
import re
import threading
from typing import Dict, Optional, Tuple

import mindspore

from ..common import ImmutableMap, ModelReference, dtype_from_name
from ..graph import Task
from .lazy_tensor_loader import LazyTensorLoader
from .tensor_writer import TensorWriter


class LoaderCache:
    """Thread-local singleton cache for :class:`LazyTensorLoader` instances."""

    loaders: Dict[ModelReference, LazyTensorLoader]
    lora_cache_dir: Optional[str]
    hf_cache_dir: Optional[str]
    lazy_loader: bool
    trust_remote_code: bool
    lora_merge_dtype: Optional[str]

    _instance = threading.local()

    def __new__(cls) -> "LoaderCache":
        if not hasattr(cls._instance, "value"):
            cls._instance.value = super(LoaderCache, cls).__new__(cls)
            cls._instance.value.loaders = {}
            cls._instance.value.lora_cache_dir = None
            cls._instance.value.hf_cache_dir = None
            cls._instance.value.lazy_loader = False
            cls._instance.value.trust_remote_code = False
            cls._instance.value.lora_merge_dtype = None
        return cls._instance.value

    def get(self, model: ModelReference) -> LazyTensorLoader:
        if model not in self.loaders:
            merged = model.merged(
                cache_dir=self.lora_cache_dir,
                trust_remote_code=self.trust_remote_code,
                lora_merge_dtype=self.lora_merge_dtype,
            )
            self.loaders[model] = merged.lazy_loader(
                cache_dir=self.hf_cache_dir,
                lazy_loader=self.lazy_loader,
            )
        return self.loaders[model]

    def flush_all(self):
        for loader in self.loaders.values():
            loader.flush()

    def setup(self, options):
        self.lora_cache_dir = options.lora_merge_cache
        self.hf_cache_dir = options.transformers_cache
        self.lazy_loader = options.lazy_loader
        self.trust_remote_code = options.trust_remote_code
        self.lora_merge_dtype = options.lora_merge_dtype


shard_name_re = re.compile(r"model\-([0-9]+)-of-([0-9]+)")


def _normalized_shard_name(path: str) -> str:
    name, _ext = os.path.splitext(os.path.basename(path))
    name = name.lower().replace("pytorch_model", "model").replace("mindspore_model", "model")
    m = shard_name_re.search(name)
    if m:
        frac = int(m.group(1)) / int(m.group(2))
        name = f"model-{int(frac * 100):03d}pct"
    return name


def _dtype_factor(dtype_name: Optional[str]) -> float:
    if not dtype_name:
        return 1.0
    name = str(dtype_name).lower()
    if "float32" in name or name == "fp32":
        return 2.0
    if "bfloat16" in name or "float16" in name or name in ("bf16", "fp16"):
        return 1.0
    if "int8" in name:
        return 0.5
    return 1.0


class LoadTensor(Task[Optional[mindspore.Tensor]]):
    model: ModelReference
    tensor: str
    dtype: Optional[str] = None
    device: Optional[str] = None
    optional: bool = False
    aliases: Optional[Tuple[str, ...]] = None
    tied_names: Optional[Tuple[str, ...]] = None
    per_gpu: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {}

    def _resolve_name(self, loader: LazyTensorLoader) -> Optional[str]:
        all_names = (
            [self.tensor]
            + list(self.aliases or [])
            + list(self.tied_names or [])
        )
        for name in all_names:
            if name in loader.index.tensor_paths:
                return name
        return None

    def execute(self) -> Optional[mindspore.Tensor]:
        loader = LoaderCache().get(self.model)
        name = self._resolve_name(loader)
        if not name:
            if not self.optional:
                raise RuntimeError(
                    f"Tensor {self.tensor} required but not present in model {self.model}"
                )
            return None

        x = loader.get_tensor(name, device=self.device or "CPU")
        if self.dtype:
            target_dtype = dtype_from_name(self.dtype)
            if target_dtype is not None and target_dtype != x.dtype:
                x = x.astype(target_dtype)
        return x

    def priority(self) -> int:
        return -1000

    def group_label(self) -> Optional[str]:
        loader = LoaderCache().get(self.model)
        tensor_name = self._resolve_name(loader)
        if tensor_name is None:
            return None
        tensor_path = loader.index.tensor_paths.get(tensor_name)
        if tensor_path:
            # Expose shard/file locality to the scheduler.
            return f"{str(self.model)}::{_normalized_shard_name(tensor_path)}"
        return tensor_name

    def duplicate_per_gpu(self):
        return self.per_gpu

    def cost_hint(self):
        return {
            "read": 1.0,
            "bytes_in": float(64 * 1024 * 1024),
            "dtype_factor": _dtype_factor(self.dtype),
            "fanout": 1.0,
        }


class GatherTensors(Task[Dict[ModelReference, mindspore.Tensor]]):
    weight_info: ImmutableMap
    dtype: Optional[str] = None
    device: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {
            f"{str(model)}:{wi.name}": LoadTensor(
                model=model,
                tensor=wi.name,
                dtype=wi.force_dtype or self.dtype,
                device=self.device,
                optional=wi.optional,
                aliases=wi.aliases,
                tied_names=wi.tied_names,
            )
            for (model, wi) in self.weight_info.items()
        }

    def group_label(self) -> Optional[str]:
        return max(
            t.group_label() or "" for t in self.arguments().values()
        )

    def priority(self) -> int:
        return -10

    def cost_hint(self):
        fan_in = max(1, len(self.weight_info))
        dtype_name = self.dtype
        return {
            "read": float(fan_in),
            "compute": 0.25,
            "bytes_in": float(fan_in * 64 * 1024 * 1024),
            "dtype_factor": _dtype_factor(dtype_name),
            "fanout": float(fan_in),
        }

    def execute(
        self, **kwargs
    ) -> Dict[ModelReference, mindspore.Tensor]:
        key2model = {
            f"{str(model)}:{wi.name}": model
            for (model, wi) in self.weight_info.items()
        }
        return {
            key2model[key]: kwargs[key]
            for key in key2model
            if kwargs[key] is not None
        }


class TensorWriterTask(Task[TensorWriter]):
    out_path: str
    max_shard_size: int
    output_format: str = "safetensors"
    override_basename: Optional[str] = None
    use_async: bool = False
    write_threads: int = 1

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **_kwargs) -> TensorWriter:
        return TensorWriter(
            self.out_path,
            max_shard_size=self.max_shard_size,
            output_format=self.output_format,
            override_basename=self.override_basename,
            use_async=self.use_async,
            max_write_threads=self.write_threads,
        )

    def priority(self):
        return 10000

    def main_thread_only(self):
        return True

    def cost_hint(self):
        return {"write": 0.5, "bytes_out": float(16 * 1024 * 1024)}


class SaveTensor(Task[None]):
    tensor_name: str
    tensor_task: Task
    writer_task: TensorWriterTask
    clone: bool
    optional: bool = False
    dtype: Optional[str] = None
    force_main_thread: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {"writer": self.writer_task, "tensor": self.tensor_task}

    def priority(self) -> int:
        return 1000

    def group_label(self) -> Optional[str]:
        return self.tensor_task.group_label()

    def main_thread_only(self):
        return self.force_main_thread

    def cost_hint(self):
        return {
            "write": 1.0,
            "bytes_out": float(64 * 1024 * 1024),
            "dtype_factor": _dtype_factor(self.dtype),
            "fanout": 1.0,
        }

    def execute(
        self,
        writer: TensorWriter,
        tensor: Optional[mindspore.Tensor],
    ) -> None:
        if tensor is None:
            if not self.optional:
                raise RuntimeError(
                    f"No value for required tensor {self.tensor_name}"
                )
            return
        if self.dtype:
            target = dtype_from_name(self.dtype)
            if target is not None and target != tensor.dtype:
                tensor = tensor.astype(target)
        writer.save_tensor(
            name=self.tensor_name, tensor=tensor, clone=self.clone
        )


class FinalizeModel(Task[None]):
    tensor_save_tasks: Tuple[Task, ...]
    writer_task: TensorWriterTask

    def arguments(self) -> Dict[str, Task]:
        return {
            "writer": self.writer_task,
            **{
                f"_unused_{idx}": t
                for idx, t in enumerate(self.tensor_save_tasks)
            },
        }

    def execute(self, writer: TensorWriter, **kwargs) -> None:
        writer.finalize()

    def main_thread_only(self):
        return True

    def cost_hint(self):
        return {"write": 0.3, "bytes_out": float(8 * 1024 * 1024)}


class ReturnTensor(Task[mindspore.Tensor]):
    weight_info: object  # WeightInfo — resolved at import time
    tensor_task: Task[mindspore.Tensor]
    dtype: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {"tensor": self.tensor_task}

    def priority(self) -> int:
        return 10000

    def group_label(self) -> Optional[str]:
        return self.tensor_task.group_label()

    def cost_hint(self):
        return {"compute": 0.1, "bytes_in": float(8 * 1024 * 1024)}

    def execute(self, tensor: mindspore.Tensor) -> mindspore.Tensor:
        if self.dtype:
            target = dtype_from_name(self.dtype)
            if target is not None and target != tensor.dtype:
                tensor = tensor.astype(target)
        return tensor
