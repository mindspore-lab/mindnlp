# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

"""
Sharded tensor index and lazy loader.

"""

import json
import logging
import os
import os.path
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mindspore
import safetensors

from ._device import move_tensor_to_device
from .loader import TensorLoader

LOG = logging.getLogger(__name__)

_FORMAT_FOR_EXTENSION = {
    ".safetensors": "safetensors",
    ".bin": "bin",
    ".ckpt": "ckpt",
}


def _detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _FORMAT_FOR_EXTENSION.get(ext, "bin")


@dataclass
class ShardInfo:
    filename: str
    contained_keys: List[str]


@dataclass
class ShardedTensorIndex:
    base_path: str
    format: str
    tensor_paths: Dict[str, str]
    shards: List[ShardInfo]

    @property
    def is_safetensors(self) -> bool:
        return self.format == "safetensors"

    @classmethod
    def from_disk(cls, base_path: str) -> "ShardedTensorIndex":
        model_path = None
        for model_file_name in [
            "model.safetensors",
            "mindspore_model.ckpt",
            "pytorch_model.bin",
        ]:
            candidate_path = os.path.join(base_path, model_file_name)
            if os.path.exists(candidate_path) or os.path.exists(
                candidate_path + ".index.json"
            ):
                model_path = candidate_path
                break

        if not model_path:
            ckpt_files = [
                f for f in os.listdir(base_path)
                if f.lower().endswith(".ckpt")
            ] if os.path.isdir(base_path) else []
            if ckpt_files:
                raise RuntimeError(
                    f"Found {len(ckpt_files)} .ckpt file(s) in {base_path} "
                    f"but no recognized entry point (mindspore_model.ckpt or "
                    f"*.index.json). If these are segmented checkpoints from "
                    f"distributed training, please consolidate them first "
                    f"using mindspore.parallel.load_segmented_checkpoints() "
                    f"and re-save as a single checkpoint."
                )
            raise RuntimeError(
                f"Unable to find model files at {base_path}"
            )

        fmt = _detect_format(model_path)
        tensor_paths = None
        shards: List[ShardInfo] = []

        if os.path.exists(model_path + ".index.json"):
            with open(model_path + ".index.json", "r", encoding="utf-8") as fd:
                weight_map = json.load(fd)["weight_map"]
            tensor_paths = weight_map

            shard_names = list(
                sorted(set(tensor_paths[e] for e in tensor_paths))
            )
            for shard_name in shard_names:
                info = ShardInfo(
                    shard_name,
                    [
                        key
                        for key in tensor_paths
                        if tensor_paths[key] == shard_name
                    ],
                )
                shards.append(info)

            return ShardedTensorIndex(
                base_path=base_path,
                format=fmt,
                tensor_paths=tensor_paths,
                shards=shards,
            )
        elif os.path.exists(model_path):
            return ShardedTensorIndex.from_file(model_path)
        else:
            raise RuntimeError(
                f"Unable to find model files at {base_path}"
            )

    @classmethod
    def from_file(cls, file_path: str) -> "ShardedTensorIndex":
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        lower = file_path.lower()
        shard_name = os.path.basename(file_path)
        fmt = _detect_format(file_path)

        if lower.endswith(".safetensors"):
            with safetensors.safe_open(
                file_path, framework="numpy"
            ) as st:
                tensor_paths = {key: shard_name for key in st.keys()}
        elif lower.endswith(".ckpt"):
            from .lazy_ckpt import CkptIndex
            idx = CkptIndex.from_file(file_path)
            tensor_paths = {key: shard_name for key in idx.entries}
        else:
            from .lazy_unpickle import load_bin_lazy
            index = load_bin_lazy(file_path)
            tensor_paths = {key: shard_name for key in index}

        return ShardedTensorIndex(
            base_path=os.path.dirname(file_path),
            format=fmt,
            tensor_paths=tensor_paths,
            shards=[ShardInfo(shard_name, list(tensor_paths.keys()))],
        )


class LazyTensorLoader:
    """Thread-safe loader that opens one shard at a time."""

    index: ShardedTensorIndex
    current_shard: Optional[TensorLoader]
    lazy_loader: bool
    ckpt_load_kwargs: Optional[Dict[str, Any]]
    lock: threading.Lock

    def __init__(
        self,
        index: ShardedTensorIndex,
        lazy_loader: bool = True,
        ckpt_load_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.current_shard = None
        self.lazy_loader = lazy_loader
        self.ckpt_load_kwargs = ckpt_load_kwargs
        self.lock = threading.Lock()

    def get_tensor(
        self,
        key: str,
        device: str = "CPU",
        aliases: Optional[List[str]] = None,
        raise_on_missing: bool = True,
    ) -> Optional[mindspore.Tensor]:
        if aliases and key not in self.index.tensor_paths:
            for alias in aliases:
                if alias in self.index.tensor_paths:
                    key = alias
                    break

        with self.lock:
            if (
                self.current_shard is None
                or key not in self.current_shard.keys()
            ):
                if key not in self.index.tensor_paths:
                    if raise_on_missing:
                        raise KeyError(key)
                    return None

                self.current_shard = None

                shard_file = self.index.tensor_paths[key]
                shard_full_path = os.path.join(
                    self.index.base_path, shard_file
                )
                logging.debug("Opening shard %s", shard_full_path)
                self.current_shard = TensorLoader.get(
                    shard_full_path,
                    use_lazy_loader=self.lazy_loader,
                    device=device,
                    ckpt_load_kwargs=self.ckpt_load_kwargs,
                )

            tensor = self.current_shard.get_tensor(key)
            return move_tensor_to_device(tensor, device, caller="LazyTensorLoader")

    def flush(self):
        with self.lock:
            self.current_shard = None

    @classmethod
    def from_disk(
        cls,
        base_path: str,
        lazy_loader: bool = True,
        ckpt_load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "LazyTensorLoader":
        return LazyTensorLoader(
            ShardedTensorIndex.from_disk(base_path),
            lazy_loader,
            ckpt_load_kwargs=ckpt_load_kwargs,
        )
