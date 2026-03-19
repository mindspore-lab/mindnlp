# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

"""
Sharded tensor writer.

Streams tensors to safetensors or MindSpore ckpt shards on disk.
"""

import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import mindspore
import numpy
import safetensors.numpy

from ..dtype_policy import mindspore_to_numpy, numpy_to_mindspore

LOG = logging.getLogger(__name__)

_BASENAME_MAP = {
    "safetensors": "model",
    "ckpt": "mindspore_model",
}


class TensorWriter:
    out_path: str
    override_basename: Optional[str]
    max_shard_size: int
    output_format: str
    use_async: bool

    shards_written: int
    weight_map: Dict[str, str]
    current_shard: Dict[str, mindspore.Tensor]
    current_shard_size: int

    _lock: threading.RLock
    _executor: Optional[ThreadPoolExecutor]
    _write_futures: List[Future]

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        out_path: str,
        max_shard_size: int = 1000 * 1000 * 1000 * 5,
        output_format: str = "safetensors",
        override_basename: Optional[str] = None,
        use_async: bool = False,
        max_write_threads: int = 1,
        *,
        safe_serialization: Optional[bool] = None,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)

        if safe_serialization is not None and output_format == "safetensors":
            output_format = "safetensors" if safe_serialization else "bin"
        if output_format not in ("safetensors", "ckpt"):
            raise ValueError(
                f"Unsupported output_format '{output_format}'. "
                f"Use 'safetensors' or 'ckpt'."
            )

        self.out_path = out_path
        self.override_basename = override_basename
        self.max_shard_size = max_shard_size
        self.output_format = output_format
        self.use_async = use_async

        self.shards_written = 0
        self.weight_map: Dict[str, str] = {}
        self.current_shard: Dict[str, mindspore.Tensor] = {}
        self.current_shard_size = 0
        self.total_size = 0

        self._lock = threading.RLock()
        self._write_futures = []
        if self.use_async:
            self._executor = ThreadPoolExecutor(
                max_workers=max_write_threads
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def save_tensor(
        self, name: str, tensor: mindspore.Tensor, clone: bool = False
    ):
        if clone:
            tensor = numpy_to_mindspore(mindspore_to_numpy(tensor).copy())

        tensor_size = int(tensor.nbytes)

        with self._lock:
            if (
                self.current_shard
                and self.max_shard_size > 0
                and self.current_shard_size + tensor_size
                > self.max_shard_size
            ):
                self._flush_current_shard()

            self.current_shard[name] = tensor
            self.current_shard_size += tensor_size

    def _flush_current_shard(self):
        if not self.current_shard:
            return

        shard_to_write = self.current_shard
        shard_index = self.shards_written

        self.total_size += self.current_shard_size
        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written += 1

        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{shard_index + 1}.{extension}"
        shard_path = os.path.join(self.out_path, shard_name)
        for key in shard_to_write:
            self.weight_map[key] = shard_name

        if self.use_async:
            LOG.info(
                "Dispatching shard #%d to be written.", shard_index + 1,
            )
            future = self._executor.submit(
                self._write_shard_task,
                shard_to_write,
                shard_index,
                shard_path,
            )
            self._write_futures.append(future)
        else:
            self._write_shard_task(
                shard_data=shard_to_write,
                shard_index=shard_index,
                shard_path=shard_path,
            )

    def _write_shard_task(
        self,
        shard_data: Dict[str, mindspore.Tensor],
        shard_index: int,
        shard_path: str,
    ):
        LOG.info("Writing shard #%d...", shard_index + 1)
        if self.output_format == "safetensors":
            self._save_st(shard_data, shard_path)
        elif self.output_format == "ckpt":
            self._save_ckpt(shard_data, shard_path)
        else:
            raise RuntimeError(
                f"Cannot write output format '{self.output_format}'. "
                f"Use 'safetensors' or 'ckpt'."
            )
        LOG.info("Finished writing shard #%d.", shard_index + 1)

    def finalize(self):
        with self._lock:
            self._flush_current_shard()

        if self.use_async:
            if self._write_futures:
                LOG.info(
                    "Waiting for %d shard(s) to finish writing...",
                    len(self._write_futures),
                )
                for future in self._write_futures:
                    future.result()
                LOG.info("All shards have been written to disk.")
                self._write_futures.clear()
            self._executor.shutdown()

        with self._lock:
            LOG.info("Finalizing shard names and creating index file.")
            prefix, extension = self._get_name_components()
            total_shards = self.shards_written

            name_remap: Dict[str, str] = {}
            if total_shards == 1:
                name_remap[f"{prefix}-1.{extension}"] = (
                    f"{prefix}.{extension}"
                )
            else:
                for idx in range(total_shards):
                    old_name = f"{prefix}-{idx + 1}.{extension}"
                    new_name = f"{prefix}-{idx + 1:05d}-of-{total_shards:05d}.{extension}"
                    name_remap[old_name] = new_name

            for old_name, new_name in name_remap.items():
                old_path = os.path.join(self.out_path, old_name)
                new_path = os.path.join(self.out_path, new_name)
                os.rename(old_path, new_path)

            if total_shards > 1:
                for key in self.weight_map:
                    self.weight_map[key] = name_remap.get(
                        self.weight_map[key], self.weight_map[key]
                    )

                index_filename = f"{prefix}.{extension}.index.json"
                index_path = os.path.join(self.out_path, index_filename)
                with open(index_path, "w", encoding="utf-8") as f:
                    content = {
                        "metadata": {
                            "total_size": self.total_size,
                            "mergekit_version": "0.1.4",
                        },
                        "weight_map": self.weight_map,
                    }
                    json.dump(content, f, indent=2)

    def _get_name_components(self):
        if self.override_basename:
            basename = self.override_basename
        else:
            basename = _BASENAME_MAP.get(self.output_format, "model")
        return basename, self.output_format

    def _save_st(self, shard_data: dict, shard_path: str):
        np_data = {}
        for key, tensor in shard_data.items():
            arr = mindspore_to_numpy(tensor)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = numpy.ascontiguousarray(arr)
            np_data[key] = arr

        def _do_save(sd):
            safetensors.numpy.save_file(
                sd, shard_path, metadata={"format": "np"}
            )

        try:
            _do_save(np_data)
        except RuntimeError as e:
            if (
                len(e.args) > 0
                and isinstance(e.args[0], str)
                and "share memory" in e.args[0]
            ):
                LOG.warning(
                    "Duplicated tensors detected — cloning before save."
                )
                np_data = {k: v.copy() for k, v in np_data.items()}
                _do_save(np_data)
            else:
                raise

    def _save_ckpt(self, shard_data: dict, shard_path: str):
        param_list = [
            {"name": key, "data": tensor}
            for key, tensor in shard_data.items()
        ]
        mindspore.save_checkpoint(param_list, shard_path, format="ckpt")
