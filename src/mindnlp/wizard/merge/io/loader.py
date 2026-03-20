# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

"""
Tensor loader abstraction and concrete implementations.

``TensorLoader.get()`` is the factory method that picks the right backend
(safetensors / lazy-ckpt / lazy-pickle / eager-pickle / eager-ckpt)
for a given shard file.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional, Sequence

import mindspore
import safetensors

from ..dtype_policy import numpy_to_mindspore
from ._device import move_tensor_to_device
from .lazy_unpickle import (
    DeferredLoad,
    TorchArchiveReader,
    load_bin_lazy,
)

LOG = logging.getLogger(__name__)


class TensorLoader(ABC):
    """Base class for (potentially lazy) tensor loaders."""

    @abstractmethod
    def get_tensor(self, key: str) -> mindspore.Tensor:
        ...

    @abstractmethod
    def keys(self) -> Sequence[str]:
        ...

    @classmethod
    def get(
        cls,
        shard_path: str,
        use_lazy_loader: bool = False,
        device: Optional[str] = None,
        ckpt_load_kwargs: Optional[Dict[str, Any]] = None,
        *,
        use_lazy_unpickle: Optional[bool] = None,
    ) -> "TensorLoader":
        if use_lazy_unpickle is not None and use_lazy_loader is False:
            use_lazy_loader = use_lazy_unpickle

        lower = shard_path.lower()

        if lower.endswith(".safetensors"):
            return SafetensorsLoader(shard_path, device=device)

        if lower.endswith(".ckpt"):
            if use_lazy_loader:
                try:
                    return LazyCkptLoader(shard_path, device=device)
                except Exception as exc:
                    from .lazy_ckpt import CkptFormatNotSupported
                    if isinstance(exc, CkptFormatNotSupported):
                        LOG.warning(
                            "Lazy ckpt loading failed for %s (%s); "
                            "falling back to eager loading.",
                            shard_path,
                            exc,
                        )
                    else:
                        raise
            return DumbCkptLoader(
                shard_path,
                device=device,
                ckpt_load_kwargs=ckpt_load_kwargs,
            )

        if use_lazy_loader:
            return LazyPickleLoader(shard_path, device=device)
        return DumbPytorchLoader(shard_path, device=device)


class SafetensorsLoader(TensorLoader):
    """Load tensors from a safetensors file via numpy → MindSpore."""

    def __init__(self, path: str, device: Optional[str] = None):
        self._handle = safetensors.safe_open(path, framework="numpy")
        self._keys = list(self._handle.keys())
        self._device = device

    def get_tensor(self, key: str) -> mindspore.Tensor:
        arr = self._handle.get_tensor(key)
        tensor = numpy_to_mindspore(arr)
        return move_tensor_to_device(tensor, self._device, caller="SafetensorsLoader")

    def keys(self) -> Sequence[str]:
        return self._keys


class LazyPickleLoader(TensorLoader):
    """Lazy loader for ``.bin`` files — reads metadata up-front, data on demand."""

    def __init__(self, path: str, device: Optional[str] = None):
        self.zip_reader = TorchArchiveReader(path)
        self.index: Dict[str, DeferredLoad] = load_bin_lazy(path)
        self.device = device

    def get_tensor(self, key: str) -> mindspore.Tensor:
        if key not in self.index:
            raise KeyError(key)
        return self.index[key].execute(
            self.zip_reader, map_location=self.device
        )

    def keys(self) -> Sequence[str]:
        return list(self.index.keys())


class DumbPytorchLoader(TensorLoader):
    """Eager full-load of a ``.bin`` file — highest memory, best compatibility."""

    def __init__(self, path: str, device: Optional[str] = None):
        self.zip_reader = TorchArchiveReader(path)
        index = load_bin_lazy(path)
        self.tensors: Dict[str, mindspore.Tensor] = {
            key: dl.execute(self.zip_reader, map_location=device)
            for key, dl in index.items()
        }

    def get_tensor(self, key: str) -> mindspore.Tensor:
        return self.tensors[key]

    def keys(self) -> Sequence[str]:
        return list(self.tensors.keys())


class LazyCkptLoader(TensorLoader):
    """B-level lazy loader for ``.ckpt`` files.

    Scans protobuf wire format once to build an index, then reads
    only the requested tensor bytes on demand.
    """

    def __init__(self, path: str, device: Optional[str] = None):
        from .lazy_ckpt import CkptIndex
        self._index = CkptIndex.from_file(path)
        self._device = device

    def get_tensor(self, key: str) -> mindspore.Tensor:
        arr = self._index.read_tensor(key)
        tensor = numpy_to_mindspore(arr)
        return move_tensor_to_device(tensor, self._device, caller="LazyCkptLoader")

    def keys(self) -> Sequence[str]:
        return list(self._index.entries.keys())


class DumbCkptLoader(TensorLoader):
    """Eager loader for ``.ckpt`` files via ``mindspore.load_checkpoint``.

    Handles encrypted checkpoints, CRC checks, and other advanced
    features that the protobuf wire-format scanner cannot support.
    """

    def __init__(
        self,
        path: str,
        device: Optional[str] = None,
        ckpt_load_kwargs: Optional[Dict[str, Any]] = None,
    ):
        kwargs = dict(ckpt_load_kwargs or {})
        param_dict = mindspore.load_checkpoint(path, **kwargs)
        self.tensors: Dict[str, mindspore.Tensor] = {}
        for key, value in param_dict.items():
            if isinstance(value, (mindspore.Tensor, mindspore.Parameter)):
                t = move_tensor_to_device(value, device, caller="DumbCkptLoader")
                self.tensors[key] = t
            else:
                LOG.debug("Skipping non-tensor entry '%s' (type=%s)", key, type(value).__name__)

    def get_tensor(self, key: str) -> mindspore.Tensor:
        return self.tensors[key]

    def keys(self) -> Sequence[str]:
        return list(self.tensors.keys())
