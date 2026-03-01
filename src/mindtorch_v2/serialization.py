"""Torch-compatible serialization helpers for common checkpoint workflows."""

import os
import pickle
from collections import OrderedDict

import numpy as np

from ._creation import tensor as mt_tensor
from ._dtype import from_name as mt_dtype_from_name
from ._tensor import Tensor as MindTensor


def _get_torch_module():
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "mindtorch_v2 serialization requires torch to be installed "
            "for torch-compatible save/load behavior"
        ) from exc
    return torch


def _check_filelike_for_read(f):
    if isinstance(f, (str, os.PathLike)):
        return
    if not hasattr(f, "read"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with a 'read' attribute"
        )


def _check_filelike_for_write(f):
    if isinstance(f, (str, os.PathLike)):
        return
    if not hasattr(f, "write"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with a 'write' attribute"
        )


def _torch_dtype_to_mindtorch_dtype(torch_dtype):
    name = str(torch_dtype).split(".")[-1]
    mt_dtype = mt_dtype_from_name(name)
    if mt_dtype is None:
        raise TypeError(f"unsupported torch dtype for mindtorch_v2 conversion: {torch_dtype}")
    return mt_dtype


def _mindtorch_dtype_to_torch_dtype(mind_dtype, torch_mod):
    name = getattr(mind_dtype, "name", None)
    if not name:
        raise TypeError(f"unsupported mindtorch_v2 dtype for torch conversion: {mind_dtype}")
    torch_dtype = getattr(torch_mod, name, None)
    if torch_dtype is None:
        raise TypeError(f"unsupported mindtorch_v2 dtype for torch conversion: {mind_dtype}")
    return torch_dtype


def _to_mindtorch_tensor(t):
    cpu_t = t.detach().cpu()
    mt_dtype = _torch_dtype_to_mindtorch_dtype(cpu_t.dtype)
    # Use numpy as a stable bridge between torch and mindtorch tensor implementations.
    mt_t = mt_tensor(cpu_t.numpy(), dtype=mt_dtype, device="cpu", requires_grad=bool(t.requires_grad))
    if t.requires_grad:
        mt_t.requires_grad_(True)
    return mt_t


def _to_torch_tensor(t, torch_mod):
    cpu_t = t.detach().to("cpu")
    arr = cpu_t.numpy()
    torch_dtype = _mindtorch_dtype_to_torch_dtype(cpu_t.dtype, torch_mod)
    torch_t = torch_mod.tensor(np.array(arr, copy=True), dtype=torch_dtype, device="cpu")
    if t.requires_grad:
        torch_t.requires_grad_(True)
    return torch_t


def _convert_mapping_values(obj, value_converter):
    items = [(k, value_converter(v)) for k, v in obj.items()]
    if isinstance(obj, OrderedDict):
        return OrderedDict(items)
    return dict(items)


def _convert_torch_to_mindtorch(obj, torch_mod):
    if isinstance(obj, torch_mod.Tensor):
        return _to_mindtorch_tensor(obj)
    if isinstance(obj, dict):
        return _convert_mapping_values(obj, lambda v: _convert_torch_to_mindtorch(v, torch_mod))
    if isinstance(obj, list):
        return [_convert_torch_to_mindtorch(v, torch_mod) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_torch_to_mindtorch(v, torch_mod) for v in obj)
    if isinstance(obj, set):
        return {_convert_torch_to_mindtorch(v, torch_mod) for v in obj}
    return obj


def _convert_mindtorch_to_torch(obj, torch_mod):
    if isinstance(obj, MindTensor):
        return _to_torch_tensor(obj, torch_mod)
    if isinstance(obj, dict):
        return _convert_mapping_values(obj, lambda v: _convert_mindtorch_to_torch(v, torch_mod))
    if isinstance(obj, list):
        return [_convert_mindtorch_to_torch(v, torch_mod) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_mindtorch_to_torch(v, torch_mod) for v in obj)
    if isinstance(obj, set):
        return {_convert_mindtorch_to_torch(v, torch_mod) for v in obj}
    return obj


def save(obj, f, pickle_module=pickle, pickle_protocol=2, **kwargs):
    _check_filelike_for_write(f)
    torch_mod = _get_torch_module()
    torch_obj = _convert_mindtorch_to_torch(obj, torch_mod)
    return torch_mod.save(
        torch_obj,
        f,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        **kwargs,
    )


def load(f, map_location=None, pickle_module=pickle, *, weights_only=False, **kwargs):
    _check_filelike_for_read(f)
    torch_mod = _get_torch_module()
    loaded = torch_mod.load(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=weights_only,
        **kwargs,
    )
    return _convert_torch_to_mindtorch(loaded, torch_mod)
