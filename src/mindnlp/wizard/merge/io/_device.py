# Copyright (c) MindNLP Wizard contributors.
# Licensed under the Apache License, Version 2.0.

"""Centralised device-movement helpers for tensor I/O.

Every loader / writer that needs to move a tensor to a specific device
should import from here instead of duplicating the logic.
"""

import logging
from typing import Optional

import mindspore

LOG = logging.getLogger(__name__)
_MOVE_WARNED_TARGETS: set = set()


def normalize_device_target(device: Optional[str]) -> Optional[str]:
    """Normalize device spec strings to MindSpore target names (CPU/Ascend/GPU)."""
    if not device:
        return None
    target = str(device).split(":", maxsplit=1)[0].strip().lower()
    mapping = {
        "cpu": "CPU",
        "ascend": "Ascend",
        "gpu": "GPU",
    }
    return mapping.get(target)


def runtime_device_target(device: Optional[str]) -> Optional[str]:
    """Normalize to torch-style device strings used by mindtorch ``Tensor.to()``."""
    if not device:
        return None
    raw = str(device).strip().lower()
    if ":" in raw:
        target, index = raw.split(":", 1)
    else:
        target, index = raw, None
    if target in ("ascend", "npu", "gpu", "cuda"):
        base = "cuda"
    elif target == "cpu":
        base = "cpu"
    else:
        base = target
    if index and base != "cpu":
        return f"{base}:{index}"
    return base


def move_tensor_to_device(
    tensor: mindspore.Tensor,
    device: Optional[str],
    *,
    caller: str = "",
) -> mindspore.Tensor:
    """Best-effort device movement for a MindSpore tensor.

    Parameters
    ----------
    tensor : mindspore.Tensor
    device : str or None
    caller : str
        Free-form label included in warning messages for diagnostics.
    """
    rt_target = runtime_device_target(device)
    ms_target = normalize_device_target(device)
    if not rt_target and not ms_target:
        return tensor

    if rt_target and hasattr(tensor, "to"):
        try:
            return tensor.to(device=rt_target, non_blocking=(rt_target != "cpu"))
        except TypeError:
            try:
                return tensor.to(rt_target)
            except Exception:
                pass
        except Exception:
            pass

    target = ms_target
    if not target:
        return tensor
    try:
        return tensor.move_to(target)
    except Exception as exc:
        if target not in _MOVE_WARNED_TARGETS:
            label = f" ({caller})" if caller else ""
            LOG.warning(
                "Failed to move tensor to %s%s (%s: %s); keeping original tensor",
                target,
                label,
                type(exc).__name__,
                exc,
            )
            _MOVE_WARNED_TARGETS.add(target)
        return tensor
