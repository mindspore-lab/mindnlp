# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.

"""CLI option handling for merge commands."""

from __future__ import annotations

import functools
import logging
import warnings
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import click

LOG = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = ("safetensors", "ckpt")


def _parse_device_spec(device: str) -> tuple[str, Optional[int]]:
    raw = str(device).strip()
    if ":" in raw:
        target, idx = raw.split(":", 1)
        try:
            return target.strip(), int(idx)
        except ValueError:
            return target.strip(), None
    return raw, None


def _to_mindspore_target(device: str) -> str:
    key = device.strip().lower()
    mapping = {
        "cpu": "CPU",
        "ascend": "Ascend",
        "npu": "Ascend",
        "gpu": "GPU",
        "cuda": "GPU",
    }
    return mapping.get(key, device)


def _parse_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().upper()
    if text.isdigit():
        return int(text)
    units = {
        "KB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "TB": 10**12,
        "K": 10**3,
        "M": 10**6,
        "G": 10**9,
        "T": 10**12,
        "B": 1,
    }
    for suffix, mul in units.items():
        if text.endswith(suffix):
            num = text[: -len(suffix)].strip()
            return int(float(num) * mul)
    return int(float(text))


@dataclass(frozen=True)
class MergeOptions:
    device: str = "CPU"
    strict_device_detect: bool = False
    allow_crimes: bool = False
    transformers_cache: Optional[str] = None
    lora_merge_cache: Optional[str] = None
    lora_merge_dtype: Optional[str] = None
    lazy_loader: bool = False
    trust_remote_code: bool = False
    random_seed: Optional[int] = None
    quiet: bool = False
    verbosity: int = 0

    multi_npu: bool = False
    low_cpu_memory: bool = False
    read_to_npu: bool = False

    out_shard_size: int = 5 * 10**9
    output_format: str = "safetensors"
    safe_serialization: bool = True
    async_write: bool = False
    write_threads: int = 1
    clone_tensors: bool = False
    max_tensor_mem_gb: Optional[float] = None
    split_pieces: int = 1
    ckpt_load_kwargs: Optional[Dict[str, Any]] = None

    copy_tokenizer: bool = True
    write_model_card: bool = True

    def __post_init__(self) -> None:
        if self.max_tensor_mem_gb is not None and self.max_tensor_mem_gb <= 0:
            raise ValueError("max_tensor_mem_gb must be > 0")
        if self.split_pieces < 1:
            raise ValueError("split_pieces must be >= 1")

        resolved_format = self.output_format
        if resolved_format == "safetensors" and not self.safe_serialization:
            warnings.warn(
                "--no-safe-serialization is deprecated. "
                "Use --output-format ckpt instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_format = "bin"

        if resolved_format == "bin":
            raise ValueError(
                "Output format 'bin' (PyTorch pickle) is not supported by "
                "MindSpore Wizard. Use --output-format safetensors (default) "
                "or --output-format ckpt instead."
            )

        if resolved_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{resolved_format}'. "
                f"Must be one of: {', '.join(VALID_OUTPUT_FORMATS)}"
            )
        object.__setattr__(self, "output_format", resolved_format)

        if str(self.device).lower() != "auto":
            return

        from . import common as common_mod

        try:
            detected = common_mod.get_accelerator_type()
            object.__setattr__(self, "device", detected)
        except Exception as exc:
            msg = f"Automatic device detection failed ({type(exc).__name__}: {exc})"
            if self.strict_device_detect:
                raise RuntimeError(msg) from exc
            LOG.warning("%s; falling back to CPU", msg)
            object.__setattr__(self, "device", "CPU")

    def apply_global_options(self) -> None:
        level = logging.WARNING
        if self.quiet:
            level = logging.ERROR
        elif self.verbosity >= 2:
            level = logging.DEBUG
        elif self.verbosity == 1:
            level = logging.INFO
        logging.getLogger().setLevel(level)

        if self.random_seed is not None:
            random.seed(self.random_seed)
            try:
                import numpy as np

                np.random.seed(self.random_seed)
            except Exception:
                pass
            try:
                import mindspore

                mindspore.set_seed(self.random_seed)
            except Exception:
                pass

        try:
            import mindspore

            target, device_id = _parse_device_spec(self.device)
            ms_target = _to_mindspore_target(target)
            mindspore.set_context(device_target=ms_target)
            if device_id is not None and ms_target != "CPU":
                mindspore.set_context(device_id=device_id)
        except Exception as exc:
            msg = (
                f"Failed to apply MindSpore device context for {self.device} "
                f"({type(exc).__name__}: {exc})"
            )
            LOG.warning(msg)
            requested_target, _ = _parse_device_spec(self.device)
            if _to_mindspore_target(requested_target) != "CPU":
                raise RuntimeError(
                    msg
                    + ". Requested accelerator is unavailable; aborting instead "
                    + "of falling back to CPU."
                ) from exc


class PrettyPrintHelp(click.Command):
    """Help command wrapper."""


_MERGE_OPTION_NAMES = [
    "device",
    "strict_device_detect",
    "allow_crimes",
    "transformers_cache",
    "lora_merge_cache",
    "lora_merge_dtype",
    "lazy_loader",
    "trust_remote_code",
    "random_seed",
    "quiet",
    "verbosity",
    "multi_npu",
    "low_cpu_memory",
    "read_to_npu",
    "out_shard_size",
    "output_format",
    "safe_serialization",
    "async_write",
    "write_threads",
    "clone_tensors",
    "max_tensor_mem_gb",
    "split_pieces",
    "copy_tokenizer",
    "write_model_card",
]


def _lazy_loader_from_cli(ctx, param, value):
    """Accept both --lazy-loader and deprecated --lazy-unpickle."""
    return value


def add_merge_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Attach shared merge CLI options and inject `merge_options`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        option_kwargs: Dict[str, Any] = {}

        # Handle deprecated --lazy-unpickle → lazy_loader
        if "lazy_unpickle" in kwargs:
            val = kwargs.pop("lazy_unpickle")
            if val:
                warnings.warn(
                    "--lazy-unpickle is deprecated. Use --lazy-loader instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            option_kwargs["lazy_loader"] = kwargs.pop("lazy_loader", False) or val
        for name in _MERGE_OPTION_NAMES:
            if name in kwargs:
                option_kwargs[name] = kwargs.pop(name)
        option_kwargs["out_shard_size"] = _parse_size(option_kwargs["out_shard_size"])
        kwargs["merge_options"] = MergeOptions(**option_kwargs)
        return func(*args, **kwargs)

    opts = [
        click.option("--device", default="CPU", show_default=True),
        click.option(
            "--strict-device-detect/--no-strict-device-detect",
            "strict_device_detect",
            default=False,
        ),
        click.option("--allow-crimes/--no-allow-crimes", "allow_crimes", default=False),
        click.option("--transformers-cache", default=None),
        click.option("--lora-merge-cache", default=None),
        click.option("--lora-merge-dtype", default=None),
        click.option("--lazy-loader/--no-lazy-loader", "lazy_loader", default=False),
        click.option(
            "--lazy-unpickle/--no-lazy-unpickle",
            "lazy_unpickle",
            default=False,
            hidden=True,
        ),
        click.option("--trust-remote-code/--no-trust-remote-code", "trust_remote_code", default=False),
        click.option("--random-seed", type=int, default=None),
        click.option("-q", "--quiet/--no-quiet", default=False),
        click.option("-v", "--verbose", "verbosity", count=True),
        click.option("--multi-npu/--no-multi-npu", "multi_npu", default=False),
        click.option("--low-cpu-memory/--no-low-cpu-memory", "low_cpu_memory", default=False),
        click.option("--read-to-npu/--no-read-to-npu", "read_to_npu", default=False),
        click.option("--out-shard-size", default="5G", show_default=True),
        click.option(
            "--output-format",
            "output_format",
            type=click.Choice(VALID_OUTPUT_FORMATS, case_sensitive=False),
            default="safetensors",
            show_default=True,
            help="Tensor output format.",
        ),
        click.option(
            "--safe-serialization/--no-safe-serialization",
            "safe_serialization",
            default=True,
            hidden=True,
        ),
        click.option("--async-write/--no-async-write", "async_write", default=False),
        click.option("--write-threads", type=int, default=1, show_default=True),
        click.option("--clone-tensors/--no-clone-tensors", "clone_tensors", default=False),
        click.option("--max-tensor-mem-gb", type=float, default=None),
        click.option("--split-pieces", type=int, default=1, show_default=True),
        click.option("--copy-tokenizer/--no-copy-tokenizer", "copy_tokenizer", default=True),
        click.option("--write-model-card/--no-write-model-card", "write_model_card", default=True),
    ]
    for opt in reversed(opts):
        wrapper = opt(wrapper)
    return wrapper


__all__ = ["MergeOptions", "PrettyPrintHelp", "add_merge_options", "VALID_OUTPUT_FORMATS"]
