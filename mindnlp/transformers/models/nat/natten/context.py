#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
from enum import Enum
from typing import Any, Dict, Optional

import torch
from torch.cuda import _device_t

from .utils import log
from .utils.misc import get_device_cc


logger = log.get_logger(__name__)


class MemoryUsagePreference(Enum):
    Default = 0
    Strict = 1
    Unrestricted = 2


# TODO: handle these with a context?


class NattenContext:
    is_deterministic_mode_enabled: bool = False
    is_fused_na_enabled: bool = False
    is_kv_parallelism_enabled: bool = False

    training_memory_preference: MemoryUsagePreference = MemoryUsagePreference.Default

    @staticmethod
    def reset():
        NattenContext.is_deterministic_mode_enabled = False
        NattenContext.is_fused_na_enabled = False
        NattenContext.is_kv_parallelism_enabled = False
        NattenContext.training_memory_preference = MemoryUsagePreference.Default


def set_memory_usage_preference(pref: str = "default"):
    if pref == "default":
        NattenContext.training_memory_preference = MemoryUsagePreference.Default
    elif pref == "strict":
        NattenContext.training_memory_preference = MemoryUsagePreference.Strict
    elif pref == "unrestricted":
        NattenContext.training_memory_preference = MemoryUsagePreference.Unrestricted
    else:
        raise ValueError(
            "natten.set_memory_usage_preference allows only one of three settings: "
            "`default`, `strict`, and `unrestricted`."
        )


def get_memory_usage_preference() -> MemoryUsagePreference:
    return NattenContext.training_memory_preference


def is_memory_usage_default() -> bool:
    return get_memory_usage_preference() == MemoryUsagePreference.Default


def is_memory_usage_strict() -> bool:
    return get_memory_usage_preference() == MemoryUsagePreference.Strict


def is_memory_usage_unrestricted() -> bool:
    return get_memory_usage_preference() == MemoryUsagePreference.Unrestricted


def use_deterministic_algorithms(mode: bool = True):
    NattenContext.is_deterministic_mode_enabled = mode
    logger.warning(
        "You're enabling NATTEN's deterministic mode. This mode does not "
        "support auto-tuning, or training with positional biases."
    )


def are_deterministic_algorithms_enabled() -> bool:
    return NattenContext.is_deterministic_mode_enabled


def use_kv_parallelism_in_fused_na(mode: bool = True):
    if not mode:
        NattenContext.is_kv_parallelism_enabled = False
        return

    if torch.are_deterministic_algorithms_enabled():
        logger.warning(
            "Attempted to enable KV parallelism in FNA, which is non-deterministic, "
            "but PyTorch's deterministic flag has been enabled. Ignoring..."
        )
        return

    if are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled NATTEN's deterministic mode, but attempted to "
            "enable KV parallelism, which results in non-determinism. "
        )

    NattenContext.is_kv_parallelism_enabled = True
    logger.warning(
        "You're enabling KV parallelism in Fused Neighborhood Attention. "
        "This feature may improve backpropagation latency, but will use some "
        "additional memory, and is non-deterministic. It is not recommended "
        "for memory-limited experiments, or those likely to suffer from "
        "exploding gradients due to non-determinism."
    )


def is_kv_parallelism_in_fused_na_enabled() -> bool:
    return NattenContext.is_kv_parallelism_enabled


def use_fused_na(mode: bool = True, kv_parallel: bool = False):
    if not mode:
        NattenContext.is_fused_na_enabled = False
        use_kv_parallelism_in_fused_na(False)
        return

    logger.info(
        "You're enabling the use of Fused Neighborhood Attention kernels. "
        "You can also consider enabling auto-tuning for potentially improved performance. "
        "Refer to the docs for more information."
    )
    use_kv_parallelism_in_fused_na(kv_parallel)
    NattenContext.is_fused_na_enabled = True


def is_fused_na_enabled() -> bool:
    return NattenContext.is_fused_na_enabled


use_fna = use_fused_na
is_fna_enabled = is_fused_na_enabled


############################################################
# Auto-tuner context
############################################################


class AutotunerContext:
    enabled_for_forward: bool = False
    enabled_for_backward: bool = False

    thorough_mode_forward: bool = False
    thorough_mode_backward: bool = False

    warmup_steps_forward: int = 5
    warmup_steps_backward: int = 5

    steps_forward: int = 5
    steps_backward: int = 5

    _FORWARD_CACHE: Dict[int, Any] = {}
    _BACKWARD_CACHE: Dict[int, Any] = {}

    @staticmethod
    def reset():
        AutotunerContext.enabled_for_forward = False
        AutotunerContext.enabled_for_backward = False

        AutotunerContext.thorough_mode_forward = False
        AutotunerContext.thorough_mode_backward = False

        AutotunerContext.warmup_steps_forward = 5
        AutotunerContext.warmup_steps_backward = 5

        AutotunerContext.steps_forward = 5
        AutotunerContext.steps_backward = 5

        AutotunerContext._FORWARD_CACHE = {}
        AutotunerContext._BACKWARD_CACHE = {}

    @staticmethod
    def set_enabled_for_forward(mode: Optional[bool]):
        if mode is None:
            return
        if not isinstance(mode, bool):
            raise ValueError(f"Expected `bool`, got {type(mode)}.")
        if are_deterministic_algorithms_enabled() and mode:
            raise RuntimeError(
                "You enabled NATTEN's deterministic mode, but attempted to "
                "enable auto-tuning for forward pass, which results in non-determinism. "
            )

        AutotunerContext.enabled_for_forward = mode
        logger.warning(
            "You're enabling NATTEN auto-tuner. This is an experimental "
            "feature intended only for fused neighborhood attention. "
            "Proceed with caution."
        )

    @staticmethod
    def set_enabled_for_backward(mode: Optional[bool]):
        if mode is None:
            return
        if not isinstance(mode, bool):
            raise ValueError(f"Expected `bool`, got {type(mode)}.")
        if are_deterministic_algorithms_enabled() and mode:
            raise RuntimeError(
                "You enabled NATTEN's deterministic mode, but attempted to "
                "enable auto-tuning for backward pass, which results in non-determinism. "
            )

        AutotunerContext.enabled_for_backward = mode
        logger.warning(
            "You're enabling NATTEN auto-tuner for BACKWARD pass. "
            "This is highly experimental and not recommended for distributed "
            "training settings."
        )

    @staticmethod
    def set_thorough_mode_forward(mode: Optional[bool]):
        if mode is None:
            return
        if not isinstance(mode, bool):
            raise ValueError(f"Expected `bool`, got {type(mode)}.")
        AutotunerContext.thorough_mode_forward = mode

    @staticmethod
    def set_thorough_mode_backward(mode: Optional[bool]):
        if mode is None:
            return
        if not isinstance(mode, bool):
            raise ValueError(f"Expected `bool`, got {type(mode)}.")
        AutotunerContext.thorough_mode_backward = mode

    @staticmethod
    def set_warmup_steps_forward(value: Optional[int]):
        if value is None:
            return
        if not isinstance(value, int):
            raise ValueError(f"Expected `int`, got {type(value)}.")
        AutotunerContext.warmup_steps_forward = value

    @staticmethod
    def set_warmup_steps_backward(value: Optional[int]):
        if value is None:
            return
        if not isinstance(value, int):
            raise ValueError(f"Expected `int`, got {type(value)}.")
        AutotunerContext.warmup_steps_backward = value

    @staticmethod
    def set_steps_forward(value: Optional[int]):
        if value is None:
            return
        if not isinstance(value, int):
            raise ValueError(f"Expected `int`, got {type(value)}.")
        AutotunerContext.steps_forward = value

    @staticmethod
    def set_steps_backward(value: Optional[int]):
        if value is None:
            return
        if not isinstance(value, int):
            raise ValueError(f"Expected `int`, got {type(value)}.")
        AutotunerContext.steps_backward = value


def use_autotuner(
    forward_pass: Optional[bool] = None,
    backward_pass: Optional[bool] = None,
    thorough_mode_forward: Optional[bool] = None,
    thorough_mode_backward: Optional[bool] = None,
    warmup_steps_forward: Optional[int] = None,
    warmup_steps_backward: Optional[int] = None,
    steps_forward: Optional[int] = None,
    steps_backward: Optional[int] = None,
):
    if (
        forward_pass is not None
        and backward_pass is not None
        and not forward_pass
        and not backward_pass
    ):
        AutotunerContext.enabled_for_backward = False
        AutotunerContext.enabled_for_forward = False
        return

    if (forward_pass or backward_pass) and torch.are_deterministic_algorithms_enabled():
        logger.warning(
            "Failed to enable NATTEN auto-tuner; PyTorch's deterministic mode "
            "was enabled, and using auto-tuner is non-deterministic."
        )
        return

    AutotunerContext.set_enabled_for_forward(forward_pass)
    AutotunerContext.set_enabled_for_backward(backward_pass)
    AutotunerContext.set_thorough_mode_forward(thorough_mode_forward)
    AutotunerContext.set_thorough_mode_backward(thorough_mode_backward)
    AutotunerContext.set_warmup_steps_forward(warmup_steps_forward)
    AutotunerContext.set_warmup_steps_backward(warmup_steps_backward)
    AutotunerContext.set_steps_forward(steps_forward)
    AutotunerContext.set_steps_backward(steps_backward)


def disable_autotuner():
    AutotunerContext.enabled_for_backward = False
    AutotunerContext.enabled_for_forward = False
    AutotunerContext.thorough_mode_forward = False
    AutotunerContext.thorough_mode_backward = False


def is_autotuner_enabled() -> bool:
    return AutotunerContext.enabled_for_forward or AutotunerContext.enabled_for_backward


def is_autotuner_enabled_for_forward() -> bool:
    return AutotunerContext.enabled_for_forward


def is_autotuner_enabled_for_backward() -> bool:
    return AutotunerContext.enabled_for_backward


def is_autotuner_thorough_for_forward() -> bool:
    return AutotunerContext.thorough_mode_forward


def is_autotuner_thorough_for_backward() -> bool:
    return AutotunerContext.thorough_mode_backward


############################################################
# Backend/lib context
############################################################

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import NATTEN's CPP backend. "
        "This could be due to an invalid/incomplete install. "
        "Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        " correct torch build: shi-labs.com/natten ."
    )


def has_cuda() -> bool:
    return torch.cuda.is_available() and libnatten.has_cuda()


def has_half(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 50


def has_bfloat(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 80


def has_gemm(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 70 and libnatten.has_gemm()


def has_fna(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 50 and libnatten.has_gemm()


has_fused_na = has_fna


def has_tf32_gemm(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 80 and libnatten.has_gemm()


has_fp32_gemm = has_tf32_gemm


def has_fp64_gemm(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 80 and libnatten.has_gemm()


def use_tf32_in_gemm_na(mode: bool = True):
    if mode:
        libnatten.set_gemm_tf32(True)
        return
    libnatten.set_gemm_tf32(False)


def is_tf32_in_gemm_na_enabled() -> bool:
    return libnatten.get_gemm_tf32()


def use_tiled_na(mode: bool = True):
    if mode:
        libnatten.set_tiled_na(True)
        return
    libnatten.set_tiled_na(False)


def is_tiled_na_enabled() -> bool:
    return libnatten.get_tiled_na()


def use_gemm_na(mode: bool = True):
    if mode:
        libnatten.set_gemm_na(True)
        return
    libnatten.set_gemm_na(False)


def is_gemm_na_enabled() -> bool:
    return libnatten.get_gemm_na()


# To be deprecated
def enable_tf32() -> bool:
    use_tf32_in_gemm_na(True)
    return is_tf32_in_gemm_na_enabled()


def disable_tf32() -> bool:
    use_tf32_in_gemm_na(False)
    return is_tf32_in_gemm_na_enabled()


def enable_gemm_na() -> bool:
    use_gemm_na(True)
    return is_gemm_na_enabled()


def disable_gemm_na() -> bool:
    use_gemm_na(False)
    return is_gemm_na_enabled()


def enable_tiled_na() -> bool:
    use_tiled_na(True)
    return is_tiled_na_enabled()


def disable_tiled_na() -> bool:
    use_tiled_na(False)
    return is_tiled_na_enabled()
