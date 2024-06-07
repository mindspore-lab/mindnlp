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
from typing import Any, List, Optional, TypeVar

import torch
from torch import Tensor

from ..context import (
    AutotunerContext,
    is_autotuner_enabled,
    is_autotuner_enabled_for_backward,
    is_autotuner_enabled_for_forward,
    is_autotuner_thorough_for_backward,
    is_autotuner_thorough_for_forward,
)

from ..types import CausalArgType, DimensionType
from ..utils import check_all_args, log

from .fna_backward import (
    get_all_tiling_configs_for_fna_backward,
    get_default_tiling_config_for_fna_backward,
    initialize_tensors_for_fna_backward,
    run_fna_backward,
)

from .fna_forward import (
    get_all_tiling_configs_for_fna_forward,
    get_default_tiling_config_for_fna_forward,
    initialize_tensors_for_fna_forward,
    run_fna_forward,
)


logger = log.get_logger(__name__)


def _debug_report(
    shape: torch.Size,
    device: Any,
    dtype: Any,
    kernel_size: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
    is_thorough: bool,
    title: str,
    msg: str,
):
    dtype_to_str = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    if dtype not in dtype_to_str:
        raise RuntimeError(f"NATTEN auto-tuner does not support data type {dtype}.")

    mode_str = "[THOROUGH]" if is_thorough else "[standard]"
    dtype_str = dtype_to_str[dtype]
    device_str = str(device)
    shape_str = "(" + ", ".join([str(x) for x in shape]) + ")"
    logger.debug(
        f"{mode_str} [device: {device_str}] {title}; {dtype_str} input of size "
        f"{shape_str} with {kernel_size=}, {dilation=}, {is_causal=}; "
        f"{msg}"
    )


def _problem_to_hash(
    na_dim: int,
    shape: torch.Size,
    device: Any,
    dtype: Any,
    kernel_size: Any,
    dilation: Any,
    is_causal: Any,
) -> int:
    kernel_size, dilation, is_causal = check_all_args(
        na_dim, kernel_size, dilation, is_causal
    )
    # NOTE: hashing dtype or even its string form is not
    # deterministic, so we'll have to map dtypes to
    # something that is.
    dtype_to_hashable_value_map = {
        torch.float32: 1,
        torch.float16: 2,
        torch.bfloat16: 3,
    }
    if dtype not in dtype_to_hashable_value_map:
        raise RuntimeError(f"NATTEN auto-tuner does not support data type {dtype}.")

    key = hash(
        (
            na_dim,
            hash(shape),
            hash(device),
            hash(dtype_to_hashable_value_map[dtype]),
            hash(kernel_size),
            hash(dilation),
            hash(is_causal),
        )
    )
    return key


ConfigT = TypeVar("ConfigT")


def _benchmark_fn(
    configs: List[ConfigT],
    warmup_steps: int,
    benchmark_steps: int,
    run_fn,
) -> Optional[ConfigT]:
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    best_config = None
    best_time = 1e9
    for config in configs:
        for _ in range(warmup_steps):
            run_fn(config)

        torch.cuda.synchronize()
        starter.record()
        torch.cuda.synchronize()

        for _ in range(benchmark_steps):
            run_fn(config)

        torch.cuda.synchronize()
        ender.record()
        torch.cuda.synchronize()

        time_ms = starter.elapsed_time(ender)

        if time_ms < best_time:
            best_time = time_ms
            best_config = config

    return best_config


def autotune_fna(
    na_dim: int,
    input_tensor: Tensor,
    kernel_size: Any,
    dilation: Any,
    is_causal: Any,
):
    requires_grad = input_tensor.requires_grad
    kernel_size, dilation, is_causal = check_all_args(
        na_dim, kernel_size, dilation, is_causal
    )

    best_forward_config = get_default_tiling_config_for_fna_forward(
        na_dim, input_tensor, dilation
    )
    best_backward_config = get_default_tiling_config_for_fna_backward(
        na_dim, input_tensor, dilation
    )

    if not is_autotuner_enabled() or torch.are_deterministic_algorithms_enabled():
        return best_forward_config, best_backward_config

    assert is_autotuner_enabled_for_forward() or is_autotuner_enabled_for_backward()

    assert input_tensor.dim() == na_dim + 3
    problem_hash = _problem_to_hash(
        na_dim=na_dim,
        shape=input_tensor.shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )

    cache_hit_forward = False
    if (
        is_autotuner_enabled_for_forward()
        and problem_hash in AutotunerContext._FORWARD_CACHE
    ):
        cache_hit_forward = True
        best_forward_config = AutotunerContext._FORWARD_CACHE[problem_hash]

    cache_hit_backward = False
    if (
        is_autotuner_enabled_for_backward()
        and problem_hash in AutotunerContext._BACKWARD_CACHE
    ):
        cache_hit_backward = True
        best_backward_config = AutotunerContext._BACKWARD_CACHE[problem_hash]

    if cache_hit_forward and cache_hit_backward:
        return best_forward_config, best_backward_config

    with torch.no_grad():
        if is_autotuner_enabled_for_forward() and not cache_hit_forward:
            forward_pass_inputs = initialize_tensors_for_fna_forward(
                na_dim, input_tensor
            )

            forward_pass_configs = get_all_tiling_configs_for_fna_forward(
                na_dim, input_tensor, dilation
            )
            _debug_report(
                shape=input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                is_thorough=is_autotuner_thorough_for_forward(),
                title="Benchmarking FNA-forward",
                msg=f"# of configs: {len(forward_pass_configs)}",
            )

            def benchmark_run_fn(config):
                return run_fna_forward(
                    na_dim=na_dim,
                    inputs=forward_pass_inputs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    is_causal=is_causal,
                    tile_config=config,
                )

            best_forward_config = (
                _benchmark_fn(
                    forward_pass_configs,
                    warmup_steps=AutotunerContext.warmup_steps_forward,
                    benchmark_steps=AutotunerContext.steps_forward,
                    run_fn=benchmark_run_fn,
                )
                or best_forward_config
            )
            AutotunerContext._FORWARD_CACHE[problem_hash] = best_forward_config
            _debug_report(
                shape=input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                is_thorough=is_autotuner_thorough_for_forward(),
                title="Successfully benchmarked FNA-forward",
                msg=f"cached config: {best_forward_config}",
            )

        if (
            is_autotuner_enabled_for_backward()
            and not cache_hit_backward
            and requires_grad
        ):
            backward_pass_inputs = initialize_tensors_for_fna_backward(
                na_dim, input_tensor
            )
            backward_pass_configs = get_all_tiling_configs_for_fna_backward(
                na_dim, input_tensor, dilation
            )
            _debug_report(
                shape=input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                is_thorough=is_autotuner_thorough_for_backward(),
                title="Benchmarking FNA-backward",
                msg=f"# of configs: {len(backward_pass_configs)}",
            )

            def benchmark_run_fn(config):
                return run_fna_backward(
                    na_dim=na_dim,
                    inputs=backward_pass_inputs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    is_causal=is_causal,
                    tile_config=config,
                )

            best_backward_config = (
                _benchmark_fn(
                    backward_pass_configs,
                    warmup_steps=AutotunerContext.warmup_steps_backward,
                    benchmark_steps=AutotunerContext.steps_backward,
                    run_fn=benchmark_run_fn,
                )
                or best_backward_config
            )
            AutotunerContext._BACKWARD_CACHE[problem_hash] = best_backward_config
            _debug_report(
                shape=input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                is_thorough=is_autotuner_thorough_for_backward(),
                title="Successfully benchmarked FNA-backward",
                msg=f"cached config: {best_backward_config}",
            )

        return best_forward_config, best_backward_config
