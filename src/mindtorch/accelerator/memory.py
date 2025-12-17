from collections import OrderedDict
from typing import Any

import mindspore
import mindtorch

from ._utils import _device_t, _get_device_index


__all__ = [
    "empty_cache",
    "get_memory_info",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other application.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if mindspore.get_context('device_target') == 'CPU':
        return
    mindspore.runtime.empty_cache()


def memory_stats(device_index: _device_t = None, /) -> OrderedDict[str, Any]:
    r"""Return a dictionary of accelerator device memory allocator statistics for a given device index.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from device memory allocation.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of June 2025, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of June 2025, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed device memory allocation calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of device memory allocation calls.
    - ``"num_device_free"``: number of device memory free calls.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        OrderedDict[str, Any]: an ordered dictionary mapping statistic names to their values.
    """
    if mindspore.get_context('device_target') == 'CPU':
        return OrderedDict()
    stats = mindspore.runtime.memory_stats()
    flat_stats = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                nested_prefix = f"{prefix}.{k}" if prefix else k
                flatten(nested_prefix, v)
        else:
            flat_stats.append((prefix, value))

    flatten("", stats)
    flat_stats.sort()
    # pyrefly: ignore [no-matching-overload]
    return OrderedDict(flat_stats)


def memory_allocated(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory occupied by tensors
    in bytes for a given device index.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory occupied by live tensors (in bytes) within the current process.
    """
    return mindspore.runtime.memory_allocated()


def max_memory_allocated(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory occupied by tensors
    in bytes for a given device index.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~mindtorch.accelerator.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory occupied by live tensors (in bytes) within the current process.
    """
    return mindspore.runtime.max_memory_allocated()


def memory_reserved(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory managed by the caching allocator
    in bytes for a given device index.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory reserved by PyTorch (in bytes) within the current process.
    """
    return mindspore.runtime.memory_reserved()


def max_memory_reserved(device_index: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory managed by the caching allocator
    in bytes for a given device index.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~mindtorch.accelerator.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory reserved by PyTorch (in bytes) within the current process.
    """
    return mindspore.runtime.max_memory_reserved()


def reset_accumulated_memory_stats(device_index: _device_t = None, /) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if mindspore.get_context('device_target') == 'CPU':
        return
    mindspore.runtime.reset_max_memory_allocated()
    mindspore.runtime.reset_max_memory_reserved()
    mindspore.runtime.reset_peak_memory_stats()


def reset_peak_memory_stats(device_index: _device_t = None, /) -> None:
    r"""Reset the "peak" stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if mindspore.get_context('device_target') == 'CPU':
        return
    mindspore.runtime.reset_peak_memory_stats()


def _try_initial():
    x = mindspore.Tensor(1)
    _ = mindspore.ops.add(x, 0)
    mindspore.runtime.synchronize()

def get_memory_info(device_index: _device_t = None, /) -> tuple[int, int]:
    r"""Return the current device memory information for a given device index.

    Args:
        device_index (:class:`mindtorch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`mindtorch.accelerator.current_device_index` by default.
            If a :class:`mindtorch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        tuple[int, int]: a tuple of two integers (free_memory, total_memory) in bytes.
            The first value is the free memory on the device (available across all processes and applications),
            The second value is the device's total hardware memory capacity.
    """
    if not isinstance(device_index, int):
        device_index = mindspore.context.get_context("device_id")

    res = mindspore.hal.get_device_properties(device_index)
    if res.total_memory == 0:
        _try_initial()
        res = mindspore.hal.get_device_properties(device_index)

    return (res.free_memory, res.total_memory)