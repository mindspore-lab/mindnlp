# Originally from MergeKit (https://github.com/arcee-ai/mergekit)
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.

"""Computational graph execution engine."""

import logging
import os
import resource
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import mindspore
import networkx
import tqdm
from pydantic import BaseModel, ConfigDict
from typing_extensions import Generic, TypeVar

ValueT = TypeVar("ValueT")
LOG = logging.getLogger(__name__)


class Task(ABC, BaseModel, Generic[ValueT], frozen=True):
    """Abstract base class representing a task in a computational graph.

    Subclasses must implement ``arguments`` and ``execute``.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def arguments(self) -> Dict[str, "Task"]:
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ValueT:
        ...

    def priority(self) -> int:
        return 0

    def group_label(self) -> Optional[str]:
        return None

    def uses_accelerator(self) -> bool:
        return False

    def main_thread_only(self) -> bool:
        return False

    def duplicate_per_gpu(self) -> bool:
        return False

    def cost_hint(self) -> Optional[Dict[str, float]]:
        return None


class TaskUniverse:
    """Container for tasks and their dependency relationships."""

    tasks: List[Task]
    task_to_index: Dict[Task, int]
    task_arguments: Dict[int, Dict[str, int]]
    _type_id_to_index: Dict[Tuple[type, int], int]

    def __init__(self, tasks: Optional[Iterable[Task]] = None):
        self.tasks = []
        self.task_to_index = {}
        self.task_arguments = {}
        self._type_id_to_index = {}
        if tasks is not None:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: Task, recursive: bool = True) -> "TaskHandle":
        _ti_key = (type(task), id(task))
        if _ti_key in self._type_id_to_index:
            index = self._type_id_to_index[_ti_key]
            assert (
                self.tasks[index] == task
            ), "Task modified after being added to universe"
            return TaskHandle(self, index)

        index = self.task_to_index.setdefault(task, len(self.tasks))
        if index < len(self.tasks):
            return TaskHandle(self, index)
        self.tasks.append(task)
        self._type_id_to_index[_ti_key] = index

        if recursive:
            self.task_arguments[index] = {}
            for k, v in task.arguments().items():
                self.task_arguments[index][k] = self.add_task(
                    v, recursive=True
                )._index
        return TaskHandle(self, index)

    def get_handle(self, task: Task) -> Optional["TaskHandle"]:
        if task not in self.task_to_index:
            return None
        return TaskHandle(self, self.task_to_index[task])


class TaskHandle:
    """Lightweight reference to a task within a :class:`TaskUniverse`."""

    __slots__ = ["_universe", "_index"]
    _universe: TaskUniverse
    _index: int

    def __init__(self, universe: TaskUniverse, index: int):
        self._universe = universe
        self._index = index

    def task(self) -> Task:
        return self._universe.tasks[self._index]

    def arguments(self) -> Dict[str, "TaskHandle"]:
        return {
            k: TaskHandle(self._universe, v)
            for k, v in self._universe.task_arguments[self._index].items()
        }

    def __eq__(self, other):
        if not isinstance(other, TaskHandle):
            return False
        if self._index != other._index:
            return False
        if self._universe is not other._universe:
            return False
        return True

    def __hash__(self):
        return self._index

    def __str__(self):
        return f"TaskHandle({type(self.task()).__name__}, {self._index})"

    __repr__ = __str__


class ExecutionSchedule:
    """Ordered schedule of tasks with lifecycle annotations."""

    tasks: List[TaskHandle]
    last_use_index: Dict[TaskHandle, int]

    def __init__(
        self, tasks: List[TaskHandle], last_use_index: Dict[TaskHandle, int]
    ):
        self.tasks = tasks
        self.last_use_index = last_use_index


def build_schedule(
    targets: List[TaskHandle],
    cached_values: Dict[TaskHandle, Any],
) -> ExecutionSchedule:
    """Build a topologically-sorted execution schedule."""
    if not targets:
        return ExecutionSchedule(tasks=[], last_use_index={})

    universe = targets[0]._universe
    assert all(
        t._universe is universe for t in targets
    ), "All tasks must be from the same universe"

    dummy_handle = TaskHandle(universe, -1)
    edge_tups: List[Tuple[TaskHandle, TaskHandle]] = []

    explored: set = set()
    to_explore: set = set(targets)
    while to_explore:
        task = to_explore.pop()
        if task in explored:
            continue
        explored.add(task)
        if task in (cached_values or {}):
            continue
        for dep in task.arguments().values():
            to_explore.add(dep)
            edge_tups.append((dep, task))

    for target in targets:
        edge_tups.append((dummy_handle, target))

    def _compare_key(node: TaskHandle) -> Tuple[str, int]:
        if node._index < 0:
            return ("", 0)
        task = node.task()
        return (
            task.group_label() or "",
            -task.priority(),
        )

    graph = networkx.DiGraph(edge_tups)
    schedule: List[TaskHandle] = [
        node
        for node in networkx.lexicographical_topological_sort(
            graph, key=_compare_key
        )
        if (node != dummy_handle) and node not in (cached_values or {})
    ]

    last_use_index: Dict[TaskHandle, int] = {}
    for idx, task in reversed(list(enumerate(schedule))):
        for dep in task.arguments().values():
            if dep not in last_use_index:
                last_use_index[dep] = idx
        if task not in last_use_index:
            last_use_index[task] = idx
    for task in cached_values or {}:
        if task not in last_use_index:
            last_use_index[task] = len(schedule) + 1

    return ExecutionSchedule(tasks=schedule, last_use_index=last_use_index)


# ---------------------------------------------------------------------------
# Device helpers for MindSpore
# ---------------------------------------------------------------------------

def _parse_device(spec: str) -> Tuple[str, Optional[int]]:
    """Parse a device specifier like ``"Ascend:0"`` or ``"CPU"``.

    Returns ``(target, device_id)`` where *target* is one of
    ``"CPU"``, ``"Ascend"``, ``"GPU"`` and *device_id* may be ``None``.
    """
    parts = str(spec).split(":")
    target = parts[0]
    device_id = int(parts[1]) if len(parts) > 1 else None
    return target, device_id


def _runtime_device_target(spec: str) -> str:
    """Map user device specs to runtime strings used by mindtorch Tensor.to()."""
    target, device_id = _parse_device(spec)
    key = str(target).strip().lower()
    if key in ("ascend", "npu", "gpu", "cuda"):
        base = "cuda"
    elif key == "cpu":
        base = "cpu"
    else:
        base = key
    if device_id is not None and base != "cpu":
        return f"{base}:{device_id}"
    return base


def _mindspore_device_target(spec: str) -> str:
    """Map user device specs to MindSpore move_to targets."""
    target, _ = _parse_device(spec)
    key = str(target).strip().lower()
    mapping = {
        "cpu": "CPU",
        "ascend": "Ascend",
        "npu": "Ascend",
        "gpu": "GPU",
        "cuda": "GPU",
    }
    return mapping.get(key, str(target))


class Executor:
    """Schedule and execute a DAG of tasks.

    ``math_device`` / ``storage_device`` are device specifier strings such as
    ``"CPU"`` or ``"Ascend:0"``.
    """

    math_device: str
    storage_device: str
    universe: TaskUniverse
    targets: List[TaskHandle]
    schedule: ExecutionSchedule
    cached_values: Optional[Dict[TaskHandle, Any]]

    def __init__(
        self,
        targets: Union[List[Task], List[TaskHandle]],
        math_device: str = "CPU",
        storage_device: str = "CPU",
        cached_values: Optional[Dict[TaskHandle, Any]] = None,
    ):
        self.cached_values = cached_values
        self.math_device = math_device
        self.storage_device = storage_device
        self._task_metrics: List[Dict[str, Any]] = []

        if targets and isinstance(targets[0], Task):
            universe = TaskUniverse(targets)
            targets = [universe.add_task(t) for t in targets]
        elif targets and isinstance(targets[0], TaskHandle):
            universe = targets[0]._universe
        elif not targets:
            universe = TaskUniverse()
        else:
            raise ValueError(
                "Targets must be a list of Task or TaskHandle instances"
            )

        self.universe = universe
        self.targets = targets
        self.schedule = build_schedule(
            targets,
            cached_values=cached_values,
        )

    # ------------------------------------------------------------------

    def _run(
        self,
        quiet: bool = False,
        desc: Optional[str] = None,
    ) -> Iterator[Tuple[TaskHandle, Any]]:
        last_use_index = self.schedule.last_use_index

        values: Dict[TaskHandle, Any] = {}
        if self.cached_values:
            for task, value in self.cached_values.items():
                values[task] = value

        for idx, task_handle in (
            pbar := tqdm.tqdm(
                list(enumerate(self.schedule.tasks)),
                disable=quiet,
                desc=desc or "Executing graph",
            )
        ):
            task = task_handle.task()
            use_math_device = task.uses_accelerator()

            arguments: Dict[str, Any] = {}
            for name, dep_handle in task_handle.arguments().items():
                value = values[dep_handle]
                if use_math_device:
                    value = self._move_tensors(value, self.math_device)
                arguments[name] = value
                del value

            start_t = time.perf_counter()
            res = task.execute(**arguments)
            elapsed = time.perf_counter() - start_t
            self._task_metrics.append(
                {
                    "task": type(task).__name__,
                    "wait_ms": 0.0,
                    "run_ms": elapsed * 1000.0,
                }
            )
            del arguments
            res = self._move_tensors(res, self.storage_device)

            values[task_handle] = res
            del res

            if task_handle in self.targets:
                yield (task_handle, values[task_handle])

            expired = [
                key for key in values if idx >= last_use_index[key]
            ]
            for key in expired:
                del values[key]

        del values
        del pbar

    def run(
        self,
        quiet: bool = False,
        desc: Optional[str] = None,
    ) -> Iterator[Tuple[Task, Any]]:
        for handle, value in self._run(quiet=quiet, desc=desc):
            yield (handle.task(), value)

    def execute(self, desc: Optional[str] = None) -> None:
        for _ in self.run(desc=desc):
            pass

    def metrics_snapshot(self) -> Dict[str, Any]:
        rss_mb = 0.0
        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports ru_maxrss in KB.
            rss_mb = float(rss_kb) / 1024.0
        except Exception:
            pass
        return {
            "executor": "single_device",
            "pid": os.getpid(),
            "task_count": len(self._task_metrics),
            "tasks": list(self._task_metrics),
            "queue_depth_samples": [],
            "backpressure_trigger_count": 0,
            "rss_peak_mb": rss_mb,
            "npu_used_peak_mb": None,
        }

    # ------------------------------------------------------------------
    # MindSpore tensor device movement
    # ------------------------------------------------------------------

    @staticmethod
    def _move_tensors(
        value: Any,
        device: str,
        non_blocking: Optional[bool] = None,
    ) -> Any:
        """Recursively move MindSpore tensors to *device*.

        MindSpore's device model is context-based.  When both math and
        storage targets are the same (the common single-NPU case) this is
        effectively a no-op.  Cross-device copies go through NumPy when the
        source and target differ.
        """
        if isinstance(value, mindspore.Tensor):
            runtime_target = _runtime_device_target(device)
            try:
                if hasattr(value, "to"):
                    try:
                        return value.to(
                            device=runtime_target,
                            non_blocking=(runtime_target != "cpu"),
                        )
                    except TypeError:
                        return value.to(runtime_target)
            except Exception as exc:
                LOG.debug(
                    "Tensor.to(%s) failed (%s: %s), falling back to move_to",
                    runtime_target,
                    type(exc).__name__,
                    exc,
                )

            ms_target = _mindspore_device_target(device)
            try:
                return value.move_to(ms_target)
            except Exception as exc:
                LOG.warning(
                    "Failed to move tensor to %s (%s: %s); keeping original tensor",
                    ms_target,
                    type(exc).__name__,
                    exc,
                )
                return value
        elif isinstance(value, dict):
            return {
                k: Executor._move_tensors(v, device, non_blocking)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [
                Executor._move_tensors(v, device, non_blocking) for v in value
            ]
        elif isinstance(value, tuple):
            return tuple(
                Executor._move_tensors(v, device, non_blocking) for v in value
            )
        return value
