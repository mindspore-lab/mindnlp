# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#
# Device strings ("Ascend:0", "CPU") replace torch.device objects.

"""
Implementation of multi-device parallel task execution.

Handles distribution of parallelizable tasks across multiple devices (NPUs)
while respecting:
- Main-thread-only task requirements
- Task dependency graphs
- Device assignment of connected task components
- Intermediate result storage locations
"""

import concurrent.futures
import contextlib
import logging
import os
import queue
import resource
import threading
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx
import tqdm

from .common import (
    get_accelerator_count,
    get_accelerator_type,
)
from .graph import (
    Executor,
    Task,
    TaskHandle,
    TaskUniverse,
    build_schedule,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream helpers — thin wrappers around mindspore.hal.Stream when available
# ---------------------------------------------------------------------------

def _make_stream(device: str):
    """Create a device stream if the runtime supports it, else return *None*."""
    try:
        import mindspore
        return mindspore.hal.Stream(device=device)
    except Exception as exc:
        LOG.debug(
            "Could not create stream for %s (%s: %s); falling back to default stream",
            device,
            type(exc).__name__,
            exc,
        )
        return None


@contextlib.contextmanager
def _stream_context(stream):
    """Context manager that activates *stream* if it is not ``None``."""
    if stream is not None:
        try:
            import mindspore
            with mindspore.hal.StreamCtx(stream):
                yield
                return
        except Exception as exc:
            LOG.debug(
                "Could not enter stream context (%s: %s); using default context",
                type(exc).__name__,
                exc,
            )
    yield


def _synchronize_device(device: str):
    """Block until all operations on *device* have finished."""
    try:
        import mindspore
        mindspore.hal.synchronize(device)
    except Exception as exc:
        LOG.debug(
            "Device synchronization failed for %s (%s: %s)",
            device,
            type(exc).__name__,
            exc,
        )


# ---------------------------------------------------------------------------
# Multi-device executor
# ---------------------------------------------------------------------------

class MultiDeviceExecutor:
    """
    Execute computational tasks in parallel across multiple devices (NPUs).

    This class analyzes the dependency structure of a task graph and distributes
    the workload across available devices while respecting:
    1. Tasks requiring main thread execution
    2. Tasks that need to be duplicated on each device
    3. Task dependencies and data locality
    4. Memory management for intermediate results

    It automatically partitions the task graph into leading tasks (main thread,
    pre-parallel), parallel tasks (distributed across devices), and trailing
    tasks (main thread, post-parallel).

    Attributes:
        num_devices: Number of devices to utilize (None = all available)
        storage_device: Device string for storing tensors between stages
        targets: Final output tasks to retain results for
    """

    def __init__(
        self,
        targets: List[Task],
        num_devices: Optional[int] = None,
        storage_device: Optional[str] = None,
    ):
        """
        Initialize the executor with a list of target tasks.

        This performs initial task graph analysis, including:
        - Finding tasks that must run on the main thread before parallel execution
        - Finding tasks that must run on the main thread after parallel execution
        - Partitioning parallel tasks into islands that can run independently
        - Assigning islands to devices using a load-balancing approach

        Args:
            targets: List of final target tasks to execute
            num_devices: Number of devices to utilize (None = all available)
            storage_device: Device string for storing intermediate results
                            between execution stages (e.g. ``"CPU"``,
                            ``"Ascend:0"``)
        """
        self.results: Dict[TaskHandle, Any] = {}
        self.storage_device = storage_device
        self._metric_lock = threading.Lock()
        self._task_metrics: List[Dict[str, Any]] = []
        self._queue_depth_samples: List[int] = []
        self._assignment_metrics: List[Dict[str, Any]] = []

        self.accelerator_type = get_accelerator_type()
        if num_devices is None:
            num_devices = get_accelerator_count()
        LOG.info(
            "Using %d %s device(s) for parallel execution",
            num_devices, self.accelerator_type,
        )

        self.universe = TaskUniverse(targets)
        self.targets = {self.universe.get_handle(t) for t in targets}
        self.serial_schedule = build_schedule(
            list(self.targets),
            {},
        )
        ordered_handles = self.serial_schedule.tasks

        self.per_device_tasks = {
            t for t in ordered_handles if t.task().duplicate_per_gpu()
        }
        leading_tasks = self._find_leading_tasks(ordered_handles)
        trailing_tasks = self._find_trailing_tasks(ordered_handles)
        self.trailing_main_handles = [
            t for t in ordered_handles if t in trailing_tasks
        ]
        self.leading_main_handles = [
            t for t in ordered_handles if t in leading_tasks
        ]

        self.trailing_dependencies: Set[TaskHandle] = set()
        for task_handle in self.trailing_main_handles:
            self.trailing_dependencies.update(task_handle.arguments().values())

        parallel_handles = [
            t
            for t in ordered_handles
            if (
                t not in trailing_tasks
                and t not in leading_tasks
                and t not in self.per_device_tasks
            )
        ]
        LOG.info(
            "Task breakdown: %d leading, %d duplicated per-device, "
            "%d parallel, %d trailing",
            len(self.leading_main_handles),
            len(self.per_device_tasks),
            len(parallel_handles),
            len(self.trailing_main_handles),
        )
        if any(t.task().main_thread_only() for t in parallel_handles):
            raise RuntimeError(
                "Main-thread-only tasks must be either leading or trailing"
            )
        if any(t.task().main_thread_only() for t in self.per_device_tasks):
            raise RuntimeError(
                "Tasks can not be both per-device and main-thread-only"
            )
        self.device_assignments = self._assign_islands_to_devices(
            parallel_handles, num_devices
        )

        self.task_completion_queue: queue.Queue = queue.Queue()
        self.done_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, quiet: bool = False) -> Iterator[Tuple[Task, Any]]:
        """
        Execute all tasks and yield target results.

        Yields:
            Iterator[Tuple[Task, Any]]: Task and result pairs
        """
        with tqdm.tqdm(
            total=len(self.serial_schedule.tasks),
            disable=quiet,
            desc="Executing graph",
        ) as pbar:
            # Phase 1: leading (main-thread, pre-parallel) tasks
            if self.leading_main_handles:
                exec = Executor(
                    self.leading_main_handles,
                    math_device=self.storage_device or "CPU",
                    storage_device=self.storage_device or "CPU",
                )
                for task_handle, result in exec._run(quiet=True):
                    pbar.update()
                    self.results[task_handle] = result
                with self._metric_lock:
                    self._task_metrics.extend(exec.metrics_snapshot()["tasks"])

            results_snapshot = dict(self.results)

            def update_progress():
                while not self.done_event.is_set():
                    try:
                        task_idx, result = self.task_completion_queue.get(
                            timeout=0.1
                        )
                        task_handle = TaskHandle(self.universe, task_idx)
                        self.results[task_handle] = result
                        pbar.update()
                    except queue.Empty:
                        continue

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            # Phase 2: parallel tasks distributed across devices
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for device, island_task_handles in (
                    self.device_assignments.items()
                ):
                    futures.append(
                        executor.submit(
                            self._device_worker,
                            task_list=(
                                list(self.per_device_tasks)
                                + island_task_handles
                            ),
                            cached_values=results_snapshot,
                            device=device,
                            quiet=True,
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    if ex := future.exception():
                        self.done_event.set()
                        executor.shutdown(wait=False)
                        raise ex

            self.done_event.set()
            progress_thread.join()

            # Phase 3: trailing (main-thread, post-parallel) tasks
            if self.trailing_main_handles:
                exec = Executor(
                    self.trailing_main_handles,
                    math_device=self.storage_device or "CPU",
                    storage_device=self.storage_device or "CPU",
                    cached_values=dict(self.results),
                )
                for task_handle, result in exec._run(quiet=True):
                    pbar.update()
                    if task_handle in self.targets:
                        self.results[task_handle] = result
                with self._metric_lock:
                    self._task_metrics.extend(exec.metrics_snapshot()["tasks"])

        for task_handle, result in self.results.items():
            if task_handle in self.targets:
                yield task_handle.task(), result

    def execute(self) -> None:
        """Execute all tasks and discard results."""
        for _ in self.run(quiet=False):
            pass

    # ------------------------------------------------------------------
    # Task graph analysis helpers
    # ------------------------------------------------------------------

    def _find_trailing_tasks(
        self, tasks: List[TaskHandle]
    ) -> Set[TaskHandle]:
        """
        Identify tasks that must execute AFTER parallel device tasks complete.

        A task is considered "trailing" if:
        - It requires main thread execution (task.main_thread_only() is True)
        - All tasks dependent on it are also trailing tasks (recursive condition)
        - OR it has no dependents (terminal task)

        Args:
            tasks: List of task handles to analyze

        Returns:
            Set[TaskHandle]: Set of tasks that should be executed after
            parallel processing
        """
        dependants: Dict[TaskHandle, Set[TaskHandle]] = defaultdict(set)
        for task_idx, arg_indices in self.universe.task_arguments.items():
            for dep_idx in arg_indices.values():
                dependants[TaskHandle(self.universe, dep_idx)].add(
                    TaskHandle(self.universe, task_idx)
                )

        trailing_tasks: Set[TaskHandle] = set()
        to_explore = {t for t in tasks if not dependants[t]}
        while to_explore:
            task_handle = to_explore.pop()
            task = task_handle.task()
            if not task.main_thread_only():
                continue
            if all(d in trailing_tasks for d in dependants[task_handle]):
                trailing_tasks.add(task_handle)
                to_explore.update(task_handle.arguments().values())
        return trailing_tasks

    def _find_leading_tasks(
        self, tasks: List[TaskHandle]
    ) -> Set[TaskHandle]:
        """
        Identify tasks that must execute BEFORE parallel device tasks.

        A task is considered "leading" if:
        - It requires main thread execution (task.main_thread_only() is True)
        - It has no dependencies, or all its dependencies are also leading tasks

        Args:
            tasks: List of task handles to analyze

        Returns:
            Set[TaskHandle]: Set of tasks that should be executed before
            parallel processing
        """
        leading_tasks: Set[TaskHandle] = set()
        for task_handle in tasks:
            task = task_handle.task()
            if not task.main_thread_only():
                continue
            args = task_handle.arguments()
            if args and any(
                dep not in leading_tasks for dep in args.values()
            ):
                continue
            leading_tasks.add(task_handle)
        return leading_tasks

    # ------------------------------------------------------------------
    # Island assignment & device worker
    # ------------------------------------------------------------------

    def _assign_islands_to_devices(
        self, tasks: List[TaskHandle], num_devices: int
    ) -> Dict[str, List[TaskHandle]]:
        """
        Assign task islands to devices for parallel execution.

        This method partitions the parallel task graph into independent
        subgraphs (islands) that can be executed independently on different
        devices.  It uses a load-balancing approach to distribute islands
        across available devices.

        Task islands are identified as weakly connected components in the
        task dependency graph, meaning groups of tasks that are connected
        through dependencies but don't have dependencies outside their group.

        Args:
            tasks: List of parallel tasks to assign to devices
            num_devices: Number of available devices

        Returns:
            Dict[str, List[TaskHandle]]: Mapping from device strings
            (e.g. ``"Ascend:0"``) to assigned tasks
        """
        task_set = set(tasks)

        edge_list = []
        for task_handle in tasks:
            for dep_handle in task_handle.arguments().values():
                if dep_handle in task_set:
                    edge_list.append(
                        (dep_handle._index, task_handle._index)
                    )

        island_graph = nx.DiGraph()
        island_graph.add_nodes_from([t._index for t in tasks])
        island_graph.add_edges_from(edge_list)
        islands: List[Set[int]] = list(
            nx.weakly_connected_components(island_graph)
        )
        LOG.info("Found %d islands in parallel task graph", len(islands))

        assignments: Dict[str, List[TaskHandle]] = {}
        assignment_metrics: List[Dict[str, Any]] = []
        island_items: List[Tuple[int, int, str, List[TaskHandle]]] = []
        for island in islands:
            if not island:
                continue
            island_tasks = [
                TaskHandle(self.universe, idx) for idx in island
            ]
            key_hist = Counter(
                self._task_locality_key(t)
                for t in island_tasks
                if self._task_locality_key(t) is not None
            )
            dominant_key = ""
            dominant_count = 0
            if key_hist:
                dominant_key, dominant_count = key_hist.most_common(1)[0]
            island_items.append(
                (
                    len(island_tasks),
                    dominant_count,
                    dominant_key,
                    island_tasks,
                )
            )

        # Large islands first, then strong-locality islands first.
        island_items.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        device_load: Dict[str, int] = defaultdict(int)
        device_locality: Dict[str, Counter] = defaultdict(Counter)
        for _size, _dom_ct, dominant_key, island_tasks in island_items:
            def _assign_key(i: int, _dk=dominant_key):
                device = f"{self.accelerator_type}:{i}"
                locality_hit = (
                    device_locality[device].get(_dk, 0)
                    if _dk
                    else 0
                )
                # Prefer locality first, then load balance.
                return (
                    0 if locality_hit > 0 else 1,
                    device_load[device],
                    -locality_hit,
                    i,
                )

            device_idx = min(range(num_devices), key=_assign_key)
            device = f"{self.accelerator_type}:{device_idx}"
            assignments[device] = (
                assignments.get(device, []) + island_tasks
            )
            device_load[device] += len(island_tasks)
            for th in island_tasks:
                key = self._task_locality_key(th)
                if key:
                    device_locality[device][key] += 1
            assignment_metrics.append(
                {
                    "device": device,
                    "task_count": len(island_tasks),
                    "dominant_locality_key": dominant_key or None,
                }
            )
        self._assignment_metrics = assignment_metrics
        return assignments

    def _task_locality_key(self, task_handle: TaskHandle) -> Optional[str]:
        label = task_handle.task().group_label()
        if not label:
            return None
        label = str(label).lower()
        if "::" in label:
            # Prefer explicit file/shard locality emitted by loader tasks.
            label = label.split("::", 1)[1]
        # Collapse shard-style names to a stable prefix for locality grouping.
        if "model-" in label and "-of-" in label:
            return label.split("-of-")[0]
        return label

    def _device_worker(
        self,
        task_list: List[TaskHandle],
        cached_values: Dict[TaskHandle, Any],
        device: str,
        quiet: bool,
    ):
        """
        Execute a set of tasks on a single device.

        This method runs as a thread worker for a specific device.  It
        creates an execution stream on the assigned device, runs the tasks,
        and queues results back to the main thread.  Only results needed for
        target tasks or trailing tasks are retained.

        Args:
            task_list: List of tasks to execute on this device
            cached_values: Values of previously-executed dependent tasks
            device: Device string (e.g. ``"Ascend:0"``)
            quiet: Whether to suppress progress bar output
        """
        LOG.debug("Device %s starting", device)
        stream = _make_stream(device)
        with _stream_context(stream):
            exec = Executor(
                targets=task_list,
                math_device=device,
                storage_device=self.storage_device or device,
                cached_values=cached_values,
            )
            count = 0
            for task_handle, result in exec._run(quiet=quiet):
                count += 1
                if not (
                    task_handle in self.targets
                    or task_handle in self.trailing_dependencies
                ):
                    result = None
                queue_depth = self.task_completion_queue.qsize()
                with self._metric_lock:
                    self._queue_depth_samples.append(int(queue_depth))
                self.task_completion_queue.put((task_handle._index, result))
            with self._metric_lock:
                self._task_metrics.extend(exec.metrics_snapshot()["tasks"])
        _synchronize_device(device)
        LOG.debug("Device %s done", device)

    def metrics_snapshot(self) -> Dict[str, Any]:
        rss_mb = 0.0
        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = float(rss_kb) / 1024.0
        except Exception:
            pass
        with self._metric_lock:
            tasks = list(self._task_metrics)
            qdepth = list(self._queue_depth_samples)
            assignment = list(self._assignment_metrics)
        return {
            "executor": "multi_device",
            "pid": os.getpid(),
            "task_count": len(tasks),
            "tasks": tasks,
            "queue_depth_samples": qdepth,
            "island_assignment": assignment,
            "backpressure_trigger_count": 0,
            "rss_peak_mb": rss_mb,
            "npu_used_peak_mb": None,
        }


# Backward-compatible alias
MultiGPUExecutor = MultiDeviceExecutor
