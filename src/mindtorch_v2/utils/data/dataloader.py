import queue
import random
from collections import deque
from dataclasses import dataclass

from .dataset import IterableDataset
from .sampler import SequentialSampler, RandomSampler, BatchSampler
from ._utils import (
    default_collate,
    default_convert,
    _pin_memory_batch,
    WorkerInfo,
    _set_worker_info,
    _clear_worker_info,
)


@dataclass
class _WorkerException:
    worker_id: int
    message: str


@dataclass
class _MapPool:
    index_queue: object
    result_queue: object
    workers: list
    pending_events: list


@dataclass
class _IterablePool:
    control_queue: object
    result_queue: object
    workers: list
    pending_events: list



def _ensure_transferable(obj):
    if hasattr(obj, "requires_grad") and hasattr(obj, "grad_fn"):
        if bool(getattr(obj, "requires_grad")) and getattr(obj, "grad_fn") is not None:
            from ...multiprocessing.reductions import non_leaf_requires_grad_error_message

            raise RuntimeError(non_leaf_requires_grad_error_message())
        return
    if isinstance(obj, tuple):
        for x in obj:
            _ensure_transferable(x)
        return
    if isinstance(obj, list):
        for x in obj:
            _ensure_transferable(x)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            _ensure_transferable(x)


def _worker_seed(base_seed, worker_id):
    # Keep seed in uint64 range similar to torch semantics.
    return (int(base_seed) + int(worker_id)) & ((1 << 64) - 1)


def _worker_loop_map(
    dataset,
    index_queue,
    result_queue,
    worker_id,
    num_workers,
    seed,
    worker_init_fn,
    collate_fn,
):
    _set_worker_info(WorkerInfo(worker_id, num_workers, seed, dataset))
    try:
        random.seed(seed)
        if worker_init_fn is not None:
            worker_init_fn(worker_id)
        result_queue.put(("worker_ready", worker_id, None))

        while True:
            task = index_queue.get()
            if task is None:
                break
            send_idx, index = task
            try:
                if isinstance(index, (list, tuple)):
                    data = [dataset[i] for i in index]
                else:
                    data = dataset[index]
                data = collate_fn(data)
                _ensure_transferable(data)
                result_queue.put(("data", send_idx, data))
            except Exception as exc:
                result_queue.put(("error", send_idx, _WorkerException(worker_id, repr(exc))))
                break
    except Exception as exc:
        result_queue.put(("error", -1, _WorkerException(worker_id, repr(exc))))
    finally:
        _clear_worker_info()
        result_queue.put(("worker_exit", worker_id, None))


def _worker_loop_iterable_once(
    dataset,
    result_queue,
    worker_id,
    num_workers,
    seed,
    worker_init_fn,
    auto_collation,
    batch_size,
    drop_last,
    collate_fn,
):
    _set_worker_info(WorkerInfo(worker_id, num_workers, seed, dataset))
    try:
        random.seed(seed)
        if worker_init_fn is not None:
            worker_init_fn(worker_id)
        result_queue.put(("worker_ready", worker_id, None))

        iterator = iter(dataset)
        if not auto_collation:
            for item in iterator:
                item = collate_fn(item)
                _ensure_transferable(item)
                result_queue.put(("data", worker_id, item))
        else:
            batch = []
            for item in iterator:
                batch.append(item)
                if len(batch) == batch_size:
                    out = collate_fn(batch)
                    _ensure_transferable(out)
                    result_queue.put(("data", worker_id, out))
                    batch = []
            if batch and not drop_last:
                out = collate_fn(batch)
                _ensure_transferable(out)
                result_queue.put(("data", worker_id, out))
    except Exception as exc:
        result_queue.put(("error", -1, _WorkerException(worker_id, repr(exc))))
    finally:
        _clear_worker_info()
        result_queue.put(("iter_done", -1, worker_id))
        result_queue.put(("worker_exit", worker_id, None))


def _worker_loop_iterable_persistent(
    dataset,
    control_queue,
    result_queue,
    worker_id,
    num_workers,
    seed,
    worker_init_fn,
    auto_collation,
    batch_size,
    drop_last,
    collate_fn,
):
    _set_worker_info(WorkerInfo(worker_id, num_workers, seed, dataset))
    try:
        random.seed(seed)
        if worker_init_fn is not None:
            worker_init_fn(worker_id)
        result_queue.put(("worker_ready", worker_id, None))

        while True:
            cmd, epoch = control_queue.get()
            if cmd == "shutdown":
                break
            if cmd != "iterate":
                continue

            try:
                iterator = iter(dataset)
                if not auto_collation:
                    for item in iterator:
                        out = collate_fn(item)
                        _ensure_transferable(out)
                        result_queue.put(("data", epoch, out))
                else:
                    batch = []
                    for item in iterator:
                        batch.append(item)
                        if len(batch) == batch_size:
                            out = collate_fn(batch)
                            _ensure_transferable(out)
                            result_queue.put(("data", epoch, out))
                            batch = []
                    if batch and not drop_last:
                        out = collate_fn(batch)
                        _ensure_transferable(out)
                        result_queue.put(("data", epoch, out))
                result_queue.put(("iter_done", epoch, worker_id))
            except Exception as exc:
                result_queue.put(("error", epoch, _WorkerException(worker_id, repr(exc))))
                break
    except Exception as exc:
        result_queue.put(("error", -1, _WorkerException(worker_id, repr(exc))))
    finally:
        _clear_worker_info()
        result_queue.put(("worker_exit", worker_id, None))


def _queue_get(q, timeout):
    if timeout and timeout > 0:
        return q.get(timeout=timeout)
    return q.get()


def _signal_map_shutdown(index_queue, num_workers):
    for _ in range(num_workers):
        try:
            index_queue.put_nowait(None)
        except Exception:
            break


def _signal_iterable_shutdown(control_queue, num_workers):
    for _ in range(num_workers):
        try:
            control_queue.put_nowait(("shutdown", -1))
        except Exception:
            break


def _shutdown_workers(workers):
    for proc in workers:
        proc.join(timeout=0.1)
    for proc in workers:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        generator=None,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        prefetch_factor=None,
        persistent_workers=False,
    ):
        self.dataset = dataset

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; use num_workers=0 to disable multiprocessing."
            )
        if timeout < 0:
            raise ValueError("timeout option should be non-negative")
        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."
            )
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")
        if multiprocessing_context is not None and num_workers == 0:
            raise ValueError(
                "multiprocessing_context can only be used with multi-process loading (num_workers > 0), "
                "but got num_workers=0"
            )

        self.num_workers = num_workers
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self._mp_context = self._resolve_multiprocessing_context(multiprocessing_context)

        if num_workers > 0:
            if prefetch_factor is None:
                prefetch_factor = 2
            elif prefetch_factor < 0:
                raise ValueError("prefetch_factor option should be non-negative")
        self.prefetch_factor = prefetch_factor

        if batch_size is None and drop_last:
            raise ValueError(
                "batch_size=None option disables auto-batching and is mutually exclusive with drop_last"
            )

        if batch_sampler is not None:
            if batch_size != 1:
                raise ValueError("batch_sampler option is mutually exclusive with batch_size")
            if shuffle:
                raise ValueError("batch_sampler option is mutually exclusive with shuffle")
            if sampler is not None:
                raise ValueError("batch_sampler option is mutually exclusive with sampler")
            if drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with drop_last")

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_sampler = batch_sampler
        self.generator = generator
        self.pin_memory = pin_memory
        self._is_iterable = isinstance(dataset, IterableDataset)
        self._auto_collation = batch_size is not None or batch_sampler is not None

        if collate_fn is None:
            self.collate_fn = default_collate if self._auto_collation else default_convert
        else:
            self.collate_fn = collate_fn

        if self._is_iterable:
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified shuffle option, "
                    f"but got shuffle={shuffle}"
                )
            if sampler is not None:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified sampler option, "
                    f"but got sampler={sampler}"
                )
            if batch_sampler is not None:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified batch_sampler option, "
                    f"but got batch_sampler={batch_sampler}"
                )

        if self.batch_sampler is None and not self._is_iterable:
            if self.sampler is None:
                if self.shuffle:
                    self.sampler = RandomSampler(dataset, generator=self.generator)
                else:
                    self.sampler = SequentialSampler(dataset)
            if self._auto_collation:
                self.batch_sampler = BatchSampler(self.sampler, self.batch_size, self.drop_last)

        self._persistent_map_pool = None
        self._persistent_iter_pool = None
        self._persistent_iter_epoch = 0

    def __del__(self):
        try:
            self._shutdown_persistent_workers()
        except Exception:
            pass

    @staticmethod
    def _resolve_multiprocessing_context(multiprocessing_context):
        if multiprocessing_context is None:
            from ... import multiprocessing as mt_mp

            return mt_mp.get_context()

        if isinstance(multiprocessing_context, str):
            try:
                from ... import multiprocessing as mt_mp

                return mt_mp.get_context(multiprocessing_context)
            except ValueError as exc:
                from ... import multiprocessing as mt_mp

                methods = mt_mp.get_all_start_methods()
                raise ValueError(
                    "multiprocessing_context option should specify a valid start method "
                    f"in {methods}, but got multiprocessing_context={multiprocessing_context!r}"
                ) from exc

        try:
            from multiprocessing.context import BaseContext
        except Exception:
            BaseContext = object

        if not isinstance(multiprocessing_context, BaseContext):
            raise TypeError(
                "multiprocessing_context option should be a valid context object or "
                "a string specifying the start method, "
                f"but got multiprocessing_context={multiprocessing_context!r}"
            )

        return multiprocessing_context

    def _queue_depth(self):
        return max(1, int(self.prefetch_factor or 1) * self.num_workers)

    def _maybe_pin(self, out):
        if self.pin_memory:
            return _pin_memory_batch(out)
        return out

    def _create_map_pool(self):
        queue_depth = self._queue_depth()
        index_queue = self._mp_context.Queue(maxsize=queue_depth)
        result_queue = self._mp_context.Queue(maxsize=queue_depth)
        workers = []
        base_seed = random.getrandbits(64)

        for worker_id in range(self.num_workers):
            seed = _worker_seed(base_seed, worker_id)
            proc = self._mp_context.Process(
                target=_worker_loop_map,
                args=(
                    self.dataset,
                    index_queue,
                    result_queue,
                    worker_id,
                    self.num_workers,
                    seed,
                    self.worker_init_fn,
                    self.collate_fn,
                ),
            )
            proc.daemon = True
            proc.start()
            workers.append(proc)

        pending_events = self._await_worker_ready(result_queue, workers)
        return _MapPool(
            index_queue=index_queue,
            result_queue=result_queue,
            workers=workers,
            pending_events=pending_events,
        )

    def _create_iterable_pool(self, persistent):
        queue_depth = self._queue_depth()
        control_queue = self._mp_context.Queue(maxsize=queue_depth) if persistent else None
        result_queue = self._mp_context.Queue(maxsize=queue_depth)
        workers = []
        base_seed = random.getrandbits(64)

        for worker_id in range(self.num_workers):
            seed = _worker_seed(base_seed, worker_id)
            if persistent:
                target = _worker_loop_iterable_persistent
                args = (
                    self.dataset,
                    control_queue,
                    result_queue,
                    worker_id,
                    self.num_workers,
                    seed,
                    self.worker_init_fn,
                    self._auto_collation,
                    self.batch_size,
                    self.drop_last,
                    self.collate_fn,
                )
            else:
                target = _worker_loop_iterable_once
                args = (
                    self.dataset,
                    result_queue,
                    worker_id,
                    self.num_workers,
                    seed,
                    self.worker_init_fn,
                    self._auto_collation,
                    self.batch_size,
                    self.drop_last,
                    self.collate_fn,
                )

            proc = self._mp_context.Process(target=target, args=args)
            proc.daemon = True
            proc.start()
            workers.append(proc)

        pending_events = []
        if persistent:
            pending_events = self._await_worker_ready(result_queue, workers)
        return _IterablePool(
            control_queue=control_queue,
            result_queue=result_queue,
            workers=workers,
            pending_events=pending_events,
        )

    def _await_worker_ready(self, result_queue, workers):
        ready_count = 0
        pending = []
        while ready_count < len(workers):
            try:
                kind, key, payload = _queue_get(result_queue, self.timeout)
            except queue.Empty as exc:
                _shutdown_workers(workers)
                raise RuntimeError(f"DataLoader timed out after {self.timeout} seconds") from exc

            if kind == "worker_ready":
                ready_count += 1
                continue
            if kind == "error":
                _shutdown_workers(workers)
                raise RuntimeError(f"DataLoader worker {payload.worker_id} failed: {payload.message}")
            if kind == "worker_exit":
                _shutdown_workers(workers)
                raise RuntimeError(f"DataLoader worker {key} exited unexpectedly")

            pending.append((kind, key, payload))

        return pending

    def _all_workers_alive(self, workers):
        return all(proc.is_alive() for proc in workers)

    def _ensure_persistent_map_pool(self):
        pool = self._persistent_map_pool
        if pool is None or not self._all_workers_alive(pool.workers):
            if pool is not None:
                _signal_map_shutdown(pool.index_queue, len(pool.workers))
                _shutdown_workers(pool.workers)
            pool = self._create_map_pool()
            self._persistent_map_pool = pool
        return pool

    def _ensure_persistent_iterable_pool(self):
        pool = self._persistent_iter_pool
        if pool is None or not self._all_workers_alive(pool.workers):
            if pool is not None:
                _signal_iterable_shutdown(pool.control_queue, len(pool.workers))
                _shutdown_workers(pool.workers)
            pool = self._create_iterable_pool(persistent=True)
            self._persistent_iter_pool = pool
        return pool

    def _shutdown_persistent_workers(self):
        if self._persistent_map_pool is not None:
            pool = self._persistent_map_pool
            _signal_map_shutdown(pool.index_queue, len(pool.workers))
            _shutdown_workers(pool.workers)
            self._persistent_map_pool = None
        if self._persistent_iter_pool is not None:
            pool = self._persistent_iter_pool
            _signal_iterable_shutdown(pool.control_queue, len(pool.workers))
            _shutdown_workers(pool.workers)
            self._persistent_iter_pool = None

    def _iter_single_process_iterable(self):
        iterator = iter(self.dataset)
        if not self._auto_collation:
            for item in iterator:
                yield self._maybe_pin(self.collate_fn(item))
            return

        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield self._maybe_pin(self.collate_fn(batch))
                batch = []
        if batch and not self.drop_last:
            yield self._maybe_pin(self.collate_fn(batch))

    def _iter_single_process_map(self):
        if self._auto_collation:
            for batch_indices in self.batch_sampler:
                items = [self.dataset[i] for i in batch_indices]
                yield self._maybe_pin(self.collate_fn(items))
            return

        for index in self.sampler:
            yield self._maybe_pin(self.collate_fn(self.dataset[index]))

    def _iter_multiprocess_map(self):
        persistent = self.persistent_workers
        pool = self._ensure_persistent_map_pool() if persistent else self._create_map_pool()
        index_queue = pool.index_queue
        result_queue = pool.result_queue
        workers = pool.workers
        pending_events = deque(pool.pending_events)
        pool.pending_events = []

        try:
            send_count = 0
            source_iter = self.batch_sampler if self._auto_collation else self.sampler
            for send_idx, index in enumerate(source_iter):
                index_queue.put((send_idx, index))
                send_count += 1

            next_idx = 0
            reorder = {}
            while next_idx < send_count:
                try:
                    if pending_events:
                        kind, key, payload = pending_events.popleft()
                    else:
                        kind, key, payload = _queue_get(result_queue, self.timeout)
                except queue.Empty as exc:
                    if persistent:
                        self._shutdown_persistent_workers()
                    raise RuntimeError(f"DataLoader timed out after {self.timeout} seconds") from exc

                if kind == "error":
                    if persistent:
                        self._shutdown_persistent_workers()
                    raise RuntimeError(
                        f"DataLoader worker {payload.worker_id} failed: {payload.message}"
                    )
                if kind != "data":
                    continue

                reorder[key] = payload
                while next_idx in reorder:
                    data = reorder.pop(next_idx)
                    yield self._maybe_pin(data)
                    next_idx += 1
        finally:
            if not persistent:
                _shutdown_workers(workers)

    def _iter_multiprocess_iterable(self):
        persistent = self.persistent_workers
        pool = self._ensure_persistent_iterable_pool() if persistent else self._create_iterable_pool(False)
        result_queue = pool.result_queue
        workers = pool.workers
        pending_events = deque(pool.pending_events)
        pool.pending_events = []

        epoch = self._persistent_iter_epoch
        if persistent:
            self._persistent_iter_epoch += 1
            for _ in workers:
                pool.control_queue.put(("iterate", epoch))

        try:
            worker_done = 0
            batch = []
            while worker_done < len(workers):
                try:
                    if pending_events:
                        kind, key, payload = pending_events.popleft()
                    else:
                        kind, key, payload = _queue_get(result_queue, self.timeout)
                except queue.Empty as exc:
                    if persistent:
                        self._shutdown_persistent_workers()
                    raise RuntimeError(f"DataLoader timed out after {self.timeout} seconds") from exc

                if kind == "iter_done":
                    if not persistent or key == epoch:
                        worker_done += 1
                    continue

                if kind == "error":
                    if not persistent or key == epoch:
                        if persistent:
                            self._shutdown_persistent_workers()
                        raise RuntimeError(
                            f"DataLoader worker {payload.worker_id} failed: {payload.message}"
                        )
                    continue

                if kind != "data":
                    continue
                if persistent and key != epoch:
                    continue

                out = payload
                yield self._maybe_pin(out)
        finally:
            if not persistent:
                _shutdown_workers(workers)

    def __iter__(self):
        if self.num_workers == 0:
            if self._is_iterable:
                return self._iter_single_process_iterable()
            return self._iter_single_process_map()

        if self._is_iterable:
            return self._iter_multiprocess_iterable()
        return self._iter_multiprocess_map()

    def __len__(self):
        if self._is_iterable:
            raise TypeError("length of IterableDataset DataLoader is undefined")
        if self._auto_collation:
            return len(self.batch_sampler)
        return len(self.sampler)
