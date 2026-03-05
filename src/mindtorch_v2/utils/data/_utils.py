import threading
from dataclasses import dataclass

from ..._functional import stack as _stack


@dataclass
class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: object


_worker_local = threading.local()


def _set_worker_info(info):
    _worker_local.info = info


def _clear_worker_info():
    if hasattr(_worker_local, "info"):
        delattr(_worker_local, "info")


def get_worker_info():
    return getattr(_worker_local, "info", None)


def default_convert(data):
    return data


def default_collate(batch):
    if len(batch) == 0:
        return batch
    first = batch[0]
    if hasattr(first, "device") and hasattr(first, "shape"):
        return _stack(batch, dim=0)
    if isinstance(first, (int, float, bool, str)):
        return list(batch)
    if isinstance(first, tuple):
        transposed = list(zip(*batch))
        return tuple(default_collate(list(samples)) for samples in transposed)
    if isinstance(first, list):
        transposed = list(zip(*batch))
        return [default_collate(list(samples)) for samples in transposed]
    if isinstance(first, dict):
        return {k: default_collate([d[k] for d in batch]) for k in first.keys()}
    return list(batch)


def _pin_memory_batch(batch):
    if hasattr(batch, "pin_memory"):
        return batch.pin_memory()
    if isinstance(batch, tuple):
        return tuple(_pin_memory_batch(x) for x in batch)
    if isinstance(batch, list):
        return [_pin_memory_batch(x) for x in batch]
    if isinstance(batch, dict):
        return {k: _pin_memory_batch(v) for k, v in batch.items()}
    return batch
