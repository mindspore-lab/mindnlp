from ..._functional import stack as _stack


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
