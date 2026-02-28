import random

import pytest

import mindtorch_v2 as torch
from mindtorch_v2.utils.data import (
    Dataset,
    IterableDataset,
    TensorDataset,
    ConcatDataset,
    Subset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    DataLoader,
    default_collate,
)


class RangeDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


class StreamDataset(IterableDataset):
    def __iter__(self):
        for i in range(4):
            yield i


def test_tensor_dataset_and_subset_concat():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    td = TensorDataset(a, b)

    assert len(td) == 3
    x0, y0 = td[0]
    assert int(x0.item()) == 1
    assert int(y0.item()) == 4

    sub = Subset(td, [2, 0])
    assert len(sub) == 2
    s0 = sub[0]
    assert int(s0[0].item()) == 3

    cat = ConcatDataset([sub, sub])
    assert len(cat) == 4
    c3 = cat[3]
    assert int(c3[0].item()) == 1


def test_sampler_basics():
    ds = RangeDataset(5)
    seq = list(iter(SequentialSampler(ds)))
    assert seq == [0, 1, 2, 3, 4]

    g = random.Random(0)
    rnd = list(iter(RandomSampler(ds, generator=g)))
    assert sorted(rnd) == [0, 1, 2, 3, 4]

    batch = list(iter(BatchSampler(SequentialSampler(ds), batch_size=2, drop_last=False)))
    assert batch == [[0, 1], [2, 3], [4]]


def test_default_collate_tensor_batch():
    batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    out = default_collate(batch)
    assert tuple(out.shape) == (2, 2)
    assert out.numpy().tolist() == [[1, 2], [3, 4]]


def test_dataloader_map_style_num_workers_zero():
    ds = RangeDataset(5)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batches = list(loader)
    assert batches == [[0, 1], [2, 3], [4]]
    assert len(loader) == 3


def test_dataloader_iterable_dataset_len_and_iter():
    loader = DataLoader(StreamDataset(), batch_size=2, num_workers=0)
    assert list(loader) == [[0, 1], [2, 3]]
    with pytest.raises(TypeError):
        len(loader)


def test_dataloader_constructor_conflicts():
    ds = RangeDataset(4)
    sampler = SequentialSampler(ds)
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=False)

    with pytest.raises(ValueError):
        DataLoader(ds, shuffle=True, sampler=sampler)

    with pytest.raises(ValueError):
        DataLoader(ds, batch_sampler=batch_sampler, batch_size=2)

    with pytest.raises(ValueError):
        DataLoader(ds, batch_sampler=batch_sampler, shuffle=True)

    with pytest.raises(ValueError):
        DataLoader(ds, batch_sampler=batch_sampler, sampler=sampler)

    with pytest.raises(ValueError):
        DataLoader(ds, batch_sampler=batch_sampler, drop_last=True)


def test_dataloader_pin_memory_calls_tensor_pin_memory():
    class SampleDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return torch.tensor([idx], device="cpu")

    loader = DataLoader(SampleDataset(), batch_size=2, pin_memory=True, num_workers=0)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0].is_pinned() is True


def test_dataloader_iterable_dataset_strict_options():
    ds = StreamDataset()

    with pytest.raises(ValueError, match="expected unspecified shuffle option"):
        DataLoader(ds, shuffle=True, num_workers=0)

    with pytest.raises(ValueError, match="expected unspecified sampler option"):
        DataLoader(ds, sampler=[0, 1], num_workers=0)

    with pytest.raises(ValueError, match="expected unspecified batch_sampler option"):
        DataLoader(ds, batch_sampler=[0, 1], num_workers=0)


def test_dataloader_batch_size_none_drop_last_conflict():
    ds = StreamDataset()
    with pytest.raises(ValueError, match="batch_size=None option disables auto-batching"):
        DataLoader(ds, batch_size=None, drop_last=True, num_workers=0)



def test_dataloader_num_workers_map_style_ordered_batches():
    ds = RangeDataset(5)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    batches = list(loader)
    assert batches == [[0, 1], [2, 3], [4]]


def test_dataloader_num_workers_worker_info_and_init_fn():
    import multiprocessing as mp

    class WorkerInfoDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            from mindtorch_v2.utils.data import get_worker_info

            wi = get_worker_info()
            if wi is None:
                return (idx, -1, -1)
            return (idx, wi.id, wi.num_workers)

    manager = mp.Manager()
    called = manager.list()

    def init_fn(worker_id):
        called.append(worker_id)

    loader = DataLoader(
        WorkerInfoDataset(),
        batch_size=2,
        shuffle=False,
        num_workers=2,
        worker_init_fn=init_fn,
    )

    batches = list(loader)
    flat = [x for batch in batches for x in zip(*batch)]
    worker_ids = [item[1] for item in flat]
    worker_counts = [item[2] for item in flat]

    assert set(called) == {0, 1}
    assert set(worker_ids).issubset({0, 1})
    assert set(worker_counts) == {2}


def test_dataloader_multiprocessing_option_validation():
    ds = RangeDataset(4)

    with pytest.raises(ValueError, match='num_workers option should be non-negative'):
        DataLoader(ds, num_workers=-1)

    with pytest.raises(ValueError, match='timeout option should be non-negative'):
        DataLoader(ds, timeout=-1)

    with pytest.raises(
        ValueError,
        match='prefetch_factor option could only be specified in multiprocessing',
    ):
        DataLoader(ds, num_workers=0, prefetch_factor=2)

    with pytest.raises(ValueError, match='persistent_workers option needs num_workers > 0'):
        DataLoader(ds, num_workers=0, persistent_workers=True)



def test_dataloader_iterable_dataset_num_workers_sharded_batches():
    class ShardedStream(IterableDataset):
        def __iter__(self):
            from mindtorch_v2.utils.data import get_worker_info

            wi = get_worker_info()
            if wi is None:
                wid = 0
                nworkers = 1
            else:
                wid = wi.id
                nworkers = wi.num_workers

            for i in range(8):
                if i % nworkers == wid:
                    yield i

    loader = DataLoader(ShardedStream(), batch_size=2, num_workers=2)
    batches = list(loader)
    flat = [x for batch in batches for x in batch]
    assert sorted(flat) == list(range(8))


def test_dataloader_iterable_dataset_num_workers_batch_size_none_sharded():
    class ShardedStream(IterableDataset):
        def __iter__(self):
            from mindtorch_v2.utils.data import get_worker_info

            wi = get_worker_info()
            if wi is None:
                wid = 0
                nworkers = 1
            else:
                wid = wi.id
                nworkers = wi.num_workers

            for i in range(6):
                if i % nworkers == wid:
                    yield i

    loader = DataLoader(ShardedStream(), batch_size=None, num_workers=2)
    values = list(loader)
    assert sorted(values) == list(range(6))



def test_dataloader_persistent_workers_map_reuse_processes_and_init_once():
    import multiprocessing as mp
    import os

    class PidDataset(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            from mindtorch_v2.utils.data import get_worker_info

            wi = get_worker_info()
            wid = -1 if wi is None else wi.id
            return (idx, wid, os.getpid())

    manager = mp.Manager()
    called = manager.list()

    def init_fn(worker_id):
        called.append(worker_id)

    loader = DataLoader(
        PidDataset(),
        batch_size=2,
        shuffle=False,
        num_workers=2,
        worker_init_fn=init_fn,
        persistent_workers=True,
    )

    first = list(loader)
    second = list(loader)

    pids_first = {pid for batch in first for pid in batch[2]}
    pids_second = {pid for batch in second for pid in batch[2]}

    assert set(called) == {0, 1}
    assert len(called) == 2
    assert pids_first == pids_second


def test_dataloader_persistent_workers_iterable_reuse_processes_and_init_once():
    import multiprocessing as mp
    import os

    class PidStream(IterableDataset):
        def __iter__(self):
            from mindtorch_v2.utils.data import get_worker_info

            wi = get_worker_info()
            wid = -1 if wi is None else wi.id
            for i in range(2):
                yield (wid, i, os.getpid())

    manager = mp.Manager()
    called = manager.list()

    def init_fn(worker_id):
        called.append(worker_id)

    loader = DataLoader(
        PidStream(),
        batch_size=2,
        num_workers=2,
        worker_init_fn=init_fn,
        persistent_workers=True,
    )

    first = list(loader)
    second = list(loader)

    pids_first = {pid for batch in first for pid in batch[2]}
    pids_second = {pid for batch in second for pid in batch[2]}

    assert set(called) == {0, 1}
    assert len(called) == 2
    assert pids_first == pids_second


def test_dataloader_multiprocess_timeout_raises_runtimeerror():
    import time

    class SlowDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            time.sleep(0.2)
            return idx

    loader = DataLoader(SlowDataset(), batch_size=1, num_workers=1, timeout=0.05)
    with pytest.raises(RuntimeError, match='DataLoader timed out after'):
        list(loader)


def test_dataloader_multiprocess_worker_error_raises_runtimeerror():
    class CrashDataset(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            if idx == 1:
                raise ValueError('boom')
            return idx

    loader = DataLoader(CrashDataset(), batch_size=1, num_workers=1)
    with pytest.raises(RuntimeError, match='DataLoader worker'):
        list(loader)
