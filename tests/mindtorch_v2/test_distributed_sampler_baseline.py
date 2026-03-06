import mindtorch_v2 as torch
from mindtorch_v2.utils.data import Dataset, DistributedSampler


class RangeDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


def test_distributed_sampler_even_partition_no_drop():
    ds = RangeDataset(8)
    s0 = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=False, drop_last=False)
    s1 = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=False, drop_last=False)

    i0 = list(iter(s0))
    i1 = list(iter(s1))

    assert i0 == [0, 2, 4, 6]
    assert i1 == [1, 3, 5, 7]


def test_distributed_sampler_padding_when_not_divisible():
    ds = RangeDataset(5)
    s0 = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=False, drop_last=False)
    s1 = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=False, drop_last=False)

    i0 = list(iter(s0))
    i1 = list(iter(s1))

    assert len(i0) == 3
    assert len(i1) == 3
    assert sorted(i0 + i1) == [0, 0, 1, 2, 3, 4]


def test_distributed_sampler_drop_last():
    ds = RangeDataset(5)
    s0 = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=False, drop_last=True)
    s1 = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=False, drop_last=True)

    i0 = list(iter(s0))
    i1 = list(iter(s1))

    assert len(i0) == 2
    assert len(i1) == 2
    assert sorted(i0 + i1) == [0, 1, 2, 3]


def test_distributed_sampler_set_epoch_changes_shuffle_order():
    ds = RangeDataset(10)
    s = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=7, drop_last=False)

    s.set_epoch(0)
    e0 = list(iter(s))
    s.set_epoch(1)
    e1 = list(iter(s))

    assert e0 != e1
    assert len(e0) == len(e1) == len(s)
