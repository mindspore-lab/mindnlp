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
