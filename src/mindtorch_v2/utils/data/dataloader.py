from .dataset import IterableDataset
from .sampler import SequentialSampler, RandomSampler, BatchSampler
from ._utils import default_collate, _pin_memory_batch


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
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        if num_workers != 0:
            raise NotImplementedError("num_workers > 0 is planned for phase 2")

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
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory

        self._is_iterable = isinstance(dataset, IterableDataset)

        if self.batch_sampler is None and not self._is_iterable:
            if self.sampler is None:
                if self.shuffle:
                    self.sampler = RandomSampler(dataset, generator=self.generator)
                else:
                    self.sampler = SequentialSampler(dataset)
            self.batch_sampler = BatchSampler(self.sampler, self.batch_size, self.drop_last)

    def __iter__(self):
        if self._is_iterable:
            iterator = iter(self.dataset)
            batch = []
            for item in iterator:
                batch.append(item)
                if len(batch) == self.batch_size:
                    out = self.collate_fn(batch)
                    if self.pin_memory:
                        out = _pin_memory_batch(out)
                    yield out
                    batch = []
            if batch and not self.drop_last:
                out = self.collate_fn(batch)
                if self.pin_memory:
                    out = _pin_memory_batch(out)
                yield out
            return

        for batch_indices in self.batch_sampler:
            items = [self.dataset[i] for i in batch_indices]
            out = self.collate_fn(items)
            if self.pin_memory:
                out = _pin_memory_batch(out)
            yield out

    def __len__(self):
        if self._is_iterable:
            raise TypeError("length of IterableDataset DataLoader is undefined")
        return len(self.batch_sampler)
