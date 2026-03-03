"""DistributedSampler for distributed data loading."""

import math


class DistributedSampler:
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with DistributedDataParallel.
    In such case, each process can pass a DistributedSampler instance as a
    DataLoader sampler, and load a subset of the original dataset that is
    exclusive to it.

    Args:
        dataset: Dataset used for sampling
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process within num_replicas
        shuffle: If True, sampler will shuffle the indices
        seed: Random seed used to shuffle the sampler if shuffle=True
        drop_last: If True, drop the tail of the data to make it evenly divisible
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        if num_replicas is None:
            from ... import distributed as dist
            if not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            from ... import distributed as dist
            if not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # If drop_last, we drop the tail of the data to make it evenly divisible
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        """Generate indices for this rank."""
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            import random
            g = random.Random()
            g.seed(self.seed + self.epoch)
            indices = list(range(len(self.dataset)))
            g.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        """Return the number of samples for this rank."""
        return self.num_samples

    def set_epoch(self, epoch):
        """Set the epoch for this sampler.

        When shuffle=True, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration
        of this sampler will yield the same ordering.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
