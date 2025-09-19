# mypy: allow-untyped-defs
from dataclasses import dataclass
from typing import List

import mindtorch
from mindtorch.distributed._shard.metadata import ShardMetadata
from mindtorch.distributed.remote_device import _remote_device


@dataclass
class Shard:
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.

    Args:
        tensor(mindtorch.Tensor): Local tensor for the shard.
        metadata(:class `mindtorch.distributed._shard.sharded_tensor.ShardMetadata`):
            The metadata for the shard, including offsets, lengths and device placement.
    """

    __slots__ = ["tensor", "metadata"]
    tensor: mindtorch.Tensor
    metadata: ShardMetadata

    def __post_init__(self):
        # verification between local tensor and metadata
        if list(self.tensor.size()) != self.metadata.shard_sizes:
            raise ValueError(
                "Shard tensor size does not match with metadata.shard_lengths! "
                f"Found shard tensor size: {list(self.tensor.size())}, "
                f"metadata.shard_lengths: {self.metadata.shard_sizes}, "
            )
        placement_device = self.metadata.placement
        # if (
        #     placement_device is not None
        #     and placement_device.device() != self.tensor.device
        # ):
        #     raise ValueError(
        #         f"Local shard tensor device does not match with local Shard's placement! "
        #         f"Found local shard tensor device: {self.tensor.device}, "
        #         f"local shard metadata placement device: {placement_device.device()}"
        #     )

    @classmethod
    def from_tensor_and_offsets(
        cls, tensor: mindtorch.Tensor, shard_offsets: List[int], rank: int
    ):
        """
        Creates a Shard of a ShardedTensor from a local mindtorch.Tensor, shard_offsets and rank.

        Args:
            tensor(mindtorch.Tensor): Local tensor for the shard.
            shard_offsets(List[int]): List of integers specify the offset
                of the shard on each dimension.
            rank(int): Specify the rank for the shard.
        """
        shard_sizes = list(tensor.size())
        placement = _remote_device(f"rank:{rank}/{str(tensor.device)}")
        shard_meta = ShardMetadata(
            shard_offsets=shard_offsets, shard_sizes=shard_sizes, placement=placement
        )
        return Shard(tensor, shard_meta)
