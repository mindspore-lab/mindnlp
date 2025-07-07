# mypy: allow-untyped-defs
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from mindnlp import core
from core.distributed._shard.metadata import ShardMetadata


class MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: core.dtype = field(default=core.get_default_dtype())
    # layout: core.layout = field(default=core.strided)
    requires_grad: bool = False
    # memory_format: core.memory_format = field(default=core.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self):
        # Since core.memory_format cannot be pickled!
        # memory_format = self.memory_format
        # if memory_format == core.contiguous_format:
        #     mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        # elif memory_format == core.channels_last:
        #     mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        # elif memory_format == core.preserve_format:
        #     mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        # else:
        #     raise RuntimeError(f"Invalid core.memory_format: {memory_format}")

        return (
            self.dtype,
            # self.layout,
            self.requires_grad,
            # mem_format_encoding,
            # self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (
            self.dtype,
            # self.layout,
            self.requires_grad,
            # mem_format_encoding,
            # self.pin_memory,
        ) = state

        # if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
        #     memory_format = core.contiguous_format
        # elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
        #     memory_format = core.channels_last
        # elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
        #     memory_format = core.preserve_format
        # else:
        #     raise RuntimeError(
        #         f"Invalid core.memory_format encoding: {mem_format_encoding}"
        #     )

        # self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: core.Tensor) -> "TensorProperties":
        return TensorProperties(
            dtype=tensor.dtype,
            # layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            # memory_format=core.contiguous_format,
            # pin_memory=tensor.is_pinned(),
        )


@dataclass
class ShardedTensorMetadata:
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: List[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: core.Size = field(default=core.Size([]))

    tensor_properties: TensorProperties = field(default_factory=TensorProperties)
