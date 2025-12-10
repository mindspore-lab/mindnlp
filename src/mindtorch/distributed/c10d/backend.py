import mindtorch
from typing import List, Optional, Callable, Any
from enum import Enum
import time

# Enum for Backend operations
class OpType(Enum):
    BROADCAST = 0
    ALLREDUCE = 1
    ALLREDUCE_COALESCED = 2
    REDUCE = 3
    ALLGATHER = 4
    _ALLGATHER_BASE = 5
    ALLGATHER_COALESCED = 6
    GATHER = 7
    SCATTER = 8
    REDUCE_SCATTER = 9
    ALLTOALL_BASE = 10
    ALLTOALL = 11
    SEND = 12
    RECV = 13
    RECVANYSOURCE = 14
    BARRIER = 15
    _REDUCE_SCATTER_BASE = 16
    COALESCED = 17
    _ALLREDUCE_SPARSE = 18
    UNKNOWN = 100

kBackendDefaultTimeout = 30 * 60 * 1000  # Default timeout in milliseconds

class Backend:

    class Options:
        def __init__(self, backend: str, timeout: int = kBackendDefaultTimeout):
            self.timeout = timeout
            self.backend = backend

    def __init__(self, rank: int, size: int):
        self.rank_ = rank
        self.size_ = size
        self.pg_uid_ = ""
        self.pg_desc_ = ""
        self.dist_debug_level_ = "Off"
        self.bound_device_id_ = None

    def get_rank(self) -> int:
        return self.rank_

    def get_size(self) -> int:
        return self.size_

    def get_id(self) -> int:
        return id(self)

    def supports_splitting(self) -> bool:
        return False

    def start_coalescing(self):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not implement startCoalescing.")

    def end_coalescing(self):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not implement endCoalescing.")

    def get_backend_name(self) -> str:
        raise NotImplementedError("getBackendName is not implemented.")

    def broadcast(self, tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support broadcast.")

    def allreduce(self, tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support allreduce.")

    def allreduce_sparse(self, tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support allreduce sparse.")

    def allreduce_coalesced(self, tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support allreduce_coalesced.")

    def reduce(self, tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support reduce.")

    def allgather(self, output_tensors: List[List[mindtorch.Tensor]], input_tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support allgather.")

    def _allgather_base(self, output_buffer: mindtorch.Tensor, input_buffer: mindtorch.Tensor, opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support _allgather_base.")

    def allgather_coalesced(self, output_tensor_lists: List[List[mindtorch.Tensor]], input_tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support allgather_coalesced.")

    def allgather_into_tensor_coalesced(self, outputs: List[mindtorch.Tensor], inputs: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support allgather_into_tensor_coalesced.")

    def gather(self, output_tensors: List[List[mindtorch.Tensor]], input_tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support gather.")

    def scatter(self, output_tensors: List[mindtorch.Tensor], input_tensors: List[List[mindtorch.Tensor]], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support scatter.")

    def reduce_scatter(self, output_tensors: List[mindtorch.Tensor], input_tensors: List[List[mindtorch.Tensor]], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support reduce_scatter.")

    def _reduce_scatter_base(self, output_buffer: mindtorch.Tensor, input_buffer: mindtorch.Tensor, opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support _reduce_scatter_base.")

    def reduce_scatter_tensor_coalesced(self, outputs: List[mindtorch.Tensor], inputs: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support reduce_scatter_tensor_coalesced.")

    def alltoall_base(self, output_buffer: mindtorch.Tensor, input_buffer: mindtorch.Tensor, output_split_sizes: List[int], input_split_sizes: List[int], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support alltoall_base.")

    def alltoall(self, output_tensors: List[mindtorch.Tensor], input_tensors: List[mindtorch.Tensor], opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support alltoall.")

    def monitored_barrier(self, opts: Optional[Any] = None, wait_all_ranks=False):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support monitoredBarrier, only GLOO supports monitored barrier.")

    def set_sequence_number_for_group(self):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not yet support sequence numbers.")

    def get_sequence_number_for_group(self) -> int:
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not yet support sequence numbers.")

    def send(self, tensors: List[mindtorch.Tensor], dst_rank: int, tag: int):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support send.")

    def recv(self, tensors: List[mindtorch.Tensor], src_rank: int, tag: int):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support recv.")

    def recv_anysource(self, tensors: List[mindtorch.Tensor], tag: int):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support recvAnysource.")

    def barrier(self, opts: Optional[Any] = None):
        raise NotImplementedError(f"Backend {self.get_backend_name()} does not support barrier.")

    def register_on_completion_hook(self, hook: Callable):
        raise NotImplementedError(f"Only ProcessGroupNCCL supports onCompletion hook, but got {self.get_backend_name()} backend.")

    def wait_for_pending_works(self):
        raise NotImplementedError(f"Only ProcessGroupNCCL supports waitForPendingWorks, but got {self.get_backend_name()} backend.")

    def enable_collectives_timing(self):
        raise NotImplementedError(f"Backend {self.get_backend_name()} is missing implementation of enableCollectivesTiming.")

    def has_hooks(self) -> bool:
        return self.on_completion_hook_ is not None

    def set_group_uid(self, pg_uid: str):
        self.pg_uid_ = pg_uid

    def get_group_uid(self) -> str:
        return self.pg_uid_

    def set_group_desc(self, desc: str):
        self.pg_desc_ = desc

    def get_group_desc(self) -> str:
        return self.pg_desc_

    def get_bound_device_id(self) -> Optional[mindtorch.device]:
        return self.bound_device_id_

    def eager_connect_single_device(self, device: mindtorch.device):
        pass

    def set_bound_device_id(self, device: Optional[mindtorch.device]):
        if device:
            assert device.index is not None, "setBoundDeviceId must have an index"
        self.bound_device_id_ = device

# Example subclass implementation (e.g., for NCCL, GLOO)
class NCCLBackend(Backend):
    def __init__(self, rank: int, size: int):
        super().__init__(rank, size)

    def get_backend_name(self) -> str:
        return "NCCL"

    def start_coalescing(self):
        pass

    def end_coalescing(self):
        pass
