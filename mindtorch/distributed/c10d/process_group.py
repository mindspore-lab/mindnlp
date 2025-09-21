import mindtorch
from mindtorch import Tensor
from mindtorch.executor import execute
from typing import List, Optional, Dict, Any
from enum import Enum


class BackendType(Enum):
    UNDEFINED = 0
    GLOO = 1
    NCCL = 2
    UCC = 3
    MPI = 4
    CUSTOM = 5


def backend_type_to_string(backend_type: BackendType) -> str:
    if backend_type == BackendType.GLOO:
        return "gloo"
    elif backend_type == BackendType.NCCL:
        return "nccl"
    elif backend_type == BackendType.UCC:
        return "ucc"
    elif backend_type == BackendType.MPI:
        return "mpi"
    elif backend_type == BackendType.UNDEFINED:
        return "undefined"
    elif backend_type == BackendType.CUSTOM:
        return "custom"
    else:
        raise ValueError("Unknown backend type!")


def str_to_backend_type(backend: str) -> BackendType:
    if backend == "undefined":
        return BackendType.UNDEFINED
    elif backend == "gloo":
        return BackendType.GLOO
    elif backend == "nccl":
        return BackendType.NCCL
    elif backend == "ucc":
        return BackendType.UCC
    elif backend == "mpi":
        return BackendType.MPI
    else:
        return BackendType.CUSTOM


class ProcessGroup:
    class BackendType(Enum):
        UNDEFINED = 0
        GLOO = 1
        NCCL = 2
        UCC = 3
        MPI = 4
        CUSTOM = 5

    def __init__(self, store: Optional[Any] = None, rank: int = 0, size: int = 0):
        self.store = store
        self._name = self.store.prefix_[:-1]
        self._rank = rank
        self._size = size
        self.backend_type = BackendType.UNDEFINED
        self.device_type_to_backend = {}
        self.backend_type_to_backend = {}
        self.device_types = set()
        self.pg_desc = ""
        self.dist_debug_level = "Off"

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def get_rank(self) -> int:
        return self._rank

    def get_size(self) -> int:
        return self._size

    def get_backend_name(self) -> str:
        return backend_type_to_string(self.backend_type)

    def set_backend(self, device_type, backend_type: BackendType, backend: Optional[Any] = None):
        self.device = device_type
        self.device_type_to_backend[device_type] = backend_type
        self.device_types.add(device_type)
        
        if backend_type in self.backend_type_to_backend:
            existing_backend = self.backend_type_to_backend[backend_type]
            self.device_type_to_backend[device_type] = existing_backend
        else:
            if backend:
                self.device_type_to_backend[device_type] = backend
                self.backend_type_to_backend[backend_type] = backend

    def get_backend(self, device_type) -> Any:
        if device_type in self.device_type_to_backend:
            return self.device_type_to_backend[device_type]
        else:
            raise ValueError(f"No backend found for device type {device_type}")

    def start_coalescing(self, device_type):
        backend = self.get_backend(device_type)
        backend.start_coalescing()

    def end_coalescing(self, device_type):
        backend = self.get_backend(device_type)
        return backend.end_coalescing()

    def broadcast(self, tensors: List[Tensor], opts: Any) -> Any:
        tensor = tensors[0]
        _, work = execute('dist_comm_broadcast', tensor, opts.rootRank, self._rank, self._name, device=self.device)
        return work

    def allreduce(self, tensors: List[Tensor], opts: Any) -> Any:
        tensor = tensors[0]
        _, handle = execute('dist_comm_all_reduce', tensor, opts.reduceOp, self._name, device=self.device)
        return handle

    def _allgather_base(self, output_tensor: Tensor, input_tensor: Tensor, opts: Any=None):
        input_size = (-1,)
        output_rank = output_tensor.ndim - 1
        if output_rank > 0:
            input_size = input_size + input_tensor.shape[input_tensor.ndim - output_rank:]
        _, handle = execute('dist_comm_all_gather_into_tensor', output_tensor, input_tensor.view(input_size), self._size, self._name, device=self.device)
        return handle

    def allgather(self, output_tensors: List[List[Tensor]], input_tensors: List[Tensor], opts: Any=None) -> Any:
        tensor_list = output_tensors[0]
        tensor = input_tensors[0]
        _, handle = execute('dist_comm_all_gather', tensor_list, tensor, self._size, self._name, device=self.device)
        return handle

    def reduce(self, tensors: List[Tensor], opts: Any) -> Any:
        out = reduce(tensors[0], opts.rootRank, opts.reduceOp, self._name)
        return out

    def gather(self, output_tensors, input_tensors, opts):
        # # do not use mindspore.communication.gather because not support uint8
        tensor = input_tensors[0]
        gather_list = output_tensors[0]

        _, work = execute('dist_comm_gather', tensor, gather_list, self._size, opts.rootRank, self._rank, self._name, device=self.device)
        return work

    def scatter(self, output_tensors: List[Tensor], input_tensors: List[List[Tensor]], opts: Any) -> Any:
        tensor = output_tensors[0]
        scatter_list = input_tensors[0]
        _, work = execute('dist_comm_scatter', tensor, scatter_list, self._size, opts.rootRank, self._rank, self._name, device=self.device)
        return work

    def reduce_scatter(self, output_tensors: List[Tensor], input_tensors: List[List[Tensor]], opts: Any) -> Any:
        output = output_tensors[0]
        input_list = input_tensors[0]
        _, work = execute('dist_comm_reduce_scatter', output, input_list, self._size, opts.reduceOp, self._name, device=self.device)
        if allow_inflight_collective_as_graph_input():
            for tensor in output_tensors:
                register_work(tensor, work)
        return work

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts: Any):
        _, work = execute('dist_comm_reduce_scatter_tensor', output_tensor, input_tensor, self._size, opts.reduceOp, self._name, device=self.device)
        return work

    def barrier(self, opts: Any) -> Any:
        _, work = execute('dist_comm_barrier', self._name, device=self.device)
        return work

    def recv(self, tensors, srcRank, tag):
        tensor = tensors[0]
        _, work = execute('dist_comm_irecv', tensor, tag, srcRank, self._name, device=self.device)
        return work

    def send(self, tensors: List[Tensor], dstRank: int, tag: int):
        tensor = tensors[0]
        _, handle = execute('dist_comm_isend', tensor, dstRank, self._name, tag, device=self.device)
        return handle

    def get_device_types(self) -> List[Any]:
        return list(self.device_types)

    def set_group_name(self, name: str):
        for backend in self.device_type_to_backend.values():
            backend.set_group_uid(name)
    
    def get_group_name(self) -> str:
        return self.device_type_to_backend[next(iter(self.device_type_to_backend))].get_group_uid()

    @property
    def group_name(self) -> str:
        return self._name

    def set_group_desc(self, desc: str):
        self.pg_desc = desc
        for backend in self.device_type_to_backend.values():
            backend.set_group_desc(desc)

    def enable_collectives_timing(self):
        for backend in self.device_type_to_backend.values():
            backend.enable_collectives_timing()

    def release_resources(self):
        self.device_type_to_backend.clear()
        self.backend_type_to_backend.clear()
        self.store = None

    def _register_backend(self, device, backend_type, backend_class):
        self.device = device

class WorkRegistry:
    def __init__(self):
        self.registry = {}
        self.allow_inflight_collective_as_graph_input = False

    def register_work(self, tensor: Tensor, work: Any):
        if not tensor.has_storage():
            print(f"Warning: Tensor {tensor} has no storage!")
            return
        storage = tensor.storage().getWeakStorageImpl()
        if storage not in self.registry:
            self.registry[storage] = [work]
        else:
            if work not in self.registry[storage]:
                self.registry[storage].append(work)

    def pop_works(self, tensor: Tensor):
        storage = tensor.storage().getWeakStorageImpl()
        if storage in self.registry:
            works = self.registry.pop(storage)
            return works
        return []

    def unregister_work(self, work: Any):
        for storage, works in list(self.registry.items()):
            self.registry[storage] = [w for w in works if w != work]
            if not self.registry[storage]:
                del self.registry[storage]

    def get_work_registry_size(self):
        return sum(len(works) for works in self.registry.values())

    def set_allow_inflight_collective_as_graph_input(self, value: bool):
        self.allow_inflight_collective_as_graph_input = value

    # @property
    # def allow_inflight_collective_as_graph_input(self) -> bool:
    #     return self.allow_inflight_collective_as_graph_input

    def __del__(self):
        if self.get_work_registry_size() > 0:
            print("Warning: Some work objects were not awaited!")


# Global WorkRegistry
process_registry = WorkRegistry()

# Helper functions
def register_work(tensor: Tensor, work: Any):
    process_registry.register_work(tensor, work)


def wait_tensor(tensor: Tensor) -> Tensor:
    works = process_registry.pop_works(tensor)
    for work in works:
        work.wait()
    return tensor


def unregister_work(work: Any):
    process_registry.unregister_work(work)


def get_work_registry_size() -> int:
    return process_registry.get_work_registry_size()


def set_allow_inflight_collective_as_graph_input(value: bool):
    process_registry.set_allow_inflight_collective_as_graph_input(value)


def allow_inflight_collective_as_graph_input() -> bool:
    return process_registry.allow_inflight_collective_as_graph_input()


def create_tensor(device: Optional[Any] = None) -> Tensor:
    # Placeholder function for tensor creation
    if device:
        return mindtorch.empty([1], device=device)
    return mindtorch.empty([1])


def get_backend_op(name: str):
    # Placeholder for fetching backend operation
    # Would need to map to actual dispatcher
    pass
