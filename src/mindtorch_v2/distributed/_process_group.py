import os
import ctypes

from ._hccl.hccl_bindings import (
    get_bindings, HcclRootInfo, HcclCommConfig, HCCL_ROOT_INFO_BYTES,
    HCCL_COMM_CONFIG_COMM_NAME,
    hccl_comm_config_init, is_hccl_feature_supported,
    dtype_to_hccl, _check,
)
from ._work import Work


def _get_stream(device_id):
    from .._backends.npu import state as npu_state
    return npu_state.current_stream(device_id).stream


class ProcessGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size
        self._group_name = ""
        self._group_desc = ""
        self._ranks = None
        self._bound_device_id = None

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def name(self):
        return self._group_name

    @property
    def group_name(self):
        return self._group_name

    @property
    def group_desc(self):
        return self._group_desc

    @property
    def bound_device_id(self):
        return self._bound_device_id


class ProcessGroupHCCL(ProcessGroup):
    """HCCL process group using direct ctypes bindings to libhccl.so.

    Initialization follows torch_npu's pattern:
    1. Try ranktable path (RANK_TABLE_FILE + HcclCommInitClusterInfoConfig)
    2. Fall back to root info path (HcclGetRootInfo + HcclCommInitRootInfoConfig)
    """

    def __init__(self, store, rank, size, device_id=None, group_name="",
                 group_ranks=None):
        super().__init__(rank, size)
        if device_id is None:
            device_id = rank % 8
        self._device_id = device_id
        self._comm = None
        self._group_name = group_name
        self._ranks = group_ranks
        self._store = store
        self._init_hccl(store, group_name)

    def _make_config(self):
        config = HcclCommConfig()
        hccl_comm_config_init(config)
        # If HCCL doesn't support COMM_NAME feature, use compat size=32
        # (torch_npu fallback for older CANN versions)
        if not is_hccl_feature_supported(HCCL_COMM_CONFIG_COMM_NAME):
            import struct
            # Overwrite only the size field (first 8 bytes of reserved)
            struct.pack_into("<Q", (ctypes.c_ubyte * 8).from_buffer(config), 0, 32)
        return config

    def _try_init_cluster_info(self):
        """Try ranktable-based init (preferred for multi-node)."""
        rank_table = os.environ.get("RANK_TABLE_FILE")
        if not rank_table or not os.path.isfile(rank_table):
            return False
        bindings = get_bindings()
        if bindings.comm_init_cluster_info_config is None:
            return False
        config = self._make_config()
        comm = ctypes.c_void_p()
        ret = bindings.comm_init_cluster_info_config(
            rank_table.encode("utf-8"),
            ctypes.c_uint32(self._rank),
            ctypes.byref(config),
            ctypes.byref(comm),
        )
        if ret != 0:
            return False
        self._comm = comm
        return True

    def _init_root_info(self, store, prefix=""):
        """Root info broadcast path (standard NCCL-like init)."""
        bindings = get_bindings()
        key = f"{prefix}/hccl_root_info" if prefix else "hccl_root_info"

        if self._rank == 0:
            root_info = HcclRootInfo()
            ret = bindings.get_root_info(ctypes.byref(root_info))
            _check(ret, "HcclGetRootInfo")
            # NOTE: must read from addressof, not .internal (which truncates at null)
            store.set(key, ctypes.string_at(ctypes.addressof(root_info), HCCL_ROOT_INFO_BYTES))
        else:
            store.wait([key])

        raw = store.get(key)
        if len(raw) != HCCL_ROOT_INFO_BYTES:
            raise RuntimeError(
                f"HCCL root info size mismatch: expected {HCCL_ROOT_INFO_BYTES}, got {len(raw)}"
            )
        root_info = HcclRootInfo()
        # NOTE: must write to addressof, not .internal (which writes to temp copy)
        ctypes.memmove(ctypes.addressof(root_info), raw, HCCL_ROOT_INFO_BYTES)

        config = self._make_config()
        comm = ctypes.c_void_p()
        ret = bindings.comm_init_root_info_config(
            ctypes.c_uint32(self._size),
            ctypes.byref(root_info),
            ctypes.c_uint32(self._rank),
            ctypes.byref(config),
            ctypes.byref(comm),
        )
        _check(ret, "HcclCommInitRootInfoConfig")
        self._comm = comm

    def _init_hccl(self, store, prefix=""):
        from .._backends.npu import state as npu_state
        npu_state.set_device(self._device_id)

        # Try ranktable path first, fall back to root info
        if not self._try_init_cluster_info():
            self._init_root_info(store, prefix)

    def _stream(self):
        return _get_stream(self._device_id)

    def _make_work(self, stream, source_rank=-1):
        return Work(stream=stream, device_id=self._device_id,
                    source_rank=source_rank)

    def _tensor_args(self, tensor):
        ptr = ctypes.c_void_p(tensor.storage().data_ptr())
        count = ctypes.c_uint64(tensor.numel())
        hccl_dtype = ctypes.c_int32(dtype_to_hccl(tensor.dtype))
        return ptr, count, hccl_dtype

    def allreduce(self, tensor, op=0):
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.all_reduce(
            ptr, ptr, count, dt, ctypes.c_int32(int(op)),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclAllReduce")
        return self._make_work(stream)

    def broadcast(self, tensor, root=0):
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.broadcast(
            ptr, count, dt, ctypes.c_uint32(root),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclBroadcast")
        return self._make_work(stream)

    def allgather(self, output_tensor, input_tensor):
        bindings = get_bindings()
        stream = self._stream()
        in_ptr, in_count, dt = self._tensor_args(input_tensor)
        out_ptr = ctypes.c_void_p(output_tensor.storage().data_ptr())
        ret = bindings.all_gather(
            in_ptr, out_ptr, in_count, dt,
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclAllGather")
        return self._make_work(stream)

    def reduce_scatter(self, output_tensor, input_tensor, op=0):
        bindings = get_bindings()
        stream = self._stream()
        in_ptr = ctypes.c_void_p(input_tensor.storage().data_ptr())
        out_ptr, out_count, dt = self._tensor_args(output_tensor)
        ret = bindings.reduce_scatter(
            in_ptr, out_ptr, out_count, dt, ctypes.c_int32(int(op)),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclReduceScatter")
        return self._make_work(stream)

    def reduce(self, tensor, dst=0, op=0):
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.reduce(
            ptr, ptr, count, dt, ctypes.c_int32(int(op)), ctypes.c_uint32(dst),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclReduce")
        return self._make_work(stream)

    def scatter(self, output_tensor, input_tensor, src=0):
        bindings = get_bindings()
        stream = self._stream()
        in_ptr = ctypes.c_void_p(input_tensor.storage().data_ptr())
        out_ptr, out_count, dt = self._tensor_args(output_tensor)
        ret = bindings.scatter(
            in_ptr, out_ptr, out_count, dt, ctypes.c_uint32(src),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclScatter")
        return self._make_work(stream)

    def barrier(self):
        bindings = get_bindings()
        stream = self._stream()
        ret = bindings.barrier(
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclBarrier")
        return self._make_work(stream)

    def send(self, tensor, dst):
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.send(
            ptr, count, dt, ctypes.c_uint32(dst),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclSend")
        return self._make_work(stream)

    def recv(self, tensor, src):
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.recv(
            ptr, count, dt, ctypes.c_uint32(src),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclRecv")
        return self._make_work(stream, source_rank=src)

    def destroy(self):
        if self._comm is not None:
            bindings = get_bindings()
            bindings.comm_destroy(self._comm)
            self._comm = None
