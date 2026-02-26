import os
import ctypes

from ._work import Work


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

    HCCL bindings are imported lazily when this class is instantiated, so
    importing the base ProcessGroup class does not require libhccl.so.
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

    @staticmethod
    def _load_bindings():
        from ._hccl.hccl_bindings import (
            get_bindings, HcclRootInfo, HcclCommConfig, HCCL_ROOT_INFO_BYTES,
            HCCL_COMM_CONFIG_COMM_NAME,
            hccl_comm_config_init, is_hccl_feature_supported,
            dtype_to_hccl, _check,
        )
        return (get_bindings, HcclRootInfo, HcclCommConfig,
                HCCL_ROOT_INFO_BYTES, HCCL_COMM_CONFIG_COMM_NAME,
                hccl_comm_config_init, is_hccl_feature_supported,
                dtype_to_hccl, _check)

    def _make_config(self):
        (_, _, HcclCommConfig, _, HCCL_COMM_CONFIG_COMM_NAME,
         hccl_comm_config_init, is_hccl_feature_supported, _, _) = self._load_bindings()
        config = HcclCommConfig()
        hccl_comm_config_init(config)
        if not is_hccl_feature_supported(HCCL_COMM_CONFIG_COMM_NAME):
            import struct
            struct.pack_into("<Q", (ctypes.c_ubyte * 8).from_buffer(config), 0, 32)
        return config

    def _try_init_cluster_info(self):
        """Try ranktable-based init (preferred for multi-node)."""
        rank_table = os.environ.get("RANK_TABLE_FILE")
        if not rank_table or not os.path.isfile(rank_table):
            return False
        get_bindings = self._load_bindings()[0]
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
        (get_bindings, HcclRootInfo, _, HCCL_ROOT_INFO_BYTES,
         _, _, _, _, _check) = self._load_bindings()
        bindings = get_bindings()
        key = f"{prefix}/hccl_root_info" if prefix else "hccl_root_info"

        if self._rank == 0:
            root_info = HcclRootInfo()
            ret = bindings.get_root_info(ctypes.byref(root_info))
            _check(ret, "HcclGetRootInfo")
            store.set(key, ctypes.string_at(ctypes.addressof(root_info), HCCL_ROOT_INFO_BYTES))
        else:
            store.wait([key])

        raw = store.get(key)
        if len(raw) != HCCL_ROOT_INFO_BYTES:
            raise RuntimeError(
                f"HCCL root info size mismatch: expected {HCCL_ROOT_INFO_BYTES}, got {len(raw)}"
            )
        root_info = HcclRootInfo()
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
        from .._backends.npu import state as npu_state
        return npu_state.current_stream(self._device_id).stream

    def _make_work(self, stream, source_rank=-1):
        return Work(stream=stream, device_id=self._device_id,
                    source_rank=source_rank)

    def _tensor_args(self, tensor):
        _, _, _, _, _, _, _, dtype_to_hccl, _ = self._load_bindings()
        ptr = ctypes.c_void_p(tensor.storage().data_ptr())
        count = ctypes.c_uint64(tensor.numel())
        hccl_dtype = ctypes.c_int32(dtype_to_hccl(tensor.dtype))
        return ptr, count, hccl_dtype

    def allreduce(self, tensor, op=0):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
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
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
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
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
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
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
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
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
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
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
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
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ret = bindings.barrier(
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclBarrier")
        return self._make_work(stream)

    def send(self, tensor, dst):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
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
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.recv(
            ptr, count, dt, ctypes.c_uint32(src),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclRecv")
        return self._make_work(stream, source_rank=src)

    def _p2p_exchange(self, send_ptr, recv_ptr, count, hccl_dtype, peer,
                      stream, bindings, _check):
        """Send/recv with a single peer using deadlock-free ordering."""
        if self._rank < peer:
            ret = bindings.send(
                send_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclSend to {peer}")
            ret = bindings.recv(
                recv_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclRecv from {peer}")
        else:
            ret = bindings.recv(
                recv_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclRecv from {peer}")
            ret = bindings.send(
                send_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclSend to {peer}")

    def all_to_all(self, output_tensors, input_tensors):
        """All-to-all using native HcclAlltoAll (2 ranks) or P2P (>2 ranks).

        HcclAlltoAll works correctly on 2 ranks but is broken on CANN 8.3.RC2
        for >2 ranks, so we use P2P send/recv fallback for larger world sizes.
        """
        from .._backends.npu import runtime as npu_runtime
        get_bindings, _, _, _, _, _, _, dtype_to_hccl, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ACL_MEMCPY_D2D = 3

        # Fast path for 2 ranks: use native HcclAlltoAll
        if self._size == 2:
            import mindtorch_v2 as torch
            count_per_rank = input_tensors[0].numel()
            dtype = input_tensors[0].dtype
            itemsize = dtype.itemsize

            # Pack into contiguous buffers
            total_count = count_per_rank * 2
            send_flat = torch.empty(total_count, dtype=dtype, device=input_tensors[0].device)
            recv_flat = torch.empty(total_count, dtype=dtype, device=output_tensors[0].device)

            dst_base = send_flat.storage().data_ptr()
            for i, t in enumerate(input_tensors):
                ret = npu_runtime.acl.rt.memcpy(
                    dst_base + i * count_per_rank * itemsize,
                    count_per_rank * itemsize,
                    t.storage().data_ptr(),
                    count_per_rank * itemsize,
                    ACL_MEMCPY_D2D)
                if ret != 0:
                    raise RuntimeError(f"D2D memcpy pack failed: {ret}")

            # Call HcclAlltoAll
            ret = bindings.all_to_all(
                ctypes.c_void_p(send_flat.storage().data_ptr()),
                ctypes.c_uint64(count_per_rank),
                ctypes.c_int32(dtype_to_hccl(dtype)),
                ctypes.c_void_p(recv_flat.storage().data_ptr()),
                ctypes.c_uint64(count_per_rank),
                ctypes.c_int32(dtype_to_hccl(dtype)),
                self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, "HcclAlltoAll")

            # Sync and unpack
            dev_id = self._device_id if self._device_id is not None else 0
            npu_runtime.get_runtime(dev_id).synchronize_stream(stream)

            src_base = recv_flat.storage().data_ptr()
            for i, t in enumerate(output_tensors):
                ret = npu_runtime.acl.rt.memcpy(
                    t.storage().data_ptr(),
                    count_per_rank * itemsize,
                    src_base + i * count_per_rank * itemsize,
                    count_per_rank * itemsize,
                    ACL_MEMCPY_D2D)
                if ret != 0:
                    raise RuntimeError(f"D2D memcpy unpack failed: {ret}")

            return self._make_work(stream)

        # Fallback for >2 ranks: use P2P
        for peer in range(self._size):
            numel = input_tensors[peer].numel()
            hccl_dt = ctypes.c_int32(dtype_to_hccl(input_tensors[peer].dtype))
            itemsize = input_tensors[peer].dtype.itemsize
            if peer == self._rank:
                nbytes = numel * itemsize
                ret = npu_runtime.acl.rt.memcpy(
                    output_tensors[peer].storage().data_ptr(), nbytes,
                    input_tensors[peer].storage().data_ptr(), nbytes,
                    ACL_MEMCPY_D2D)
                if ret != 0:
                    raise RuntimeError(f"D2D memcpy failed: {ret}")
            else:
                send_ptr = ctypes.c_void_p(input_tensors[peer].storage().data_ptr())
                recv_ptr = ctypes.c_void_p(output_tensors[peer].storage().data_ptr())
                self._p2p_exchange(send_ptr, recv_ptr, numel, hccl_dt,
                                   peer, stream, bindings, _check)

        return self._make_work(stream)

    def all_to_all_single(self, output, input, count_per_rank):
        """All-to-all on contiguous buffers using native API (2 ranks) or P2P (>2 ranks)."""
        from .._backends.npu import runtime as npu_runtime
        get_bindings, _, _, _, _, _, _, dtype_to_hccl, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()

        # Fast path for 2 ranks: use native HcclAlltoAll directly
        if self._size == 2:
            ret = bindings.all_to_all(
                ctypes.c_void_p(input.storage().data_ptr()),
                ctypes.c_uint64(count_per_rank),
                ctypes.c_int32(dtype_to_hccl(input.dtype)),
                ctypes.c_void_p(output.storage().data_ptr()),
                ctypes.c_uint64(count_per_rank),
                ctypes.c_int32(dtype_to_hccl(output.dtype)),
                self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, "HcclAlltoAll")
            return self._make_work(stream)

        # Fallback for >2 ranks: use P2P
        ACL_MEMCPY_D2D = 3
        hccl_dt = ctypes.c_int32(dtype_to_hccl(input.dtype))
        itemsize = input.dtype.itemsize
        chunk_bytes = count_per_rank * itemsize
        in_base = input.storage().data_ptr()
        out_base = output.storage().data_ptr()

        for peer in range(self._size):
            if peer == self._rank:
                ret = npu_runtime.acl.rt.memcpy(
                    out_base + peer * chunk_bytes, chunk_bytes,
                    in_base + peer * chunk_bytes, chunk_bytes,
                    ACL_MEMCPY_D2D)
                if ret != 0:
                    raise RuntimeError(f"D2D memcpy failed: {ret}")
            else:
                send_ptr = ctypes.c_void_p(in_base + peer * chunk_bytes)
                recv_ptr = ctypes.c_void_p(out_base + peer * chunk_bytes)
                self._p2p_exchange(send_ptr, recv_ptr, count_per_rank,
                                   hccl_dt, peer, stream, bindings, _check)

        return self._make_work(stream)

    def destroy(self):
        if self._comm is not None:
            get_bindings = self._load_bindings()[0]
            bindings = get_bindings()
            bindings.comm_destroy(self._comm)
            self._comm = None
