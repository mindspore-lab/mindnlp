import ctypes

from .hccl_loader import ensure_hccl

HCCL_SUCCESS = 0
HCCL_ROOT_INFO_BYTES = 4108


class HcclRootInfo(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * HCCL_ROOT_INFO_BYTES)]


# HcclDataType enum values
HCCL_DATA_TYPE = {
    "int8": 0, "int16": 1, "int32": 2, "float16": 3, "float32": 4,
    "int64": 5, "uint64": 6, "uint8": 7, "uint16": 8, "uint32": 9,
    "float64": 10, "bfloat16": 11,
}

# HcclReduceOp enum values
HCCL_REDUCE_SUM = 0
HCCL_REDUCE_PROD = 1
HCCL_REDUCE_MAX = 2
HCCL_REDUCE_MIN = 3


def _bind(lib, name, restype, argtypes):
    func = getattr(lib, name)
    func.restype = restype
    func.argtypes = argtypes
    return func


def _check(ret, name):
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"{name} failed with error code {ret}")


_BINDINGS = None


class HcclBindings:
    def __init__(self, lib):
        i32 = ctypes.c_int32
        u32 = ctypes.c_uint32
        u64 = ctypes.c_uint64
        vp = ctypes.c_void_p

        self.get_root_info = _bind(
            lib, "HcclGetRootInfo", i32,
            [ctypes.POINTER(HcclRootInfo)])
        self.comm_init_root_info = _bind(
            lib, "HcclCommInitRootInfo", i32,
            [u32, ctypes.POINTER(HcclRootInfo), u32, ctypes.POINTER(vp)])
        self.all_reduce = _bind(
            lib, "HcclAllReduce", i32,
            [vp, vp, u64, i32, i32, vp, vp])
        self.broadcast = _bind(
            lib, "HcclBroadcast", i32,
            [vp, u64, i32, u32, vp, vp])
        self.all_gather = _bind(
            lib, "HcclAllGather", i32,
            [vp, vp, u64, i32, vp, vp])
        self.reduce_scatter = _bind(
            lib, "HcclReduceScatter", i32,
            [vp, vp, u64, i32, i32, vp, vp])
        self.reduce = _bind(
            lib, "HcclReduce", i32,
            [vp, vp, u64, i32, i32, u32, vp, vp])
        self.scatter = _bind(
            lib, "HcclScatter", i32,
            [vp, vp, u64, i32, u32, vp, vp])
        self.barrier = _bind(
            lib, "HcclBarrier", i32,
            [vp, vp])
        self.send = _bind(
            lib, "HcclSend", i32,
            [vp, u64, i32, u32, vp, vp])
        self.recv = _bind(
            lib, "HcclRecv", i32,
            [vp, u64, i32, u32, vp, vp])
        self.comm_destroy = _bind(
            lib, "HcclCommDestroy", i32,
            [vp])


def get_bindings():
    global _BINDINGS
    if _BINDINGS is None:
        lib = ensure_hccl()
        _BINDINGS = HcclBindings(lib)
    return _BINDINGS


def dtype_to_hccl(dtype):
    name = getattr(dtype, "name", None) or str(dtype)
    if name not in HCCL_DATA_TYPE:
        raise ValueError(f"Unsupported dtype for HCCL: {name}")
    return HCCL_DATA_TYPE[name]
