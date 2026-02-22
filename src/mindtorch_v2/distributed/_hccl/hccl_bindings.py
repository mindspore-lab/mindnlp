import ctypes
import struct

from .hccl_loader import ensure_hccl

HCCL_SUCCESS = 0
HCCL_ROOT_INFO_BYTES = 4108
HCCL_COMM_CONFIG_INFO_BYTES = 24
COMM_NAME_MAX_LENGTH = 128
UDI_MAX_LENGTH = 128
HCCL_COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0
HCCL_COMM_CONFIG_VERSION = 6
HCCL_COMM_DEFAULT_BUFFSIZE = 200
HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xffffffff
HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET = 0xffffffff
HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0
HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xffffffff
HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xffffffff

# HcclCommConfigCapability enum
HCCL_COMM_CONFIG_BUFFER_SIZE = 0
HCCL_COMM_CONFIG_DETERMINISTIC = 1
HCCL_COMM_CONFIG_COMM_NAME = 2
HCCL_COMM_CONFIG_OP_EXPANSION_MODE = 3
HCCL_COMM_CONFIG_SUPPORT_INIT_BY_ENV = 4
HCCL_COMM_CONFIG_WORLD_RANKID = 5
HCCL_COMM_CONFIG_JOBID = 6


class HcclRootInfo(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * HCCL_ROOT_INFO_BYTES)]


class HcclCommConfig(ctypes.Structure):
    _fields_ = [
        ("reserved", ctypes.c_char * HCCL_COMM_CONFIG_INFO_BYTES),
        ("hcclBufferSize", ctypes.c_uint32),
        ("hcclDeterministic", ctypes.c_uint32),
        ("hcclCommName", ctypes.c_char * COMM_NAME_MAX_LENGTH),
        ("hcclUdi", ctypes.c_char * UDI_MAX_LENGTH),
        ("hcclOpExpansionMode", ctypes.c_uint32),
        ("hcclRdmaTrafficClass", ctypes.c_uint32),
        ("hcclRdmaServiceLevel", ctypes.c_uint32),
        ("hcclWorldRankID", ctypes.c_uint32),
        ("hcclJobID", ctypes.c_uint64),
    ]


def hccl_comm_config_init(config):
    """Python equivalent of the inline HcclCommConfigInit() from hccl.h."""
    ctypes.memset(ctypes.byref(config), 0, ctypes.sizeof(config))
    # reserved header: size(8B) + magicWord(4B) + version(4B) + reserved(8B)
    header = struct.pack(
        "<QII Q",
        ctypes.sizeof(HcclCommConfig),  # size = sizeof(HcclCommConfig)
        HCCL_COMM_CONFIG_MAGIC_WORD,
        HCCL_COMM_CONFIG_VERSION,
        0,
    )
    # NOTE: must write to addressof(config), NOT config.reserved
    # (ctypes char array attribute access returns a temporary copy)
    ctypes.memmove(ctypes.addressof(config), header, len(header))
    config.hcclBufferSize = HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET
    config.hcclDeterministic = HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET
    config.hcclOpExpansionMode = HCCL_COMM_DEFAULT_OP_EXPANSION_MODE
    config.hcclRdmaTrafficClass = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET
    config.hcclRdmaServiceLevel = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET
    config.hcclWorldRankID = 0
    config.hcclJobID = 0


def is_hccl_feature_supported(capability_bit):
    """Check if an HCCL feature is supported via HcclGetCommConfigCapability."""
    bindings = get_bindings()
    if bindings.get_comm_config_capability is None:
        return False
    cap = bindings.get_comm_config_capability()
    return bool(cap & (1 << capability_bit))


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


def _try_bind(lib, name, restype, argtypes):
    """Bind a function if it exists in the library, return None otherwise."""
    if hasattr(lib, name):
        return _bind(lib, name, restype, argtypes)
    return None


def _check(ret, name):
    if ret != HCCL_SUCCESS:
        # Try to get error string
        bindings = get_bindings()
        if bindings.get_error_string is not None:
            err_str = bindings.get_error_string(ret)
            if err_str:
                raise RuntimeError(f"{name} failed: {err_str} (error code {ret})")
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
        self.comm_init_root_info_config = _bind(
            lib, "HcclCommInitRootInfoConfig", i32,
            [u32, ctypes.POINTER(HcclRootInfo), u32,
             ctypes.POINTER(HcclCommConfig), ctypes.POINTER(vp)])
        self.comm_init_cluster_info = _try_bind(
            lib, "HcclCommInitClusterInfo", i32,
            [ctypes.c_char_p, u32, ctypes.POINTER(vp)])
        self.comm_init_cluster_info_config = _try_bind(
            lib, "HcclCommInitClusterInfoConfig", i32,
            [ctypes.c_char_p, u32, ctypes.POINTER(HcclCommConfig),
             ctypes.POINTER(vp)])
        self.create_sub_comm_config = _try_bind(
            lib, "HcclCreateSubCommConfig", i32,
            [ctypes.POINTER(vp), u32, ctypes.POINTER(u32), u64, u32,
             ctypes.POINTER(HcclCommConfig), ctypes.POINTER(vp)])
        self.comm_init_all = _try_bind(
            lib, "HcclCommInitAll", i32,
            [u32, ctypes.POINTER(i32), ctypes.POINTER(vp)])
        self.get_comm_config_capability = _try_bind(
            lib, "HcclGetCommConfigCapability", u32, [])
        self.get_error_string = _try_bind(
            lib, "HcclGetErrorString", ctypes.c_char_p, [i32])
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
