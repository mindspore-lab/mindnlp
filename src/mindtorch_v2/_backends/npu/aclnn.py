import ctypes
import atexit
import os
import numpy as np

from .acl_loader import ensure_acl

acl = None




def _launch_blocking_enabled():
    value = os.getenv("ACL_LAUNCH_BLOCKING", "").strip().lower()
    return value not in ("", "0", "false", "no")


def _maybe_sync(runtime):
    if runtime is None:
        return
    if _launch_blocking_enabled():
        runtime.synchronize()


_LIB_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
    "/usr/local/Ascend/latest/aarch64-linux/lib64",
    "/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/lib64",
)

_BASE_LIBS = (
    "libnnopbase.so",
)

_PRELOAD_LIBS = (
    "libopapi.so",
)

_LIBS = (
    "libaclnn_ops_infer.so",
    "libaclnn_math.so",
    "libopapi.so",
)

_LIB_HANDLES = None
_BINDINGS = None
_ACLNN_INITIALIZED = False
_ACLNN_FINALIZED = False
_DEFERRED_EXECUTORS = []
_CLEANUP_REGISTERED = False


class AclnnBindings:
    def __init__(self, libs):
        self.libs = libs
        self.aclnn_init = _bind_symbol(
            libs,
            "aclnnInit",
            ctypes.c_int32,
            [ctypes.c_char_p],
        )
        self.aclnn_finalize = _optional_symbol(
            libs,
            "aclnnFinalize",
            ctypes.c_int32,
            [],
        )
        self.acl_create_tensor = _bind_symbol(
            libs,
            "aclCreateTensor",
            ctypes.c_void_p,
            [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_uint64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_uint64,
                ctypes.c_void_p,
            ],
        )
        self.acl_create_scalar = _bind_symbol(
            libs,
            "aclCreateScalar",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_int32],
        )
        self.acl_create_int_array = _bind_symbol(
            libs,
            "aclCreateIntArray",
            ctypes.c_void_p,
            [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64],
        )
        self.acl_destroy_tensor = _bind_symbol(
            libs,
            "aclDestroyTensor",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.acl_destroy_scalar = _bind_symbol(
            libs,
            "aclDestroyScalar",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.acl_destroy_int_array = _bind_symbol(
            libs,
            "aclDestroyIntArray",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.acl_destroy_executor = _bind_symbol(
            libs,
            "aclDestroyAclOpExecutor",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        self.aclnn_add_get_workspace = _bind_symbol(
            libs,
            "aclnnAddGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_add = _bind_symbol(
            libs,
            "aclnnAdd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_mul_get_workspace = _bind_symbol(
            libs,
            "aclnnMulGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_mul = _bind_symbol(
            libs,
            "aclnnMul",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_relu_get_workspace = _bind_symbol(
            libs,
            "aclnnReluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_relu = _bind_symbol(
            libs,
            "aclnnRelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_abs_get_workspace = _optional_symbol(
            libs,
            "aclnnAbsGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_abs = _optional_symbol(
            libs,
            "aclnnAbs",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_neg_get_workspace = _optional_symbol(
            libs,
            "aclnnNegGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_neg = _optional_symbol(
            libs,
            "aclnnNeg",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_exp_get_workspace = _optional_symbol(
            libs,
            "aclnnExpGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_exp = _optional_symbol(
            libs,
            "aclnnExp",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log_get_workspace = _optional_symbol(
            libs,
            "aclnnLogGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log = _optional_symbol(
            libs,
            "aclnnLog",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sqrt_get_workspace = _optional_symbol(
            libs,
            "aclnnSqrtGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sqrt = _optional_symbol(
            libs,
            "aclnnSqrt",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_rsqrt_get_workspace = _optional_symbol(
            libs,
            "aclnnRsqrtGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_rsqrt = _optional_symbol(
            libs,
            "aclnnRsqrt",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sin_get_workspace = _optional_symbol(
            libs,
            "aclnnSinGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sin = _optional_symbol(
            libs,
            "aclnnSin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cos_get_workspace = _optional_symbol(
            libs,
            "aclnnCosGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_cos = _optional_symbol(
            libs,
            "aclnnCos",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_tan_get_workspace = _optional_symbol(
            libs,
            "aclnnTanGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_tan = _optional_symbol(
            libs,
            "aclnnTan",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_tanh_get_workspace = _optional_symbol(
            libs,
            "aclnnTanhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_tanh = _optional_symbol(
            libs,
            "aclnnTanh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sigmoid_get_workspace = _optional_symbol(
            libs,
            "aclnnSigmoidGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sigmoid = _optional_symbol(
            libs,
            "aclnnSigmoid",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_floor_get_workspace = _optional_symbol(
            libs,
            "aclnnFloorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_floor = _optional_symbol(
            libs,
            "aclnnFloor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_ceil_get_workspace = _optional_symbol(
            libs,
            "aclnnCeilGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_ceil = _optional_symbol(
            libs,
            "aclnnCeil",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_round_get_workspace = _optional_symbol(
            libs,
            "aclnnRoundGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_round = _optional_symbol(
            libs,
            "aclnnRound",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_trunc_get_workspace = _optional_symbol(
            libs,
            "aclnnTruncGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_trunc = _optional_symbol(
            libs,
            "aclnnTrunc",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_frac_get_workspace = _optional_symbol(
            libs,
            "aclnnFracGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_frac = _optional_symbol(
            libs,
            "aclnnFrac",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log2_get_workspace = _optional_symbol(
            libs,
            "aclnnLog2GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log2 = _optional_symbol(
            libs,
            "aclnnLog2",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log10_get_workspace = _optional_symbol(
            libs,
            "aclnnLog10GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log10 = _optional_symbol(
            libs,
            "aclnnLog10",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_exp2_get_workspace = _optional_symbol(
            libs,
            "aclnnExp2GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_exp2 = _optional_symbol(
            libs,
            "aclnnExp2",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_pow_tensor_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnPowTensorTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_pow_tensor_tensor = _optional_symbol(
            libs,
            "aclnnPowTensorTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_pow_tensor_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnPowTensorScalarGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_pow_tensor_scalar = _optional_symbol(
            libs,
            "aclnnPowTensorScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_reduce_sum_get_workspace = _bind_symbol(
            libs,
            "aclnnReduceSumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_reduce_sum = _bind_symbol(
            libs,
            "aclnnReduceSum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_matmul_get_workspace = _optional_symbol(
            libs,
            "aclnnMatmulGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int8,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_matmul = _optional_symbol(
            libs,
            "aclnnMatmul",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_batch_matmul_get_workspace = _optional_symbol(
            libs,
            "aclnnBatchMatMulGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int8,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_batch_matmul = _optional_symbol(
            libs,
            "aclnnBatchMatMul",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_inplace_one_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceOneGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_one = _optional_symbol(
            libs,
            "aclnnInplaceOne",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_inplace_zero_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceZeroGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_zero = _optional_symbol(
            libs,
            "aclnnInplaceZero",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )


_ACL_DTYPE = {
    "float32": 0,
    "float16": 1,
    "float64": 11,
    "bfloat16": 27,
    "int8": 2,
    "int16": 6,
    "int32": 3,
    "int64": 9,
    "uint8": 4,
    "bool": 12,
    "complex64": 16,
    "complex128": 17,
}

_ACL_FORMAT_ND = 2

_NP_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bfloat16": np.uint16,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "bool": np.bool_,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


def _normalize_dtype(dtype):
    name = getattr(dtype, "name", None)
    if name is not None:
        return name
    return str(dtype)


def _dtype_to_acl(dtype):
    dtype = _normalize_dtype(dtype)
    if dtype not in _ACL_DTYPE:
        raise ValueError(f"Unsupported dtype for ACLNN: {dtype}")
    return _ACL_DTYPE[dtype]


def _numpy_scalar(value, dtype):
    dtype = _normalize_dtype(dtype)
    return np.array(value, dtype=_NP_DTYPE[dtype])


def _make_int64_array(values):
    if not values:
        return None
    data = (ctypes.c_int64 * len(values))()
    for i, v in enumerate(values):
        data[i] = int(v)
    return data


def _create_tensor(bindings, shape, stride, dtype, data_ptr):
    view_dims = _make_int64_array(shape)
    stride_dims = _make_int64_array(stride)
    storage_dims = _make_int64_array(shape)
    view_num = ctypes.c_uint64(len(shape))
    storage_num = ctypes.c_uint64(len(shape))
    tensor = bindings.acl_create_tensor(
        view_dims,
        view_num,
        ctypes.c_int32(_dtype_to_acl(dtype)),
        stride_dims,
        ctypes.c_int64(0),
        ctypes.c_int32(_ACL_FORMAT_ND),
        storage_dims,
        storage_num,
        ctypes.c_void_p(int(data_ptr)),
    )
    if not tensor:
        raise RuntimeError("aclCreateTensor returned null")
    keepalive = (view_dims, stride_dims, storage_dims)
    return tensor, keepalive


def _create_scalar(bindings, value, dtype):
    arr = _numpy_scalar(value, dtype)
    ptr = ctypes.c_void_p(int(arr.ctypes.data))
    scalar = bindings.acl_create_scalar(ptr, ctypes.c_int32(_dtype_to_acl(dtype)))
    if not scalar:
        raise RuntimeError("aclCreateScalar returned null")
    return scalar, arr


def _bind_symbol(libs, name, restype, argtypes):
    for lib in libs:
        if hasattr(lib, name):
            func = getattr(lib, name)
            func.restype = restype
            func.argtypes = argtypes
            return func
    raise AttributeError(f"ACLNN symbol not found: {name}")


def _optional_symbol(libs, name, restype, argtypes):
    try:
        return _bind_symbol(libs, name, restype, argtypes)
    except AttributeError:
        return None


def _load_libs():
    global _LIB_HANDLES
    if _LIB_HANDLES is not None:
        return _LIB_HANDLES
    libs = []
    for lib_name in _BASE_LIBS:
        lib_path = None
        for base in _LIB_DIRS:
            candidate = os.path.join(base, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break
        if lib_path is None:
            raise FileNotFoundError(f"ACLNN base library not found: {lib_name}")
        libs.append(ctypes.CDLL(lib_path))
    for lib_name in _PRELOAD_LIBS:
        lib_path = None
        for base in _LIB_DIRS:
            candidate = os.path.join(base, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break
        if lib_path is not None:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    for lib_name in _LIBS:
        lib_path = None
        for base in _LIB_DIRS:
            candidate = os.path.join(base, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break
        if lib_path is None:
            raise FileNotFoundError(f"ACLNN library not found: {lib_name}")
        libs.append(ctypes.CDLL(lib_path))
    _LIB_HANDLES = libs
    return libs


def _init_aclnn(bindings):
    global _ACLNN_INITIALIZED
    if _ACLNN_INITIALIZED:
        return
    ret = bindings.aclnn_init(None)
    if ret != 0:
        raise RuntimeError(f"aclnnInit failed: {ret}")
    _ACLNN_INITIALIZED = True
    _register_cleanup()




def _register_cleanup():
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    atexit.register(_cleanup_aclnn)
    _CLEANUP_REGISTERED = True


def _cleanup_aclnn():
    global _ACLNN_FINALIZED, _DEFERRED_EXECUTORS
    if _ACLNN_FINALIZED:
        return
    bindings = _BINDINGS
    if bindings is not None:
        for executor in _DEFERRED_EXECUTORS:
            try:
                bindings.acl_destroy_executor(executor)
            except Exception:
                pass
        _DEFERRED_EXECUTORS = []
    _ACLNN_FINALIZED = True


def _defer_executor(executor):
    if not executor:
        return
    try:
        if int(executor) == 0:
            return
    except Exception:
        return
    _register_cleanup()
    _DEFERRED_EXECUTORS.append(executor)


def get_bindings():
    global _BINDINGS
    if _BINDINGS is None:
        libs = _load_libs()
        _BINDINGS = AclnnBindings(libs)
        _init_aclnn(_BINDINGS)
    return _BINDINGS


def symbols_ok():
    try:
        bindings = get_bindings()
        required = [
            bindings.aclnn_add_get_workspace,
            bindings.aclnn_add,
            bindings.aclnn_mul_get_workspace,
            bindings.aclnn_mul,
            bindings.aclnn_relu_get_workspace,
            bindings.aclnn_relu,
            bindings.aclnn_reduce_sum_get_workspace,
            bindings.aclnn_reduce_sum,
        ]
        return all(required)
    except Exception:
        return False


def is_available():
    try:
        _load_libs()
        return True
    except Exception:
        return False


def add(self_ptr, other_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, shape, stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    scalar = None
    alpha_arr = None
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        scalar, alpha_arr = _create_scalar(bindings, 1, dtype)
        ret = bindings.aclnn_add_get_workspace(
            self_tensor,
            other_tensor,
            scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAddGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_add(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAdd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if scalar:
            bindings.acl_destroy_scalar(scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep, alpha_arr)


def mul(self_ptr, other_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, shape, stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_mul_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMulGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_mul(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMul failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def relu(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_relu_get_workspace(
            self_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnReluGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_relu(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRelu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)




def inplace_one(out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_one_get_workspace is None or bindings.aclnn_inplace_one is None:
        raise RuntimeError("aclnnInplaceOne not available")
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_inplace_one_get_workspace(
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceOneGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_inplace_one(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceOne failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = out_keep


def inplace_zero(out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_zero_get_workspace is None or bindings.aclnn_inplace_zero is None:
        raise RuntimeError("aclnnInplaceZero not available")
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_inplace_zero_get_workspace(
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceZeroGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_inplace_zero(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceZero failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = out_keep


def reduce_sum(self_ptr, out_ptr, shape, stride, dtype, dims, keepdim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, dims["out_shape"], dims["out_stride"], dtype, out_ptr)
    dim_array = None
    dim_handle = None
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        dim_values = dims["dims"]
        if dim_values is None:
            dim_values = []
        dim_array = _make_int64_array(dim_values)
        dim_handle = bindings.acl_create_int_array(dim_array, ctypes.c_uint64(len(dim_values)))
        if not dim_handle:
            raise RuntimeError("aclCreateIntArray returned null")
        ret = bindings.aclnn_reduce_sum_get_workspace(
            self_tensor,
            dim_handle,
            ctypes.c_bool(keepdim),
            ctypes.c_int32(_dtype_to_acl(dtype)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnReduceSumGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_reduce_sum(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnReduceSum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if dim_handle:
            bindings.acl_destroy_int_array(dim_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, dim_array)


def _unary_call(bindings, name, get_workspace_fn, exec_fn, self_ptr, out_ptr, shape, stride, dtype, runtime, stream):
    if get_workspace_fn is None or exec_fn is None:
        raise RuntimeError(f"{name} symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = get_workspace_fn(
            self_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"{name}GetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = exec_fn(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"{name} failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def abs(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnAbs", bindings.aclnn_abs_get_workspace, bindings.aclnn_abs,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def neg(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnNeg", bindings.aclnn_neg_get_workspace, bindings.aclnn_neg,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def exp(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnExp", bindings.aclnn_exp_get_workspace, bindings.aclnn_exp,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLog", bindings.aclnn_log_get_workspace, bindings.aclnn_log,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sqrt(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSqrt", bindings.aclnn_sqrt_get_workspace, bindings.aclnn_sqrt,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def rsqrt(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnRsqrt", bindings.aclnn_rsqrt_get_workspace, bindings.aclnn_rsqrt,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sin(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSin", bindings.aclnn_sin_get_workspace, bindings.aclnn_sin,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def cos(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnCos", bindings.aclnn_cos_get_workspace, bindings.aclnn_cos,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def tan(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnTan", bindings.aclnn_tan_get_workspace, bindings.aclnn_tan,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def tanh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnTanh", bindings.aclnn_tanh_get_workspace, bindings.aclnn_tanh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sigmoid(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSigmoid", bindings.aclnn_sigmoid_get_workspace, bindings.aclnn_sigmoid,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def floor(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnFloor", bindings.aclnn_floor_get_workspace, bindings.aclnn_floor,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def ceil(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnCeil", bindings.aclnn_ceil_get_workspace, bindings.aclnn_ceil,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def round(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnRound", bindings.aclnn_round_get_workspace, bindings.aclnn_round,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def trunc(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnTrunc", bindings.aclnn_trunc_get_workspace, bindings.aclnn_trunc,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def frac(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnFrac", bindings.aclnn_frac_get_workspace, bindings.aclnn_frac,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log2(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLog2", bindings.aclnn_log2_get_workspace, bindings.aclnn_log2,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log10(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLog10", bindings.aclnn_log10_get_workspace, bindings.aclnn_log10,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def exp2(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnExp2", bindings.aclnn_exp2_get_workspace, bindings.aclnn_exp2,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def pow_tensor_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                      out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_pow_tensor_tensor_get_workspace is None or bindings.aclnn_pow_tensor_tensor is None:
        raise RuntimeError("aclnnPowTensorTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_pow_tensor_tensor_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnPowTensorTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_pow_tensor_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnPowTensorTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def pow_tensor_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_pow_tensor_scalar_get_workspace is None or bindings.aclnn_pow_tensor_scalar is None:
        raise RuntimeError("aclnnPowTensorScalar symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    scalar, scalar_keep = _create_scalar(bindings, scalar_value, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_pow_tensor_scalar_get_workspace(
            self_tensor,
            scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnPowTensorScalarGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_pow_tensor_scalar(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnPowTensorScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if scalar:
            bindings.acl_destroy_scalar(scalar)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, scalar_keep)


def matmul(a_ptr, b_ptr, out_ptr, a_shape, a_stride, b_shape, b_stride, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not bindings.aclnn_matmul_get_workspace or not bindings.aclnn_matmul:
        raise RuntimeError("aclnnMatmul symbols not available")
    is_batched = len(a_shape) > 2 or len(b_shape) > 2
    if is_batched:
        if not bindings.aclnn_batch_matmul_get_workspace or not bindings.aclnn_batch_matmul:
            raise RuntimeError("aclnnBatchMatMul symbols not available")
    a_tensor, a_keep = _create_tensor(bindings, a_shape, a_stride, dtype, a_ptr)
    b_tensor, b_keep = _create_tensor(bindings, b_shape, b_stride, dtype, b_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        if is_batched:
            ret = bindings.aclnn_batch_matmul_get_workspace(
                a_tensor,
                b_tensor,
                out_tensor,
                ctypes.c_int8(0),
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
        else:
            ret = bindings.aclnn_matmul_get_workspace(
                a_tensor,
                b_tensor,
                out_tensor,
                ctypes.c_int8(0),
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
        if ret != 0:
            if is_batched:
                raise RuntimeError(f"aclnnBatchMatMulGetWorkspaceSize failed: {ret}")
            raise RuntimeError(f"aclnnMatmulGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        if is_batched:
            ret = bindings.aclnn_batch_matmul(
                ctypes.c_void_p(0 if workspace is None else int(workspace)),
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
            )
        else:
            ret = bindings.aclnn_matmul(
                ctypes.c_void_p(0 if workspace is None else int(workspace)),
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
            )
        if ret != 0:
            if is_batched:
                raise RuntimeError(f"aclnnBatchMatMul failed: {ret}")
            raise RuntimeError(f"aclnnMatmul failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(a_tensor)
        bindings.acl_destroy_tensor(b_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (a_keep, b_keep, out_keep)


def ones_zero_symbols_ok():
    try:
        bindings = get_bindings()
        return all(
            [
                bindings.aclnn_inplace_one_get_workspace,
                bindings.aclnn_inplace_one,
                bindings.aclnn_inplace_zero_get_workspace,
                bindings.aclnn_inplace_zero,
            ]
        )
    except Exception:
        return False
