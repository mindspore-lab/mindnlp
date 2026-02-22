import ctypes
import atexit
import os
import struct

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
        self.aclnn_sign_get_workspace = _optional_symbol(
            libs,
            "aclnnSignGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sign = _optional_symbol(
            libs,
            "aclnnSign",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_signbit_get_workspace = _optional_symbol(
            libs,
            "aclnnSignbitGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_signbit = _optional_symbol(
            libs,
            "aclnnSignbit",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        
        self.aclnn_logical_not_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalNotGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_logical_not = _optional_symbol(
            libs,
            "aclnnLogicalNot",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_and_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalAndGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_logical_and = _optional_symbol(
            libs,
            "aclnnLogicalAnd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_or_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalOrGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_logical_or = _optional_symbol(
            libs,
            "aclnnLogicalOr",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_xor_get_workspace = _optional_symbol(
            libs,
            "aclnnLogicalXorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_logical_xor = _optional_symbol(
            libs,
            "aclnnLogicalXor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isfinite_get_workspace = _optional_symbol(
            libs,
            "aclnnIsFiniteGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isfinite = _optional_symbol(
            libs,
            "aclnnIsFinite",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isinf_get_workspace = _optional_symbol(
            libs,
            "aclnnIsInfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isinf = _optional_symbol(
            libs,
            "aclnnIsInf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isposinf_get_workspace = _optional_symbol(
            libs,
            "aclnnIsPosInfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isposinf = _optional_symbol(
            libs,
            "aclnnIsPosInf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_isneginf_get_workspace = _optional_symbol(
            libs,
            "aclnnIsNegInfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_isneginf = _optional_symbol(
            libs,
            "aclnnIsNegInf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_ne_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnNeTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_ne_tensor = _optional_symbol(
            libs,
            "aclnnNeTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_eq_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnEqTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_eq_tensor = _optional_symbol(
            libs,
            "aclnnEqTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_eq_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnEqScalarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_eq_scalar = _optional_symbol(
            libs,
            "aclnnEqScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_argmax_get_workspace = _optional_symbol(
            libs,
            "aclnnArgMaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_argmax = _optional_symbol(
            libs,
            "aclnnArgMax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_argmin_get_workspace = _optional_symbol(
            libs,
            "aclnnArgMinGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_argmin = _optional_symbol(
            libs,
            "aclnnArgMin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_max_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_dim = _optional_symbol(
            libs,
            "aclnnMaxDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_min_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnMinDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_min_dim = _optional_symbol(
            libs,
            "aclnnMinDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cast_get_workspace = _optional_symbol(
            libs,
            "aclnnCastGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cast = _optional_symbol(
            libs,
            "aclnnCast",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cosh_get_workspace = _optional_symbol(
            libs,
            "aclnnCoshGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_cosh = _optional_symbol(
            libs,
            "aclnnCosh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sinh_get_workspace = _optional_symbol(
            libs,
            "aclnnSinhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sinh = _optional_symbol(
            libs,
            "aclnnSinh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_erf_get_workspace = _optional_symbol(
            libs,
            "aclnnErfGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_erf = _optional_symbol(
            libs,
            "aclnnErf",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_erfc_get_workspace = _optional_symbol(
            libs,
            "aclnnErfcGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_erfc = _optional_symbol(
            libs,
            "aclnnErfc",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_softplus_get_workspace = _optional_symbol(
            libs,
            "aclnnSoftplusGetWorkspaceSize",
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
        self.aclnn_softplus = _optional_symbol(
            libs,
            "aclnnSoftplus",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_hardtanh_get_workspace = _optional_symbol(
            libs,
            "aclnnHardtanhGetWorkspaceSize",
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
        self.aclnn_hardtanh = _optional_symbol(
            libs,
            "aclnnHardtanh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_get_workspace = _optional_symbol(
            libs,
            "aclnnClampGetWorkspaceSize",
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
        self.aclnn_clamp = _optional_symbol(
            libs,
            "aclnnClamp",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_min_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMinGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_min = _optional_symbol(
            libs,
            "aclnnClampMin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_max_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_max = _optional_symbol(
            libs,
            "aclnnClampMax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnClampTensorGetWorkspaceSize",
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
        self.aclnn_clamp_tensor = _optional_symbol(
            libs,
            "aclnnClampTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_min_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMinTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_min_tensor = _optional_symbol(
            libs,
            "aclnnClampMinTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_clamp_max_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnClampMaxTensorGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_clamp_max_tensor = _optional_symbol(
            libs,
            "aclnnClampMaxTensor",
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


def _float32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _float_to_float16_bits(value):
    f32 = _float32_bits(value)
    sign = (f32 >> 31) & 0x1
    exponent = (f32 >> 23) & 0xFF
    mantissa = f32 & 0x7FFFFF
    if exponent == 0xFF:
        half_exp = 0x1F
        half_mant = 0x200 if mantissa != 0 else 0
    elif exponent > 142:
        half_exp = 0x1F
        half_mant = 0
    elif exponent < 113:
        if exponent < 103:
            half_exp = 0
            half_mant = 0
        else:
            shift = 113 - exponent
            mantissa = mantissa | 0x800000
            half_mant = mantissa >> (shift + 13)
            round_bit = (mantissa >> (shift + 12)) & 1
            sticky = mantissa & ((1 << (shift + 12)) - 1)
            if round_bit and (sticky or (half_mant & 1)):
                half_mant += 1
            half_exp = 0
            if half_mant == 0x400:
                half_exp = 1
                half_mant = 0
    else:
        half_exp = exponent - 112
        half_mant = mantissa >> 13
        round_bit = (mantissa >> 12) & 1
        sticky = mantissa & 0xFFF
        if round_bit and (sticky or (half_mant & 1)):
            half_mant += 1
            if half_mant == 0x400:
                half_mant = 0
                half_exp += 1
                if half_exp >= 0x1F:
                    half_exp = 0x1F
                    half_mant = 0
    return (sign << 15) | (half_exp << 10) | half_mant


def _float_to_bfloat16_bits(value):
    bits = _float32_bits(value)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16) & 0xFFFF


def _scalar_bytes(value, dtype):
    dtype = _normalize_dtype(dtype)
    if dtype == "float16":
        bits = _float_to_float16_bits(float(value))
        return int(bits).to_bytes(2, byteorder="little", signed=False)
    if dtype == "bfloat16":
        bits = _float_to_bfloat16_bits(float(value))
        return int(bits).to_bytes(2, byteorder="little", signed=False)
    if dtype == "float32":
        return struct.pack("<f", float(value))
    if dtype == "float64":
        return struct.pack("<d", float(value))
    if dtype == "int8":
        return int(value).to_bytes(1, byteorder="little", signed=True)
    if dtype == "uint8":
        return int(value).to_bytes(1, byteorder="little", signed=False)
    if dtype == "int16":
        return int(value).to_bytes(2, byteorder="little", signed=True)
    if dtype == "int32":
        return int(value).to_bytes(4, byteorder="little", signed=True)
    if dtype == "int64":
        return int(value).to_bytes(8, byteorder="little", signed=True)
    if dtype == "bool":
        return (1 if bool(value) else 0).to_bytes(1, byteorder="little", signed=False)
    raise ValueError(f"Unsupported scalar dtype for ACLNN: {dtype}")


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
    data = _scalar_bytes(value, dtype)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    ptr = ctypes.c_void_p(ctypes.addressof(buf))
    scalar = bindings.acl_create_scalar(ptr, ctypes.c_int32(_dtype_to_acl(dtype)))
    if not scalar:
        raise RuntimeError("aclCreateScalar returned null")
    return scalar, buf


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





def argmax(self_ptr, out_ptr, shape, stride, dtype, dim, keepdim, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_argmax_get_workspace is None or bindings.aclnn_argmax is None:
        raise RuntimeError("aclnnArgMax symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "int64", out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_argmax_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            ctypes.c_bool(keepdim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArgMaxGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_argmax(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArgMax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def argmin(self_ptr, out_ptr, shape, stride, dtype, dim, keepdim, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_argmin_get_workspace is None or bindings.aclnn_argmin is None:
        raise RuntimeError("aclnnArgMin symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "int64", out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_argmin_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            ctypes.c_bool(keepdim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArgMinGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_argmin(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArgMin failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def max_dim(self_ptr, out_ptr, indices_ptr, shape, stride, dtype, dim, keepdim,
            out_shape, out_stride, index_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_max_dim_get_workspace is None or bindings.aclnn_max_dim is None:
        raise RuntimeError("aclnnMaxDim symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    index_tensor, index_keep = _create_tensor(bindings, out_shape, index_stride, "int64", indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_max_dim_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            ctypes.c_bool(keepdim),
            out_tensor,
            index_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxDimGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_max_dim(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxDim failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, index_keep)


def min_dim(self_ptr, out_ptr, indices_ptr, shape, stride, dtype, dim, keepdim,
            out_shape, out_stride, index_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_min_dim_get_workspace is None or bindings.aclnn_min_dim is None:
        raise RuntimeError("aclnnMinDim symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    index_tensor, index_keep = _create_tensor(bindings, out_shape, index_stride, "int64", indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_min_dim_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            ctypes.c_bool(keepdim),
            out_tensor,
            index_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMinDimGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_min_dim(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMinDim failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, index_keep)



def cast(self_ptr, out_ptr, shape, stride, src_dtype, dst_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cast_get_workspace is None or bindings.aclnn_cast is None:
        raise RuntimeError("aclnnCast symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, src_dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dst_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_cast_get_workspace(
            self_tensor,
            ctypes.c_int32(_dtype_to_acl(dst_dtype)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCastGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_cast(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCast failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)
def _unary_call(bindings, name, get_workspace_fn, exec_fn, self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype=None):
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    if out_dtype is None:
        out_dtype = dtype
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, out_dtype, out_ptr)
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


def sign(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSign", bindings.aclnn_sign_get_workspace, bindings.aclnn_sign,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)










def logical_xor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_logical_xor_get_workspace is None or bindings.aclnn_logical_xor is None:
        raise RuntimeError("aclnnLogicalXor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_logical_xor_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogicalXorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_logical_xor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogicalXor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)

def logical_or(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
               out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_logical_or_get_workspace is None or bindings.aclnn_logical_or is None:
        raise RuntimeError("aclnnLogicalOr symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_logical_or_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogicalOrGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_logical_or(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogicalOr failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)

def logical_and(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_logical_and_get_workspace is None or bindings.aclnn_logical_and is None:
        raise RuntimeError("aclnnLogicalAnd symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_logical_and_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogicalAndGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_logical_and(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogicalAnd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)

def logical_not(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLogicalNot", bindings.aclnn_logical_not_get_workspace,
                       bindings.aclnn_logical_not, self_ptr, out_ptr, shape, stride, dtype, runtime, stream)

def signbit(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSignbit", bindings.aclnn_signbit_get_workspace, bindings.aclnn_signbit,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")


def isfinite(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsFinite", bindings.aclnn_isfinite_get_workspace, bindings.aclnn_isfinite,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")


def isinf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsInf", bindings.aclnn_isinf_get_workspace, bindings.aclnn_isinf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")



def isposinf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsPosInf", bindings.aclnn_isposinf_get_workspace, bindings.aclnn_isposinf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")


def isneginf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnIsNegInf", bindings.aclnn_isneginf_get_workspace, bindings.aclnn_isneginf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream, out_dtype="bool")

def cosh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnCosh", bindings.aclnn_cosh_get_workspace, bindings.aclnn_cosh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def sinh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnSinh", bindings.aclnn_sinh_get_workspace, bindings.aclnn_sinh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def erf(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnErf", bindings.aclnn_erf_get_workspace, bindings.aclnn_erf,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def erfc(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnErfc", bindings.aclnn_erfc_get_workspace, bindings.aclnn_erfc,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def softplus(self_ptr, out_ptr, shape, stride, dtype, beta, threshold, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_softplus_get_workspace is None or bindings.aclnn_softplus is None:
        raise RuntimeError("aclnnSoftplus symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    beta_scalar, beta_keep = _create_scalar(bindings, beta, dtype)
    threshold_scalar, threshold_keep = _create_scalar(bindings, threshold, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_softplus_get_workspace(
            self_tensor,
            beta_scalar,
            threshold_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSoftplusGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_softplus(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSoftplus failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_scalar(beta_scalar)
        bindings.acl_destroy_scalar(threshold_scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, beta_keep, threshold_keep)


def hardtanh(self_ptr, out_ptr, shape, stride, dtype, min_val, max_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_hardtanh_get_workspace is None or bindings.aclnn_hardtanh is None:
        raise RuntimeError("aclnnHardtanh symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    min_scalar, min_keep = _create_scalar(bindings, min_val, dtype)
    max_scalar, max_keep = _create_scalar(bindings, max_val, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_hardtanh_get_workspace(
            self_tensor,
            min_scalar,
            max_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnHardtanhGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_hardtanh(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnHardtanh failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_scalar(min_scalar)
        bindings.acl_destroy_scalar(max_scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, min_keep, max_keep)


def clamp_scalar(self_ptr, out_ptr, shape, stride, dtype, min_val, max_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_get_workspace is None or bindings.aclnn_clamp is None:
        raise RuntimeError("aclnnClamp symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    min_scalar = None
    max_scalar = None
    min_keep = None
    max_keep = None
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        if min_val is not None:
            min_scalar, min_keep = _create_scalar(bindings, min_val, dtype)
        if max_val is not None:
            max_scalar, max_keep = _create_scalar(bindings, max_val, dtype)
        ret = bindings.aclnn_clamp_get_workspace(
            self_tensor,
            min_scalar,
            max_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_clamp(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClamp failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if min_scalar is not None:
            bindings.acl_destroy_scalar(min_scalar)
        if max_scalar is not None:
            bindings.acl_destroy_scalar(max_scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, min_keep, max_keep)


def clamp_min_scalar(self_ptr, out_ptr, shape, stride, dtype, min_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_min_get_workspace is None or bindings.aclnn_clamp_min is None:
        raise RuntimeError("aclnnClampMin symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    min_scalar, min_keep = _create_scalar(bindings, min_val, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_clamp_min_get_workspace(
            self_tensor,
            min_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMinGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_clamp_min(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMin failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_scalar(min_scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, min_keep)


def clamp_max_scalar(self_ptr, out_ptr, shape, stride, dtype, max_val, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_max_get_workspace is None or bindings.aclnn_clamp_max is None:
        raise RuntimeError("aclnnClampMax symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    max_scalar, max_keep = _create_scalar(bindings, max_val, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_clamp_max_get_workspace(
            self_tensor,
            max_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMaxGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_clamp_max(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_scalar(max_scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, max_keep)


def clamp_tensor(self_ptr, min_ptr, max_ptr, out_ptr, self_shape, self_stride, min_shape, min_stride,
                 max_shape, max_stride, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_tensor_get_workspace is None or bindings.aclnn_clamp_tensor is None:
        raise RuntimeError("aclnnClampTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    min_tensor, min_keep = _create_tensor(bindings, min_shape, min_stride, dtype, min_ptr)
    max_tensor, max_keep = _create_tensor(bindings, max_shape, max_stride, dtype, max_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_clamp_tensor_get_workspace(
            self_tensor,
            min_tensor,
            max_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_clamp_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(min_tensor)
        bindings.acl_destroy_tensor(max_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, min_keep, max_keep, out_keep)


def clamp_min_tensor(self_ptr, min_ptr, out_ptr, self_shape, self_stride, min_shape, min_stride,
                     out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_min_tensor_get_workspace is None or bindings.aclnn_clamp_min_tensor is None:
        raise RuntimeError("aclnnClampMinTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    min_tensor, min_keep = _create_tensor(bindings, min_shape, min_stride, dtype, min_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_clamp_min_tensor_get_workspace(
            self_tensor,
            min_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMinTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_clamp_min_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMinTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(min_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, min_keep, out_keep)


def clamp_max_tensor(self_ptr, max_ptr, out_ptr, self_shape, self_stride, max_shape, max_stride,
                     out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_clamp_max_tensor_get_workspace is None or bindings.aclnn_clamp_max_tensor is None:
        raise RuntimeError("aclnnClampMaxTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    max_tensor, max_keep = _create_tensor(bindings, max_shape, max_stride, dtype, max_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_clamp_max_tensor_get_workspace(
            self_tensor,
            max_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMaxTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_clamp_max_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnClampMaxTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(max_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, max_keep, out_keep)




def eq_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_eq_scalar_get_workspace is None or bindings.aclnn_eq_scalar is None:
        raise RuntimeError("aclnnEqScalar symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, 'bool', out_ptr)
    scalar, scalar_keep = _create_scalar(bindings, scalar_value, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_eq_scalar_get_workspace(
            self_tensor,
            scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEqScalarGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_eq_scalar(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEqScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_scalar(scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, scalar_keep)



def eq_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_eq_tensor_get_workspace is None or bindings.aclnn_eq_tensor is None:
        raise RuntimeError("aclnnEqTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_eq_tensor_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEqTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_eq_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEqTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)

def ne_tensor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
              out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_ne_tensor_get_workspace is None or bindings.aclnn_ne_tensor is None:
        raise RuntimeError("aclnnNeTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_ne_tensor_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnNeTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_ne_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnNeTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)


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


def trunc_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_trunc_get_workspace, bindings.aclnn_trunc])
    except Exception:
        return False


def frac_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_frac_get_workspace, bindings.aclnn_frac])
    except Exception:
        return False


def sign_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_sign_get_workspace, bindings.aclnn_sign])
    except Exception:
        return False


def signbit_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_signbit_get_workspace, bindings.aclnn_signbit])
    except Exception:
        return False


def ne_tensor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_ne_tensor_get_workspace, bindings.aclnn_ne_tensor])
    except Exception:
        return False

def logical_not_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_not_get_workspace, bindings.aclnn_logical_not])
    except Exception:
        return False

def logical_and_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_and_get_workspace, bindings.aclnn_logical_and])
    except Exception:
        return False

def eq_scalar_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_eq_scalar_get_workspace, bindings.aclnn_eq_scalar])
    except Exception:
        return False

def logical_or_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_or_get_workspace, bindings.aclnn_logical_or])
    except Exception:
        return False

def logical_xor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_logical_xor_get_workspace, bindings.aclnn_logical_xor])
    except Exception:
        return False

def eq_tensor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_eq_tensor_get_workspace, bindings.aclnn_eq_tensor])
    except Exception:
        return False


def argmax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_argmax_get_workspace, bindings.aclnn_argmax])
    except Exception:
        return False


def argmin_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_argmin_get_workspace, bindings.aclnn_argmin])
    except Exception:
        return False


def max_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_max_dim_get_workspace, bindings.aclnn_max_dim])
    except Exception:
        return False


def min_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_min_dim_get_workspace, bindings.aclnn_min_dim])
    except Exception:
        return False

def cast_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cast_get_workspace, bindings.aclnn_cast])
    except Exception:
        return False

def isposinf_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_isposinf_get_workspace, bindings.aclnn_isposinf])
    except Exception:
        return False


def isneginf_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_isneginf_get_workspace, bindings.aclnn_isneginf])
    except Exception:
        return False
