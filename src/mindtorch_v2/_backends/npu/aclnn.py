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
        self.acl_create_bool_array = _bind_symbol(
            libs,
            "aclCreateBoolArray",
            ctypes.c_void_p,
            [ctypes.POINTER(ctypes.c_bool), ctypes.c_uint64],
        )
        self.acl_destroy_bool_array = _bind_symbol(
            libs,
            "aclDestroyBoolArray",
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
        self.aclnn_arange_get_workspace = _optional_symbol(
            libs,
            "aclnnArangeGetWorkspaceSize",
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
        self.aclnn_arange = _optional_symbol(
            libs,
            "aclnnArange",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_linspace_get_workspace = _optional_symbol(
            libs,
            "aclnnLinspaceGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_linspace = _optional_symbol(
            libs,
            "aclnnLinspace",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_eye_get_workspace = _optional_symbol(
            libs,
            "aclnnEyeGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_eye = _optional_symbol(
            libs,
            "aclnnEye",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_range_get_workspace = _optional_symbol(
            libs,
            "aclnnRangeGetWorkspaceSize",
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
        self.aclnn_range = _optional_symbol(
            libs,
            "aclnnRange",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_flip_get_workspace = _optional_symbol(
            libs,
            "aclnnFlipGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_flip = _optional_symbol(
            libs,
            "aclnnFlip",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_roll_get_workspace = _optional_symbol(
            libs,
            "aclnnRollGetWorkspaceSize",
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
        self.aclnn_roll = _optional_symbol(
            libs,
            "aclnnRoll",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cumsum_get_workspace = _optional_symbol(
            libs,
            "aclnnCumsumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cumsum = _optional_symbol(
            libs,
            "aclnnCumsum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cumprod_get_workspace = _optional_symbol(
            libs,
            "aclnnCumprodGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cumprod = _optional_symbol(
            libs,
            "aclnnCumprod",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_cummax_get_workspace = _optional_symbol(
            libs,
            "aclnnCummaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cummax = _optional_symbol(
            libs,
            "aclnnCummax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_argsort_get_workspace = _optional_symbol(
            libs,
            "aclnnArgsortGetWorkspaceSize",
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
        self.aclnn_argsort = _optional_symbol(
            libs,
            "aclnnArgsort",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sort_get_workspace = _optional_symbol(
            libs,
            "aclnnSortGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_sort = _optional_symbol(
            libs,
            "aclnnSort",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_topk_get_workspace = _optional_symbol(
            libs,
            "aclnnTopkGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_topk = _optional_symbol(
            libs,
            "aclnnTopk",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_tril_get_workspace = _optional_symbol(
            libs,
            "aclnnTrilGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_tril = _optional_symbol(
            libs,
            "aclnnTril",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_triu_get_workspace = _optional_symbol(
            libs,
            "aclnnTriuGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_triu = _optional_symbol(
            libs,
            "aclnnTriu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_nonzero_get_workspace = _optional_symbol(
            libs,
            "aclnnNonzeroGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_nonzero = _optional_symbol(
            libs,
            "aclnnNonzero",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat = _optional_symbol(
            libs,
            "aclnnRepeat",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_interleave_int_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveIntGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat_interleave_int = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveInt",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_repeat_interleave_int_with_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveIntWithDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_repeat_interleave_int_with_dim = _optional_symbol(
            libs,
            "aclnnRepeatInterleaveIntWithDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_scatter_get_workspace = _optional_symbol(
            libs,
            "aclnnScatterGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_scatter = _optional_symbol(
            libs,
            "aclnnScatter",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_diag_get_workspace = _optional_symbol(
            libs,
            "aclnnDiagGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_diag = _optional_symbol(
            libs,
            "aclnnDiag",
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
        self.aclnn_index_put_impl_get_workspace = _optional_symbol(
            libs,
            "aclnnIndexPutImplGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_index_put_impl = _optional_symbol(
            libs,
            "aclnnIndexPutImpl",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnIndex (advanced indexing getitem)
        self.aclnn_index_get_workspace = _optional_symbol(
            libs,
            "aclnnIndexGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_index = _optional_symbol(
            libs,
            "aclnnIndex",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSlice (strided slicing on any dim)
        self.aclnn_slice_get_workspace = _optional_symbol(
            libs,
            "aclnnSliceGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_slice = _optional_symbol(
            libs,
            "aclnnSlice",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceMaskedFillScalar
        self.aclnn_inplace_masked_fill_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceMaskedFillScalarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_void_p,      # const aclTensor* mask
                ctypes.c_void_p,      # const aclScalar* value
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_masked_fill_scalar = _optional_symbol(
            libs,
            "aclnnInplaceMaskedFillScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceIndexCopy
        self.aclnn_inplace_index_copy_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceIndexCopyGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclTensor* source
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_index_copy = _optional_symbol(
            libs,
            "aclnnInplaceIndexCopy",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceIndexFill (scalar value variant)
        self.aclnn_inplace_index_fill_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceIndexFillGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclScalar* value
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_index_fill = _optional_symbol(
            libs,
            "aclnnInplaceIndexFill",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnIndexAdd
        self.aclnn_index_add_get_workspace = _optional_symbol(
            libs,
            "aclnnIndexAddGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # const aclTensor* self
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclTensor* source
                ctypes.c_void_p,      # const aclScalar* alpha
                ctypes.c_void_p,      # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_index_add = _optional_symbol(
            libs,
            "aclnnIndexAdd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnScatterAdd
        self.aclnn_scatter_add_get_workspace = _optional_symbol(
            libs,
            "aclnnScatterAddGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # const aclTensor* self
                ctypes.c_int64,       # int64_t dim
                ctypes.c_void_p,      # const aclTensor* index
                ctypes.c_void_p,      # const aclTensor* src
                ctypes.c_void_p,      # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_scatter_add = _optional_symbol(
            libs,
            "aclnnScatterAdd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnInplaceMaskedScatter
        self.aclnn_inplace_masked_scatter_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceMaskedScatterGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,      # aclTensor* selfRef
                ctypes.c_void_p,      # const aclTensor* mask
                ctypes.c_void_p,      # const aclTensor* source
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_masked_scatter = _optional_symbol(
            libs,
            "aclnnInplaceMaskedScatter",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        self.aclnn_sub_get_workspace = _optional_symbol(
            libs,
            "aclnnSubGetWorkspaceSize",
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
        self.aclnn_sub = _optional_symbol(
            libs,
            "aclnnSub",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_div_get_workspace = _optional_symbol(
            libs,
            "aclnnDivGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_div = _optional_symbol(
            libs,
            "aclnnDiv",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_add_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnAddsGetWorkspaceSize",
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
        self.aclnn_add_scalar = _optional_symbol(
            libs,
            "aclnnAdds",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_sub_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnSubsGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_sub_scalar = _optional_symbol(
            libs,
            "aclnnSubs",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_maximum_get_workspace = _optional_symbol(
            libs,
            "aclnnMaximumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_maximum = _optional_symbol(
            libs,
            "aclnnMaximum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_minimum_get_workspace = _optional_symbol(
            libs,
            "aclnnMinimumGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_minimum = _optional_symbol(
            libs,
            "aclnnMinimum",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_atan_get_workspace = _optional_symbol(
            libs,
            "aclnnAtanGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_atan = _optional_symbol(
            libs,
            "aclnnAtan",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_atan2_get_workspace = _optional_symbol(
            libs,
            "aclnnAtan2GetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_atan2 = _optional_symbol(
            libs,
            "aclnnAtan2",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_asin_get_workspace = _optional_symbol(
            libs,
            "aclnnAsinGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_asin = _optional_symbol(
            libs,
            "aclnnAsin",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_acos_get_workspace = _optional_symbol(
            libs,
            "aclnnAcosGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_acos = _optional_symbol(
            libs,
            "aclnnAcos",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_asinh_get_workspace = _optional_symbol(
            libs,
            "aclnnAsinhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_asinh = _optional_symbol(
            libs,
            "aclnnAsinh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_acosh_get_workspace = _optional_symbol(
            libs,
            "aclnnAcoshGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_acosh = _optional_symbol(
            libs,
            "aclnnAcosh",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_atanh_get_workspace = _optional_symbol(
            libs,
            "aclnnAtanhGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_atanh = _optional_symbol(
            libs,
            "aclnnAtanh",
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
        self.aclnn_expm1_get_workspace = _optional_symbol(
            libs,
            "aclnnExpm1GetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_expm1 = _optional_symbol(
            libs,
            "aclnnExpm1",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_log1p_get_workspace = _optional_symbol(
            libs,
            "aclnnLog1pGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log1p = _optional_symbol(
            libs,
            "aclnnLog1p",
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

        self.aclnn_swhere_get_workspace = _optional_symbol(
            libs,
            "aclnnSWhereGetWorkspaceSize",
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
        self.aclnn_swhere = _optional_symbol(
            libs,
            "aclnnSWhere",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_logical_xor = _optional_symbol(
            libs,
            "aclnnLogicalXor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Bitwise ops
        self.aclnn_bitwise_not_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseNotGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_not = _optional_symbol(
            libs,
            "aclnnBitwiseNot",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_bitwise_and_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseAndTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_and_tensor = _optional_symbol(
            libs,
            "aclnnBitwiseAndTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_bitwise_or_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseOrTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_or_tensor = _optional_symbol(
            libs,
            "aclnnBitwiseOrTensor",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_bitwise_xor_tensor_get_workspace = _optional_symbol(
            libs,
            "aclnnBitwiseXorTensorGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_bitwise_xor_tensor = _optional_symbol(
            libs,
            "aclnnBitwiseXorTensor",
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
        # Dot: aclnnDotGetWorkspaceSize(self, tensor, out, workspaceSize, executor)
        self.aclnn_dot_get_workspace = _optional_symbol(
            libs,
            "aclnnDotGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_dot = _optional_symbol(
            libs,
            "aclnnDot",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Mv: aclnnMvGetWorkspaceSize(self, vec, out, cubeMathType, workspaceSize, executor)
        self.aclnn_mv_get_workspace = _optional_symbol(
            libs,
            "aclnnMvGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int8, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_mv = _optional_symbol(
            libs,
            "aclnnMv",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Ger: aclnnGerGetWorkspaceSize(self, vec2, out, workspaceSize, executor)
        self.aclnn_ger_get_workspace = _optional_symbol(
            libs,
            "aclnnGerGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_ger = _optional_symbol(
            libs,
            "aclnnGer",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Median (global): aclnnMedianGetWorkspaceSize(self, valuesOut, workspaceSize, executor)
        self.aclnn_median_get_workspace = _optional_symbol(
            libs,
            "aclnnMedianGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_median = _optional_symbol(
            libs,
            "aclnnMedian",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # MedianDim: aclnnMedianDimGetWorkspaceSize(self, dim, keepDim, valuesOut, indicesOut, workspaceSize, executor)
        self.aclnn_median_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnMedianDimGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_median_dim = _optional_symbol(
            libs,
            "aclnnMedianDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Kthvalue: aclnnKthvalueGetWorkspaceSize(self, k, dim, keepdim, valuesOut, indicesOut, workspaceSize, executor)
        self.aclnn_kthvalue_get_workspace = _optional_symbol(
            libs,
            "aclnnKthvalueGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_kthvalue = _optional_symbol(
            libs,
            "aclnnKthvalue",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # SearchSorted: aclnnSearchSortedGetWorkspaceSize(sortedSequence, self, outInt32, right, sorter, out, workspaceSize, executor)
        self.aclnn_search_sorted_get_workspace = _optional_symbol(
            libs,
            "aclnnSearchSortedGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_search_sorted = _optional_symbol(
            libs,
            "aclnnSearchSorted",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Unique: aclnnUniqueGetWorkspaceSize(self, sorted, returnInverse, valueOut, inverseOut, workspaceSize, executor)
        self.aclnn_unique_get_workspace = _optional_symbol(
            libs,
            "aclnnUniqueGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_unique = _optional_symbol(
            libs,
            "aclnnUnique",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Randperm: aclnnRandpermGetWorkspaceSize(n, seed, offset, out, workspaceSize, executor)
        self.aclnn_randperm_get_workspace = _optional_symbol(
            libs,
            "aclnnRandpermGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_randperm = _optional_symbol(
            libs,
            "aclnnRandperm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Flatten: aclnnFlattenGetWorkspaceSize(self, axis, out, workspaceSize, executor)
        # NOTE: aclnnFlatten always produces 2D output, different from torch.flatten
        self.aclnn_flatten_get_workspace = _optional_symbol(
            libs,
            "aclnnFlattenGetWorkspaceSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_flatten = _optional_symbol(
            libs,
            "aclnnFlatten",
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
        # TensorList support
        self.acl_create_tensor_list = _optional_symbol(
            libs,
            "aclCreateTensorList",
            ctypes.c_void_p,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64],
        )
        self.acl_destroy_tensor_list = _optional_symbol(
            libs,
            "aclDestroyTensorList",
            ctypes.c_int32,
            [ctypes.c_void_p],
        )
        # Cat
        self.aclnn_cat_get_workspace = _optional_symbol(
            libs,
            "aclnnCatGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # tensorList
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_cat = _optional_symbol(
            libs,
            "aclnnCat",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Stack
        self.aclnn_stack_get_workspace = _optional_symbol(
            libs,
            "aclnnStackGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # tensorList
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_stack = _optional_symbol(
            libs,
            "aclnnStack",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        # Where (SWhere = Select Where)
        self.aclnn_s_where_get_workspace = _optional_symbol(
            libs,
            "aclnnSWhereGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # condition
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # other
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_s_where = _optional_symbol(
            libs,
            "aclnnSWhere",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Mean
        self.aclnn_mean_get_workspace = _optional_symbol(
            libs,
            "aclnnMeanGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # dim (IntArray)
                ctypes.c_bool,    # keepdim
                ctypes.c_int32,   # dtype
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_mean = _optional_symbol(
            libs,
            "aclnnMean",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Softmax
        self.aclnn_softmax_get_workspace = _optional_symbol(
            libs,
            "aclnnSoftmaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_softmax = _optional_symbol(
            libs,
            "aclnnSoftmax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LogSoftmax
        self.aclnn_log_softmax_get_workspace = _optional_symbol(
            libs,
            "aclnnLogSoftmaxGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_log_softmax = _optional_symbol(
            libs,
            "aclnnLogSoftmax",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Gelu
        self.aclnn_gelu_get_workspace = _optional_symbol(
            libs,
            "aclnnGeluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_gelu = _optional_symbol(
            libs,
            "aclnnGelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LayerNorm
        self.aclnn_layer_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnLayerNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # normalizedShape (IntArray)
                ctypes.c_void_p,  # weight (optional)
                ctypes.c_void_p,  # bias (optional)
                ctypes.c_double,  # eps
                ctypes.c_void_p,  # out
                ctypes.c_void_p,  # mean (optional)
                ctypes.c_void_p,  # rstd (optional)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_layer_norm = _optional_symbol(
            libs,
            "aclnnLayerNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Silu
        self.aclnn_silu_get_workspace = _optional_symbol(
            libs,
            "aclnnSiluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_silu = _optional_symbol(
            libs,
            "aclnnSilu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LeakyRelu
        self.aclnn_leaky_relu_get_workspace = _optional_symbol(
            libs,
            "aclnnLeakyReluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # negative_slope (Scalar)
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_leaky_relu = _optional_symbol(
            libs,
            "aclnnLeakyRelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Elu
        self.aclnn_elu_get_workspace = _optional_symbol(
            libs,
            "aclnnEluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # alpha (Scalar)
                ctypes.c_void_p,  # scale (Scalar)
                ctypes.c_void_p,  # input_scale (Scalar)
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_elu = _optional_symbol(
            libs,
            "aclnnElu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Mish
        self.aclnn_mish_get_workspace = _optional_symbol(
            libs,
            "aclnnMishGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_mish = _optional_symbol(
            libs,
            "aclnnMish",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Prelu
        self.aclnn_prelu_get_workspace = _optional_symbol(
            libs,
            "aclnnPreluGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # weight
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_prelu = _optional_symbol(
            libs,
            "aclnnPrelu",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # BatchNorm
        self.aclnn_batch_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnBatchNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # input
                ctypes.c_void_p,  # weight (optional)
                ctypes.c_void_p,  # bias (optional)
                ctypes.c_void_p,  # running_mean (optional)
                ctypes.c_void_p,  # running_var (optional)
                ctypes.c_bool,    # training
                ctypes.c_double,  # momentum
                ctypes.c_double,  # eps
                ctypes.c_void_p,  # out
                ctypes.c_void_p,  # save_mean (optional)
                ctypes.c_void_p,  # save_invstd (optional)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_batch_norm = _optional_symbol(
            libs,
            "aclnnBatchNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # GroupNorm
        self.aclnn_group_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnGroupNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self (input)
                ctypes.c_void_p,  # gamma (weight, optional)
                ctypes.c_void_p,  # beta (bias, optional)
                ctypes.c_int64,   # N (batch size)
                ctypes.c_int64,   # C (channels)
                ctypes.c_int64,   # HxW (spatial dimensions)
                ctypes.c_int64,   # group (num_groups)
                ctypes.c_double,  # eps
                ctypes.c_void_p,  # out
                ctypes.c_void_p,  # meanOut (optional)
                ctypes.c_void_p,  # rstdOut (optional)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_group_norm = _optional_symbol(
            libs,
            "aclnnGroupNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Gather
        self.aclnn_gather_get_workspace = _optional_symbol(
            libs,
            "aclnnGatherGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # index
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_gather = _optional_symbol(
            libs,
            "aclnnGather",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # ConstantPadNd
        self.aclnn_constant_pad_nd_get_workspace = _optional_symbol(
            libs,
            "aclnnConstantPadNdGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # pad (IntArray)
                ctypes.c_void_p,  # value (Scalar)
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_constant_pad_nd = _optional_symbol(
            libs,
            "aclnnConstantPadNd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Gather
        self.aclnn_gather_get_workspace = _optional_symbol(
            libs,
            "aclnnGatherGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # dim
                ctypes.c_void_p,  # index
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_gather = _optional_symbol(
            libs,
            "aclnnGather",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # MaskedSelect
        self.aclnn_masked_select_get_workspace = _optional_symbol(
            libs,
            "aclnnMaskedSelectGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # mask
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_masked_select = _optional_symbol(
            libs,
            "aclnnMaskedSelect",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Embedding
        self.aclnn_embedding_get_workspace = _optional_symbol(
            libs,
            "aclnnEmbeddingGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # weight
                ctypes.c_void_p,  # indices
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_embedding = _optional_symbol(
            libs,
            "aclnnEmbedding",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Dropout GenMask + DoMask (two-step dropout)
        self.aclnn_dropout_gen_mask_get_workspace = _optional_symbol(
            libs,
            "aclnnDropoutGenMaskGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # shape (IntArray)
                ctypes.c_double,  # prob
                ctypes.c_int64,   # seed
                ctypes.c_int64,   # offset
                ctypes.c_void_p,  # out (uint8 mask)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_dropout_gen_mask = _optional_symbol(
            libs,
            "aclnnDropoutGenMask",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.aclnn_dropout_do_mask_get_workspace = _optional_symbol(
            libs,
            "aclnnDropoutDoMaskGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self (input)
                ctypes.c_void_p,  # mask
                ctypes.c_double,  # prob
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_dropout_do_mask = _optional_symbol(
            libs,
            "aclnnDropoutDoMask",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # InplaceNormal (for randn)
        self.aclnn_inplace_normal_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceNormalGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_float,   # mean
                ctypes.c_float,   # std
                ctypes.c_int64,   # seed
                ctypes.c_int64,   # offset
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_normal = _optional_symbol(
            libs,
            "aclnnInplaceNormal",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # InplaceUniform (for rand)
        self.aclnn_inplace_uniform_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceUniformGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_double,  # from
                ctypes.c_double,  # to
                ctypes.c_int64,   # seed
                ctypes.c_int64,   # offset
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_uniform = _optional_symbol(
            libs,
            "aclnnInplaceUniform",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # InplaceFillScalar (for fill_)
        self.aclnn_inplace_fill_scalar_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceFillScalarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # value (acl scalar)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_fill_scalar = _optional_symbol(
            libs,
            "aclnnInplaceFillScalar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Copy (for copy_)
        self.aclnn_inplace_copy_get_workspace = _optional_symbol(
            libs,
            "aclnnInplaceCopyGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # dst (self)
                ctypes.c_void_p,  # src
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_inplace_copy = _optional_symbol(
            libs,
            "aclnnInplaceCopy",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # Erfinv (for erfinv_)
        self.aclnn_erfinv_get_workspace = _optional_symbol(
            libs,
            "aclnnErfinvGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_void_p,  # out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_erfinv = _optional_symbol(
            libs,
            "aclnnErfinv",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # LinalgQr (for torch.linalg.qr)
        self.aclnn_linalg_qr_get_workspace = _optional_symbol(
            libs,
            "aclnnLinalgQrGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,  # self
                ctypes.c_int64,   # mode
                ctypes.c_void_p,  # Q out
                ctypes.c_void_p,  # R out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_linalg_qr = _optional_symbol(
            libs,
            "aclnnLinalgQr",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnVar: (self, dim:IntArray, unbiased:bool, keepdim:bool, out) -> status
        self.aclnn_var_get_workspace = _optional_symbol(
            libs,
            "aclnnVarGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* dim
                ctypes.c_bool,                     # bool unbiased
                ctypes.c_bool,                     # bool keepdim
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_var = _optional_symbol(
            libs,
            "aclnnVar",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnNorm: (self, p:Scalar, dim:IntArray, keepdim:bool, out) -> status
        self.aclnn_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclScalar* pScalar
                ctypes.c_void_p,                   # const aclIntArray* dim
                ctypes.c_bool,                     # bool keepdim
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_norm = _optional_symbol(
            libs,
            "aclnnNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnProd: (self, dtype:DataType, out) -> status (all-reduce)
        self.aclnn_prod_get_workspace = _optional_symbol(
            libs,
            "aclnnProdGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_int32,                    # aclDataType dtype
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_prod = _optional_symbol(
            libs,
            "aclnnProd",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnProdDim: (self, dim:int64, keepDim:bool, dtype:DataType, out) -> status
        self.aclnn_prod_dim_get_workspace = _optional_symbol(
            libs,
            "aclnnProdDimGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_int64,                    # int64_t dim
                ctypes.c_bool,                     # bool keepDim
                ctypes.c_int32,                    # aclDataType dtype
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_prod_dim = _optional_symbol(
            libs,
            "aclnnProdDim",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnFloorDivide: (self, other, out) -> status
        self.aclnn_floor_divide_get_workspace = _optional_symbol(
            libs,
            "aclnnFloorDivideGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclTensor* other
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_floor_divide = _optional_symbol(
            libs,
            "aclnnFloorDivide",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnRmsNorm: (x, gamma, epsilon:double, yOut, rstdOut) -> status
        self.aclnn_rms_norm_get_workspace = _optional_symbol(
            libs,
            "aclnnRmsNormGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* x
                ctypes.c_void_p,                   # const aclTensor* gamma
                ctypes.c_double,                   # double epsilon
                ctypes.c_void_p,                   # const aclTensor* yOut
                ctypes.c_void_p,                   # const aclTensor* rstdOut
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_rms_norm = _optional_symbol(
            libs,
            "aclnnRmsNorm",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnConvolution
        self.aclnn_convolution_get_workspace = _optional_symbol(
            libs,
            "aclnnConvolutionGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* input
                ctypes.c_void_p,                   # const aclTensor* weight
                ctypes.c_void_p,                   # const aclTensor* bias (nullable)
                ctypes.c_void_p,                   # const aclIntArray* stride
                ctypes.c_void_p,                   # const aclIntArray* padding
                ctypes.c_void_p,                   # const aclIntArray* dilation
                ctypes.c_bool,                     # bool transposed
                ctypes.c_void_p,                   # const aclIntArray* outputPadding
                ctypes.c_int64,                    # int64_t groups
                ctypes.c_void_p,                   # aclTensor* output
                ctypes.c_int8,                     # int8_t cubeMathType
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_convolution = _optional_symbol(
            libs,
            "aclnnConvolution",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool
        self.aclnn_max_pool_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxPoolGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelShape
                ctypes.c_void_p,                   # const aclIntArray* strides
                ctypes.c_int64,                    # int64_t autoPad
                ctypes.c_void_p,                   # const aclIntArray* pads
                ctypes.c_void_p,                   # const aclIntArray* dilations
                ctypes.c_int64,                    # int64_t ceilMode
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_pool = _optional_symbol(
            libs,
            "aclnnMaxPool",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool2dWithIndices — supports fp32/fp16/bf16, preferred over aclnnMaxPool
        self.aclnn_max_pool2d_with_indices_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithIndicesGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelSize
                ctypes.c_void_p,                   # const aclIntArray* stride
                ctypes.c_void_p,                   # const aclIntArray* padding
                ctypes.c_void_p,                   # const aclIntArray* dilation
                ctypes.c_bool,                     # bool ceilMode
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.c_void_p,                   # aclTensor* indices
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_pool2d_with_indices = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithIndices",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool2dWithMask — used on Ascend910B (pre-910C), supports fp32/fp16
        # indices are int8 mask tensors (not actual position indices)
        self.aclnn_max_pool2d_with_mask_get_workspace = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithMaskGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelSize
                ctypes.c_void_p,                   # const aclIntArray* stride
                ctypes.c_void_p,                   # const aclIntArray* padding
                ctypes.c_void_p,                   # const aclIntArray* dilation
                ctypes.c_bool,                     # bool ceilMode
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.c_void_p,                   # aclTensor* indices (mask)
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_max_pool2d_with_mask = _optional_symbol(
            libs,
            "aclnnMaxPool2dWithMask",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAvgPool2d
        self.aclnn_avg_pool2d_get_workspace = _optional_symbol(
            libs,
            "aclnnAvgPool2dGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* kernelSize
                ctypes.c_void_p,                   # const aclIntArray* strides
                ctypes.c_void_p,                   # const aclIntArray* paddings
                ctypes.c_bool,                     # bool ceilMode
                ctypes.c_bool,                     # bool countIncludePad
                ctypes.c_int64,                    # int64_t divisorOverride
                ctypes.c_int8,                     # int8_t cubeMathType
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_avg_pool2d = _optional_symbol(
            libs,
            "aclnnAvgPool2d",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAdaptiveAvgPool2d
        self.aclnn_adaptive_avg_pool2d_get_workspace = _optional_symbol(
            libs,
            "aclnnAdaptiveAvgPool2dGetWorkspaceSize",
            ctypes.c_int32,
            [
                ctypes.c_void_p,                   # const aclTensor* self
                ctypes.c_void_p,                   # const aclIntArray* outputSize
                ctypes.c_void_p,                   # aclTensor* out
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.aclnn_adaptive_avg_pool2d = _optional_symbol(
            libs,
            "aclnnAdaptiveAvgPool2d",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # ---- Backward kernel bindings ----

        # aclnnSoftmaxBackward(gradOutput, output, dim, gradInput)
        self.aclnn_softmax_backward_get_workspace = _optional_symbol(
            libs, "aclnnSoftmaxBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_softmax_backward = _optional_symbol(
            libs, "aclnnSoftmaxBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnLogSoftmaxBackward(gradOutput, output, dim, gradInput)
        self.aclnn_log_softmax_backward_get_workspace = _optional_symbol(
            libs, "aclnnLogSoftmaxBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_log_softmax_backward = _optional_symbol(
            libs, "aclnnLogSoftmaxBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnGeluBackward(gradOutput, self, gradInput)
        self.aclnn_gelu_backward_get_workspace = _optional_symbol(
            libs, "aclnnGeluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_gelu_backward = _optional_symbol(
            libs, "aclnnGeluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnLayerNormBackward(gradOut, input, normalizedShape, mean, rstd,
        #                        weight, bias, outputMask, gradInput, gradWeight, gradBias)
        self.aclnn_layer_norm_backward_get_workspace = _optional_symbol(
            libs, "aclnnLayerNormBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOut, input, normalizedShape
             ctypes.c_void_p, ctypes.c_void_p,  # mean, rstd
             ctypes.c_void_p, ctypes.c_void_p,  # weight, bias
             ctypes.c_void_p,  # outputMask (aclBoolArray)
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradInput, gradWeight, gradBias
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_layer_norm_backward = _optional_symbol(
            libs, "aclnnLayerNormBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnThresholdBackward(gradOutput, self, threshold, gradInput) — relu backward
        self.aclnn_threshold_backward_get_workspace = _optional_symbol(
            libs, "aclnnThresholdBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_threshold_backward = _optional_symbol(
            libs, "aclnnThresholdBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSiluBackward(gradOutput, self, gradInput)
        self.aclnn_silu_backward_get_workspace = _optional_symbol(
            libs, "aclnnSiluBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_silu_backward = _optional_symbol(
            libs, "aclnnSiluBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnSigmoidBackward(gradOutput, output, gradInput)
        self.aclnn_sigmoid_backward_get_workspace = _optional_symbol(
            libs, "aclnnSigmoidBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_sigmoid_backward = _optional_symbol(
            libs, "aclnnSigmoidBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnTanhBackward(gradOutput, output, gradInput)
        self.aclnn_tanh_backward_get_workspace = _optional_symbol(
            libs, "aclnnTanhBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_tanh_backward = _optional_symbol(
            libs, "aclnnTanhBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnConvolutionBackward
        self.aclnn_convolution_backward_get_workspace = _optional_symbol(
            libs, "aclnnConvolutionBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, input, weight
             ctypes.c_void_p,  # biasSizes (IntArray, nullable)
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # stride, padding, dilation
             ctypes.c_bool,  # transposed
             ctypes.c_void_p,  # outputPadding
             ctypes.c_int64,  # groups
             ctypes.c_void_p,  # outputMask (aclBoolArray)
             ctypes.c_int8,  # cubeMathType
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradInput, gradWeight, gradBias
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_convolution_backward = _optional_symbol(
            libs, "aclnnConvolutionBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnMaxPool2dWithMaskBackward
        self.aclnn_max_pool2d_with_mask_backward_get_workspace = _optional_symbol(
            libs, "aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, input, mask
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # ksize, stride, pad, dilation
             ctypes.c_bool,  # ceilMode
             ctypes.c_void_p,  # gradInput
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_max_pool2d_with_mask_backward = _optional_symbol(
            libs, "aclnnMaxPool2dWithMaskBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAvgPool2dBackward
        self.aclnn_avg_pool2d_backward_get_workspace = _optional_symbol(
            libs, "aclnnAvgPool2dBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, self
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # kernelSize, stride, padding
             ctypes.c_bool, ctypes.c_bool,  # ceilMode, countIncludePad
             ctypes.c_int64,  # divisorOverride (int64_t, 0 means no override)
             ctypes.c_int8,  # cubeMathType
             ctypes.c_void_p,  # gradInput
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_avg_pool2d_backward = _optional_symbol(
            libs, "aclnnAvgPool2dBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnBatchNormBackward
        self.aclnn_batch_norm_backward_get_workspace = _optional_symbol(
            libs, "aclnnBatchNormBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradOut, input, weight
             ctypes.c_void_p, ctypes.c_void_p,  # runningMean, runningVar
             ctypes.c_void_p, ctypes.c_void_p,  # saveMean, saveInvstd
             ctypes.c_bool, ctypes.c_double,  # train, eps
             ctypes.c_void_p,  # outputMask (aclBoolArray)
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gradInput, gradWeight, gradBias
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_batch_norm_backward = _optional_symbol(
            libs, "aclnnBatchNormBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnEmbeddingDenseBackward
        self.aclnn_embedding_dense_backward_get_workspace = _optional_symbol(
            libs, "aclnnEmbeddingDenseBackwardGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,  # gradOutput, indices
             ctypes.c_int64, ctypes.c_int64, ctypes.c_bool,  # numWeights, paddingIdx, scaleGradByFreq
             ctypes.c_void_p,  # gradWeight
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_embedding_dense_backward = _optional_symbol(
            libs, "aclnnEmbeddingDenseBackward", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnRmsNormGrad(dy, x, rstd, gamma, dx, dgamma)
        self.aclnn_rms_norm_grad_get_workspace = _optional_symbol(
            libs, "aclnnRmsNormGradGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # dy, x, rstd, gamma
             ctypes.c_void_p, ctypes.c_void_p,  # dx, dgamma
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_rms_norm_grad = _optional_symbol(
            libs, "aclnnRmsNormGrad", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # ---------------------------------------------------------------
        # P1 missing ops
        # ---------------------------------------------------------------

        # aclnnReciprocal: (self, out) — standard unary
        self.aclnn_reciprocal_get_workspace = _optional_symbol(
            libs, "aclnnReciprocalGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_reciprocal = _optional_symbol(
            libs, "aclnnReciprocal", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnAddmm: (self, mat1, mat2, beta:Scalar, alpha:Scalar, out, cubeMathType)
        self.aclnn_addmm_get_workspace = _optional_symbol(
            libs, "aclnnAddmmGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_int8,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_addmm = _optional_symbol(
            libs, "aclnnAddmm", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnEinsum: (tensorList, equation:char*, output)
        self.aclnn_einsum_get_workspace = _optional_symbol(
            libs, "aclnnEinsumGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_einsum = _optional_symbol(
            libs, "aclnnEinsum", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleNearest2d: (self, outputSize:IntArray, out)
        self.aclnn_upsample_nearest2d_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleNearest2dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_nearest2d = _optional_symbol(
            libs, "aclnnUpsampleNearest2d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnUpsampleBilinear2d: (self, outputSize, alignCorners, scalesH, scalesW, out)
        self.aclnn_upsample_bilinear2d_get_workspace = _optional_symbol(
            libs, "aclnnUpsampleBilinear2dGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
             ctypes.c_double, ctypes.c_double, ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_upsample_bilinear2d = _optional_symbol(
            libs, "aclnnUpsampleBilinear2d", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )

        # aclnnOneHot: (self, numClasses, onValue, offValue, axis, out)
        self.aclnn_one_hot_get_workspace = _optional_symbol(
            libs, "aclnnOneHotGetWorkspaceSize", ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
             ctypes.c_void_p,
             ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
        )
        self.aclnn_one_hot = _optional_symbol(
            libs, "aclnnOneHot", ctypes.c_int32,
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
_ACL_FORMAT_NCHW = 0


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


def _make_bool_array(values):
    """Create a ctypes bool array for aclCreateBoolArray."""
    if not values:
        return None
    data = (ctypes.c_bool * len(values))()
    for i, v in enumerate(values):
        data[i] = bool(v)
    return data


def _create_tensor(bindings, shape, stride, dtype, data_ptr, fmt=_ACL_FORMAT_ND):
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
        ctypes.c_int32(fmt),
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
    # Write a minimal config JSON for aclnnInit.
    # Without this config, some ACLNN ops (e.g. aclnnGroupNorm) can corrupt
    # internal state and cause subsequent ops to fail with error 561103.
    # This matches the pattern used by torch_npu and MindSpore.
    import json, tempfile, os
    config = {"dump": {"dump_scene": "lite_exception"}}
    config_path = os.path.join(tempfile.gettempdir(), "mindtorch_aclnn_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f)
    ret = bindings.aclnn_init(config_path.encode("utf-8"))
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


def add(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
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
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep, alpha_arr)


def mul(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
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
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def sub(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_sub_get_workspace is None or bindings.aclnn_sub is None:
        raise RuntimeError("aclnnSub symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    scalar = None
    alpha_arr = None
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        scalar, alpha_arr = _create_scalar(bindings, 1, dtype)
        ret = bindings.aclnn_sub_get_workspace(
            self_tensor,
            other_tensor,
            scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSubGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_sub(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSub failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if scalar:
            bindings.acl_destroy_scalar(scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep, alpha_arr)


def div(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_div_get_workspace is None or bindings.aclnn_div is None:
        raise RuntimeError("aclnnDiv symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_div_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDivGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_div(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDiv failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def add_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_add_scalar_get_workspace is None or bindings.aclnn_add_scalar is None:
        raise RuntimeError("aclnnAdds symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    scalar, scalar_keep = _create_scalar(bindings, scalar_value, dtype)
    alpha, alpha_keep = _create_scalar(bindings, 1, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_add_scalar_get_workspace(
            self_tensor,
            scalar,
            alpha,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAddsGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_add_scalar(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAdds failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_scalar(scalar)
        bindings.acl_destroy_scalar(alpha)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep, scalar_keep)


def sub_scalar(self_ptr, scalar_value, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_sub_scalar_get_workspace is None or bindings.aclnn_sub_scalar is None:
        raise RuntimeError("aclnnSubs symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    scalar, scalar_keep = _create_scalar(bindings, scalar_value, dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_sub_scalar_get_workspace(
            self_tensor,
            scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSubsGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_sub_scalar(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSubs failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_scalar(scalar)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep, scalar_keep)


def maximum(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride, out_shape, out_stride,
            dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_maximum_get_workspace is None or bindings.aclnn_maximum is None:
        raise RuntimeError("aclnnMaximum symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_maximum_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaximumGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_maximum(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaximum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def minimum(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride, out_shape, out_stride,
            dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_minimum_get_workspace is None or bindings.aclnn_minimum is None:
        raise RuntimeError("aclnnMinimum symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_minimum_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMinimumGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_minimum(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMinimum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def atan(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_atan_get_workspace is None or bindings.aclnn_atan is None:
        raise RuntimeError("aclnnAtan symbols not available")
    return _unary_call(bindings, "aclnnAtan", bindings.aclnn_atan_get_workspace, bindings.aclnn_atan,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def atan2(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride, out_shape, out_stride,
          dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_atan2_get_workspace is None or bindings.aclnn_atan2 is None:
        raise RuntimeError("aclnnAtan2 symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_atan2_get_workspace(
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAtan2GetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_atan2(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAtan2 failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def asin(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_asin_get_workspace is None or bindings.aclnn_asin is None:
        raise RuntimeError("aclnnAsin symbols not available")
    return _unary_call(bindings, "aclnnAsin", bindings.aclnn_asin_get_workspace, bindings.aclnn_asin,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def acos(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_acos_get_workspace is None or bindings.aclnn_acos is None:
        raise RuntimeError("aclnnAcos symbols not available")
    return _unary_call(bindings, "aclnnAcos", bindings.aclnn_acos_get_workspace, bindings.aclnn_acos,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def asinh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_asinh_get_workspace is None or bindings.aclnn_asinh is None:
        raise RuntimeError("aclnnAsinh symbols not available")
    return _unary_call(bindings, "aclnnAsinh", bindings.aclnn_asinh_get_workspace, bindings.aclnn_asinh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def acosh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_acosh_get_workspace is None or bindings.aclnn_acosh is None:
        raise RuntimeError("aclnnAcosh symbols not available")
    return _unary_call(bindings, "aclnnAcosh", bindings.aclnn_acosh_get_workspace, bindings.aclnn_acosh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def atanh(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_atanh_get_workspace is None or bindings.aclnn_atanh is None:
        raise RuntimeError("aclnnAtanh symbols not available")
    return _unary_call(bindings, "aclnnAtanh", bindings.aclnn_atanh_get_workspace, bindings.aclnn_atanh,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep)


def arange(start, end, step, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_arange_get_workspace is None or bindings.aclnn_arange is None:
        raise RuntimeError("aclnnArange symbols not available")
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    start_scalar = None
    end_scalar = None
    step_scalar = None
    start_keep = None
    end_keep = None
    step_keep = None
    try:
        start_scalar, start_keep = _create_scalar(bindings, start, dtype)
        end_scalar, end_keep = _create_scalar(bindings, end, dtype)
        step_scalar, step_keep = _create_scalar(bindings, step, dtype)
        ret = bindings.aclnn_arange_get_workspace(
            start_scalar,
            end_scalar,
            step_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArangeGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_arange(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArange failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(out_tensor)
        if start_scalar is not None:
            bindings.acl_destroy_scalar(start_scalar)
        if end_scalar is not None:
            bindings.acl_destroy_scalar(end_scalar)
        if step_scalar is not None:
            bindings.acl_destroy_scalar(step_scalar)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (out_keep, start_keep, end_keep, step_keep)


def linspace(start, end, steps, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_linspace_get_workspace is None or bindings.aclnn_linspace is None:
        raise RuntimeError("aclnnLinspace symbols not available")
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    start_scalar = None
    end_scalar = None
    start_keep = None
    end_keep = None
    try:
        start_scalar, start_keep = _create_scalar(bindings, start, dtype)
        end_scalar, end_keep = _create_scalar(bindings, end, dtype)
        ret = bindings.aclnn_linspace_get_workspace(
            start_scalar,
            end_scalar,
            ctypes.c_int64(int(steps)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLinspaceGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_linspace(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLinspace failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(out_tensor)
        if start_scalar is not None:
            bindings.acl_destroy_scalar(start_scalar)
        if end_scalar is not None:
            bindings.acl_destroy_scalar(end_scalar)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (out_keep, start_keep, end_keep)


def eye(n, m, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_eye_get_workspace is None or bindings.aclnn_eye is None:
        raise RuntimeError("aclnnEye symbols not available")
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_eye_get_workspace(
            ctypes.c_int64(int(n)),
            ctypes.c_int64(int(m)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEyeGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_eye(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEye failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (out_keep,)


def range_(start, end, step, out_ptr, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_range_get_workspace is None or bindings.aclnn_range is None:
        raise RuntimeError("aclnnRange symbols not available")
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    start_scalar = None
    end_scalar = None
    step_scalar = None
    start_keep = None
    end_keep = None
    step_keep = None
    try:
        start_scalar, start_keep = _create_scalar(bindings, start, dtype)
        end_scalar, end_keep = _create_scalar(bindings, end, dtype)
        step_scalar, step_keep = _create_scalar(bindings, step, dtype)
        ret = bindings.aclnn_range_get_workspace(
            start_scalar,
            end_scalar,
            step_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRangeGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_range(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRange failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(out_tensor)
        if start_scalar is not None:
            bindings.acl_destroy_scalar(start_scalar)
        if end_scalar is not None:
            bindings.acl_destroy_scalar(end_scalar)
        if step_scalar is not None:
            bindings.acl_destroy_scalar(step_scalar)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (out_keep, start_keep, end_keep, step_keep)


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def flip(self_ptr, out_ptr, shape, stride, dtype, dims, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_flip_get_workspace is None or bindings.aclnn_flip is None:
        raise RuntimeError("aclnnFlip symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    dims_arr = _make_int64_array(list(dims))
    dims_handle = bindings.acl_create_int_array(dims_arr, ctypes.c_uint64(len(dims)))
    if not dims_handle:
        raise RuntimeError("aclCreateIntArray returned null")
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_flip_get_workspace(
            self_tensor,
            dims_handle,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnFlipGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_flip(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnFlip failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(dims_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep, dims_arr)


def roll(self_ptr, out_ptr, shape, stride, dtype, shifts, dims, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_roll_get_workspace is None or bindings.aclnn_roll is None:
        raise RuntimeError("aclnnRoll symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    shifts_arr = _make_int64_array(list(shifts))
    dims_arr = _make_int64_array(list(dims))
    shifts_handle = bindings.acl_create_int_array(shifts_arr, ctypes.c_uint64(len(shifts)))
    if not shifts_handle:
        raise RuntimeError("aclCreateIntArray for shifts returned null")
    dims_handle = bindings.acl_create_int_array(dims_arr, ctypes.c_uint64(len(dims)))
    if not dims_handle:
        raise RuntimeError("aclCreateIntArray for dims returned null")
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_roll_get_workspace(
            self_tensor,
            shifts_handle,
            dims_handle,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRollGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_roll(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRoll failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if dims_handle is not None:
            bindings.acl_destroy_int_array(dims_handle)
        if shifts_handle is not None:
            bindings.acl_destroy_int_array(shifts_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep, shifts_arr, dims_arr)


def cumsum(self_ptr, out_ptr, shape, stride, self_dtype, dim, out_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cumsum_get_workspace is None or bindings.aclnn_cumsum is None:
        raise RuntimeError("aclnnCumsum symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, self_dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_cumsum_get_workspace(
            self_tensor,
            ctypes.c_int64(int(dim)),
            ctypes.c_int32(_dtype_to_acl(out_dtype)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCumsumGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_cumsum(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCumsum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def cumprod(self_ptr, out_ptr, shape, stride, self_dtype, dim, out_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cumprod_get_workspace is None or bindings.aclnn_cumprod is None:
        raise RuntimeError("aclnnCumprod symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, self_dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, out_dtype, out_ptr)
    dim_scalar = None
    dim_keep = None
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        dim_scalar, dim_keep = _create_scalar(bindings, int(dim), "int32")
        ret = bindings.aclnn_cumprod_get_workspace(
            self_tensor,
            dim_scalar,
            ctypes.c_int32(_dtype_to_acl(out_dtype)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCumprodGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_cumprod(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCumprod failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if dim_scalar is not None:
            bindings.acl_destroy_scalar(dim_scalar)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, dim_keep)


def cummax(self_ptr, values_ptr, indices_ptr, shape, stride, dtype, dim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cummax_get_workspace is None or bindings.aclnn_cummax is None:
        raise RuntimeError("aclnnCummax symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    values_tensor, values_keep = _create_tensor(bindings, shape, stride, dtype, values_ptr)
    indices_tensor, indices_keep = _create_tensor(bindings, shape, stride, "int64", indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_cummax_get_workspace(
            self_tensor,
            ctypes.c_int64(int(dim)),
            values_tensor,
            indices_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCummaxGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_cummax(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCummax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(values_tensor)
        bindings.acl_destroy_tensor(indices_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, values_keep, indices_keep)


def argsort(self_ptr, out_ptr, shape, stride, dim, descending, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_argsort_get_workspace is None or bindings.aclnn_argsort is None:
        raise RuntimeError("aclnnArgsort symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, "int64", out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_argsort_get_workspace(
            self_tensor,
            ctypes.c_int64(int(dim)),
            ctypes.c_bool(bool(descending)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArgsortGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_argsort(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnArgsort failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def sort(self_ptr, values_ptr, indices_ptr, shape, stride, dim, descending, stable, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_sort_get_workspace is None or bindings.aclnn_sort is None:
        raise RuntimeError("aclnnSort symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    values_tensor, values_keep = _create_tensor(bindings, shape, stride, dtype, values_ptr)
    indices_tensor, indices_keep = _create_tensor(bindings, shape, stride, "int64", indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_sort_get_workspace(
            self_tensor,
            ctypes.c_bool(bool(stable)),
            ctypes.c_int64(int(dim)),
            ctypes.c_bool(bool(descending)),
            values_tensor,
            indices_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSortGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_sort(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSort failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(values_tensor)
        bindings.acl_destroy_tensor(indices_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, values_keep, indices_keep)


def topk(self_ptr, values_ptr, indices_ptr, shape, stride, k, dim, largest, sorted_flag, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_topk_get_workspace is None or bindings.aclnn_topk is None:
        raise RuntimeError("aclnnTopk symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_shape = list(shape)
    out_shape[int(dim)] = int(k)
    out_shape = tuple(out_shape)
    out_stride = _contiguous_stride(out_shape)
    values_tensor, values_keep = _create_tensor(bindings, out_shape, out_stride, dtype, values_ptr)
    indices_tensor, indices_keep = _create_tensor(bindings, out_shape, out_stride, "int64", indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_topk_get_workspace(
            self_tensor,
            ctypes.c_int64(int(k)),
            ctypes.c_int64(int(dim)),
            ctypes.c_bool(bool(largest)),
            ctypes.c_bool(bool(sorted_flag)),
            values_tensor,
            indices_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnTopkGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_topk(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnTopk failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(values_tensor)
        bindings.acl_destroy_tensor(indices_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, values_keep, indices_keep)


def tril(self_ptr, out_ptr, shape, stride, diagonal, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_tril_get_workspace is None or bindings.aclnn_tril is None:
        raise RuntimeError("aclnnTril symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_tril_get_workspace(
            self_tensor,
            ctypes.c_int64(int(diagonal)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnTrilGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_tril(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnTril failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def triu(self_ptr, out_ptr, shape, stride, diagonal, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_triu_get_workspace is None or bindings.aclnn_triu is None:
        raise RuntimeError("aclnnTriu symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_triu_get_workspace(
            self_tensor,
            ctypes.c_int64(int(diagonal)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnTriuGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_triu(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnTriu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def nonzero(self_ptr, out_ptr, shape, stride, dtype, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_nonzero_get_workspace is None or bindings.aclnn_nonzero is None:
        raise RuntimeError("aclnnNonzero symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "int64", out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_nonzero_get_workspace(
            self_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnNonzeroGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_nonzero(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnNonzero failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def repeat(self_ptr, out_ptr, shape, stride, dtype, repeats, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_repeat_get_workspace is None or bindings.aclnn_repeat is None:
        raise RuntimeError("aclnnRepeat symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    repeats_arr = _make_int64_array(repeats)
    repeats_handle = bindings.acl_create_int_array(repeats_arr, ctypes.c_uint64(len(repeats)))
    if not repeats_handle:
        raise RuntimeError("aclCreateIntArray returned null")

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_repeat_get_workspace(
            self_tensor,
            repeats_handle,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRepeatGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_repeat(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRepeat failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_int_array(repeats_handle)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, repeats_arr)


def repeat_interleave_int(
    self_ptr,
    out_ptr,
    shape,
    stride,
    dtype,
    repeats,
    dim,
    output_size,
    out_shape,
    out_stride,
    runtime,
    stream=None,
):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    use_dim = dim is not None
    if use_dim:
        if (
            bindings.aclnn_repeat_interleave_int_with_dim_get_workspace is None
            or bindings.aclnn_repeat_interleave_int_with_dim is None
        ):
            raise RuntimeError("aclnnRepeatInterleaveIntWithDim symbols not available")
    else:
        if bindings.aclnn_repeat_interleave_int_get_workspace is None or bindings.aclnn_repeat_interleave_int is None:
            raise RuntimeError("aclnnRepeatInterleaveInt symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        if use_dim:
            ret = bindings.aclnn_repeat_interleave_int_with_dim_get_workspace(
                self_tensor,
                ctypes.c_int64(int(repeats)),
                ctypes.c_int64(int(dim)),
                ctypes.c_int64(int(output_size)),
                out_tensor,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnRepeatInterleaveIntWithDimGetWorkspaceSize failed: {ret}")
        else:
            ret = bindings.aclnn_repeat_interleave_int_get_workspace(
                self_tensor,
                ctypes.c_int64(int(repeats)),
                ctypes.c_int64(int(output_size)),
                out_tensor,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnRepeatInterleaveIntGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        if use_dim:
            ret = bindings.aclnn_repeat_interleave_int_with_dim(
                ctypes.c_void_p(0 if workspace is None else int(workspace)),
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnRepeatInterleaveIntWithDim failed: {ret}")
        else:
            ret = bindings.aclnn_repeat_interleave_int(
                ctypes.c_void_p(0 if workspace is None else int(workspace)),
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnRepeatInterleaveInt failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def scatter(
    self_ptr,
    index_ptr,
    src_ptr,
    out_ptr,
    self_shape,
    self_stride,
    self_dtype,
    index_shape,
    index_stride,
    index_dtype,
    src_shape,
    src_stride,
    src_dtype,
    dim,
    reduce,
    runtime,
    stream=None,
):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_scatter_get_workspace is None or bindings.aclnn_scatter is None:
        raise RuntimeError("aclnnScatter symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    index_tensor, index_keep = _create_tensor(bindings, index_shape, index_stride, index_dtype, index_ptr)
    src_tensor, src_keep = _create_tensor(bindings, src_shape, src_stride, src_dtype, src_ptr)
    out_tensor, out_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_scatter_get_workspace(
            self_tensor,
            ctypes.c_int64(int(dim)),
            index_tensor,
            src_tensor,
            ctypes.c_int64(int(reduce)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnScatterGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_scatter(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnScatter failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        bindings.acl_destroy_tensor(src_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, index_keep, src_keep, out_keep)


def diag(self_ptr, out_ptr, shape, stride, dtype, diagonal, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_diag_get_workspace is None or bindings.aclnn_diag is None:
        raise RuntimeError("aclnnDiag symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_diag_get_workspace(
            self_tensor,
            ctypes.c_int64(int(diagonal)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDiagGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_diag(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDiag failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def index_put_impl(self_ptr, self_shape, self_stride, self_dtype,
                   index_ptrs, index_shapes, index_strides, index_dtypes,
                   values_ptr, values_shape, values_stride, values_dtype,
                   accumulate, unsafe, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_index_put_impl_get_workspace is None or bindings.aclnn_index_put_impl is None:
        raise RuntimeError("aclnnIndexPutImpl symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    values_tensor, values_keep = _create_tensor(bindings, values_shape, values_stride, values_dtype, values_ptr)
    tensor_list = None
    tensor_keeps = []
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        tensor_list, tensor_keeps = _create_tensor_list(
            bindings,
            index_ptrs,
            index_shapes,
            index_strides,
            index_dtypes,
        )
        ret = bindings.aclnn_index_put_impl_get_workspace(
            self_tensor,
            tensor_list,
            values_tensor,
            ctypes.c_bool(bool(accumulate)),
            ctypes.c_bool(bool(unsafe)),
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnIndexPutImplGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_index_put_impl(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnIndexPutImpl failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if tensor_list is not None and bindings.acl_destroy_tensor_list:
            bindings.acl_destroy_tensor_list(tensor_list)
        else:
            for tensor, _ in tensor_keeps:
                bindings.acl_destroy_tensor(tensor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(values_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, values_keep)


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
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "bool", out_ptr)
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
            runtime.defer_raw_free(workspace)
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
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "bool", out_ptr)
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
            runtime.defer_raw_free(workspace)
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
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "bool", out_ptr)
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
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)



def swhere(cond_ptr, self_ptr, other_ptr, out_ptr, cond_shape, cond_stride, self_shape, self_stride,
          other_shape, other_stride, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_swhere_get_workspace is None or bindings.aclnn_swhere is None:
        raise RuntimeError("aclnnSWhere symbols not available")
    cond_tensor, cond_keep = _create_tensor(bindings, cond_shape, cond_stride, "bool", cond_ptr)
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_swhere_get_workspace(
            cond_tensor,
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSWhereGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_swhere(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSWhere failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(cond_tensor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (cond_keep, self_keep, other_keep, out_keep)


def logical_not(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnLogicalNot", bindings.aclnn_logical_not_get_workspace,
                       bindings.aclnn_logical_not, self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


# Bitwise operations
def bitwise_not(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_bitwise_not_get_workspace is None or bindings.aclnn_bitwise_not is None:
        raise RuntimeError("aclnnBitwiseNot symbols not available")
    return _unary_call(bindings, "aclnnBitwiseNot", bindings.aclnn_bitwise_not_get_workspace,
                       bindings.aclnn_bitwise_not, self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def bitwise_and(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_bitwise_and_tensor_get_workspace is None or bindings.aclnn_bitwise_and_tensor is None:
        raise RuntimeError("aclnnBitwiseAndTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_bitwise_and_tensor_get_workspace(
            self_tensor, other_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBitwiseAndTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_bitwise_and_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBitwiseAndTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def bitwise_or(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
               out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_bitwise_or_tensor_get_workspace is None or bindings.aclnn_bitwise_or_tensor is None:
        raise RuntimeError("aclnnBitwiseOrTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_bitwise_or_tensor_get_workspace(
            self_tensor, other_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBitwiseOrTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_bitwise_or_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBitwiseOrTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


def bitwise_xor(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
                out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_bitwise_xor_tensor_get_workspace is None or bindings.aclnn_bitwise_xor_tensor is None:
        raise RuntimeError("aclnnBitwiseXorTensor symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_bitwise_xor_tensor_get_workspace(
            self_tensor, other_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBitwiseXorTensorGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_bitwise_xor_tensor(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBitwiseXorTensor failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "bool", out_ptr)
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
            runtime.defer_raw_free(workspace)
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
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, "bool", out_ptr)
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
            runtime.defer_raw_free(workspace)
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


def expm1(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_expm1_get_workspace is None or bindings.aclnn_expm1 is None:
        raise RuntimeError("aclnnExpm1 symbols not available")
    return _unary_call(bindings, "aclnnExpm1", bindings.aclnn_expm1_get_workspace, bindings.aclnn_expm1,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def log1p(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_log1p_get_workspace is None or bindings.aclnn_log1p is None:
        raise RuntimeError("aclnnLog1p symbols not available")
    return _unary_call(bindings, "aclnnLog1p", bindings.aclnn_log1p_get_workspace, bindings.aclnn_log1p,
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
            runtime.defer_raw_free(workspace)
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
            runtime.defer_raw_free(workspace)
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
                ctypes.c_int8(1),
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
        else:
            ret = bindings.aclnn_matmul_get_workspace(
                a_tensor,
                b_tensor,
                out_tensor,
                ctypes.c_int8(1),
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
            runtime.defer_raw_free(workspace)
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


def add_scalar_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_add_scalar_get_workspace, bindings.aclnn_add_scalar])
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


def bitwise_not_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_not_get_workspace, bindings.aclnn_bitwise_not])
    except Exception:
        return False


def bitwise_and_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_and_tensor_get_workspace, bindings.aclnn_bitwise_and_tensor])
    except Exception:
        return False


def bitwise_or_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_or_tensor_get_workspace, bindings.aclnn_bitwise_or_tensor])
    except Exception:
        return False


def bitwise_xor_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_bitwise_xor_tensor_get_workspace, bindings.aclnn_bitwise_xor_tensor])
    except Exception:
        return False


def expm1_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_expm1_get_workspace, bindings.aclnn_expm1])
    except Exception:
        return False


def log1p_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_log1p_get_workspace, bindings.aclnn_log1p])
    except Exception:
        return False


def dot_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_dot_get_workspace, bindings.aclnn_dot])
    except Exception:
        return False


def mv_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_mv_get_workspace, bindings.aclnn_mv])
    except Exception:
        return False


def ger_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_ger_get_workspace, bindings.aclnn_ger])
    except Exception:
        return False


def median_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_median_get_workspace, bindings.aclnn_median])
    except Exception:
        return False


def median_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_median_dim_get_workspace, bindings.aclnn_median_dim])
    except Exception:
        return False


def kthvalue_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_kthvalue_get_workspace, bindings.aclnn_kthvalue])
    except Exception:
        return False


def search_sorted_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_search_sorted_get_workspace, bindings.aclnn_search_sorted])
    except Exception:
        return False


def unique_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_unique_get_workspace, bindings.aclnn_unique])
    except Exception:
        return False


def randperm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_randperm_get_workspace, bindings.aclnn_randperm])
    except Exception:
        return False


def flatten_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_flatten_get_workspace, bindings.aclnn_flatten])
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


def arange_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_arange_get_workspace, bindings.aclnn_arange])
    except Exception:
        return False


def linspace_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_linspace_get_workspace, bindings.aclnn_linspace])
    except Exception:
        return False


def eye_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_eye_get_workspace, bindings.aclnn_eye])
    except Exception:
        return False


def range_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_range_get_workspace, bindings.aclnn_range])
    except Exception:
        return False


def flip_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_flip_get_workspace, bindings.aclnn_flip])
    except Exception:
        return False


def roll_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_roll_get_workspace, bindings.aclnn_roll])
    except Exception:
        return False


def cumsum_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cumsum_get_workspace, bindings.aclnn_cumsum])
    except Exception:
        return False


def cumprod_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cumprod_get_workspace, bindings.aclnn_cumprod])
    except Exception:
        return False


def cummax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_cummax_get_workspace, bindings.aclnn_cummax])
    except Exception:
        return False


def argsort_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_argsort_get_workspace, bindings.aclnn_argsort])
    except Exception:
        return False


def sort_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_sort_get_workspace, bindings.aclnn_sort])
    except Exception:
        return False


def topk_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_topk_get_workspace, bindings.aclnn_topk])
    except Exception:
        return False


def tril_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_tril_get_workspace, bindings.aclnn_tril])
    except Exception:
        return False


def triu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_triu_get_workspace, bindings.aclnn_triu])
    except Exception:
        return False


def nonzero_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_nonzero_get_workspace, bindings.aclnn_nonzero])
    except Exception:
        return False


def repeat_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_repeat_get_workspace, bindings.aclnn_repeat])
    except Exception:
        return False


def repeat_interleave_int_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_repeat_interleave_int_get_workspace,
            bindings.aclnn_repeat_interleave_int,
            bindings.aclnn_repeat_interleave_int_with_dim_get_workspace,
            bindings.aclnn_repeat_interleave_int_with_dim,
        ])
    except Exception:
        return False


def scatter_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_scatter_get_workspace, bindings.aclnn_scatter])
    except Exception:
        return False


def diag_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_diag_get_workspace, bindings.aclnn_diag])
    except Exception:
        return False


def index_put_impl_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_index_put_impl_get_workspace,
            bindings.aclnn_index_put_impl,
        ])
    except Exception:
        return False


def _create_tensor_list(bindings, tensor_ptrs, shapes, strides, dtypes):
    """Create aclTensorList from multiple tensors."""
    if bindings.acl_create_tensor_list is None:
        raise RuntimeError("aclCreateTensorList not available")

    num_tensors = len(tensor_ptrs)
    tensor_array = (ctypes.c_void_p * num_tensors)()
    tensor_keeps = []

    for i in range(num_tensors):
        tensor, keep = _create_tensor(bindings, shapes[i], strides[i], dtypes[i], tensor_ptrs[i])
        tensor_array[i] = tensor
        tensor_keeps.append((tensor, keep))

    tensor_list = bindings.acl_create_tensor_list(tensor_array, ctypes.c_uint64(num_tensors))
    if not tensor_list:
        raise RuntimeError("aclCreateTensorList failed")

    return tensor_list, tensor_keeps


def _create_tensor_list_with_nones(bindings, entries):
    """Create aclTensorList where entries may be None (null pointer in the list).

    *entries* is a list of either ``None`` or a tuple
    ``(data_ptr, shape, stride, dtype)`` for each dimension.
    """
    if bindings.acl_create_tensor_list is None:
        raise RuntimeError("aclCreateTensorList not available")

    num = len(entries)
    tensor_array = (ctypes.c_void_p * num)()
    tensor_keeps = []

    for i, entry in enumerate(entries):
        if entry is None:
            tensor_array[i] = ctypes.c_void_p(0)
            tensor_keeps.append(None)
        else:
            data_ptr, shape, stride, dtype = entry
            tensor, keep = _create_tensor(bindings, shape, stride, dtype, data_ptr)
            tensor_array[i] = tensor
            tensor_keeps.append((tensor, keep))

    tensor_list = bindings.acl_create_tensor_list(tensor_array, ctypes.c_uint64(num))
    if not tensor_list:
        raise RuntimeError("aclCreateTensorList failed")

    return tensor_list, tensor_keeps


def index_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_index_get_workspace,
            bindings.aclnn_index,
        ])
    except Exception:
        return False


def slice_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_slice_get_workspace,
            bindings.aclnn_slice,
        ])
    except Exception:
        return False


def index(self_ptr, self_shape, self_stride, self_dtype,
          index_entries, out_ptr, out_shape, out_stride, out_dtype,
          runtime, stream=None):
    """aclnnIndex — advanced indexing getitem.

    *index_entries* is a list (length == ndim of self) where each element is
    either ``None`` (dimension not indexed) or a tuple
    ``(data_ptr, shape, stride, dtype)`` for an index tensor.
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_index_get_workspace is None or bindings.aclnn_index is None:
        raise RuntimeError("aclnnIndex symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    tensor_list = None
    tensor_keeps = []
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        tensor_list, tensor_keeps = _create_tensor_list_with_nones(bindings, index_entries)

        ret = bindings.aclnn_index_get_workspace(
            self_tensor,
            tensor_list,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnIndexGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_index(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnIndex failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if tensor_list is not None and bindings.acl_destroy_tensor_list:
            bindings.acl_destroy_tensor_list(tensor_list)
        else:
            for item in tensor_keeps:
                if item is not None:
                    bindings.acl_destroy_tensor(item[0])
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def slice_op(self_ptr, self_shape, self_stride, self_dtype,
             dim, start, end, step,
             out_ptr, out_shape, out_stride, out_dtype,
             runtime, stream=None):
    """aclnnSlice — strided slicing on a single dimension."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_slice_get_workspace is None or bindings.aclnn_slice is None:
        raise RuntimeError("aclnnSlice symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_slice_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            ctypes.c_int64(start),
            ctypes.c_int64(end),
            ctypes.c_int64(step),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSliceGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_slice(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSlice failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


def cat(tensor_ptrs, shapes, strides, dtypes, dim, out_ptr, out_shape, out_stride, out_dtype, runtime, stream=None):
    """Concatenate tensors along an existing dimension using aclnnCat."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_cat_get_workspace is None or bindings.aclnn_cat is None:
        raise RuntimeError("aclnnCat symbols not available")

    tensor_list, tensor_keeps = _create_tensor_list(bindings, tensor_ptrs, shapes, strides, dtypes)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_cat_get_workspace(
            tensor_list,
            ctypes.c_int64(dim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCatGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_cat(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnCat failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if tensor_list is not None and bindings.acl_destroy_tensor_list:
            bindings.acl_destroy_tensor_list(tensor_list)
        else:
            for tensor, _ in tensor_keeps:
                bindings.acl_destroy_tensor(tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def stack(tensor_ptrs, shapes, strides, dtypes, dim, out_ptr, out_shape, out_stride, out_dtype, runtime, stream=None):
    """Stack tensors along a new dimension using aclnnStack."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_stack_get_workspace is None or bindings.aclnn_stack is None:
        raise RuntimeError("aclnnStack symbols not available")

    tensor_list, tensor_keeps = _create_tensor_list(bindings, tensor_ptrs, shapes, strides, dtypes)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_stack_get_workspace(
            tensor_list,
            ctypes.c_int64(dim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnStackGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_stack(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnStack failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        if tensor_list is not None and bindings.acl_destroy_tensor_list:
            bindings.acl_destroy_tensor_list(tensor_list)
        else:
            for tensor, _ in tensor_keeps:
                bindings.acl_destroy_tensor(tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def s_where(condition_ptr, self_ptr, other_ptr, out_ptr,
            condition_shape, condition_stride, condition_dtype,
            self_shape, self_stride, self_dtype,
            other_shape, other_stride, other_dtype,
            out_shape, out_stride, out_dtype,
            runtime, stream=None):
    """Element-wise where using aclnnSWhere."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_s_where_get_workspace is None or bindings.aclnn_s_where is None:
        raise RuntimeError("aclnnSWhere symbols not available")

    condition_tensor, condition_keep = _create_tensor(bindings, condition_shape, condition_stride, condition_dtype, condition_ptr)
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, other_dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_s_where_get_workspace(
            condition_tensor,
            self_tensor,
            other_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSWhereGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_s_where(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSWhere failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(condition_tensor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (condition_keep, self_keep, other_keep, out_keep)


def cat_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.acl_create_tensor_list,
            bindings.acl_destroy_tensor_list,
            bindings.aclnn_cat_get_workspace,
            bindings.aclnn_cat
        ])
    except Exception:
        return False


def stack_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.acl_create_tensor_list,
            bindings.acl_destroy_tensor_list,
            bindings.aclnn_stack_get_workspace,
            bindings.aclnn_stack
        ])
    except Exception:
        return False


def s_where_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_s_where_get_workspace, bindings.aclnn_s_where])
    except Exception:
        return False


def mean(self_ptr, out_ptr, shape, stride, dtype, dims, keepdim, out_shape, out_stride, runtime, stream=None):
    """Compute mean along dimensions using aclnnMean."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_mean_get_workspace is None or bindings.aclnn_mean is None:
        raise RuntimeError("aclnnMean symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)

    # Create IntArray for dims
    if dims:
        dim_array = _make_int64_array(dims)
        dim_handle = bindings.acl_create_int_array(dim_array, ctypes.c_uint64(len(dims)))
    else:
        dim_array = (ctypes.c_int64 * 0)()
        dim_handle = bindings.acl_create_int_array(dim_array, ctypes.c_uint64(0))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_mean_get_workspace(
            self_tensor,
            dim_handle,
            ctypes.c_bool(keepdim),
            ctypes.c_int32(_dtype_to_acl(dtype)),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMeanGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_mean(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMean failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(dim_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def softmax(self_ptr, out_ptr, shape, stride, dtype, dim, runtime, stream=None):
    """Compute softmax along a dimension using aclnnSoftmax."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_softmax_get_workspace is None or bindings.aclnn_softmax is None:
        raise RuntimeError("aclnnSoftmax symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_softmax_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSoftmaxGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_softmax(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSoftmax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def log_softmax(self_ptr, out_ptr, shape, stride, dtype, dim, runtime, stream=None):
    """Compute log_softmax along a dimension using aclnnLogSoftmax."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_log_softmax_get_workspace is None or bindings.aclnn_log_softmax is None:
        raise RuntimeError("aclnnLogSoftmax symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_log_softmax_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogSoftmaxGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_log_softmax(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLogSoftmax failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def gelu(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    """Compute GELU activation using aclnnGelu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_gelu_get_workspace is None or bindings.aclnn_gelu is None:
        raise RuntimeError("aclnnGelu symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_gelu_get_workspace(
            self_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGeluGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_gelu(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGelu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def layer_norm(input_ptr, weight_ptr, bias_ptr, out_ptr, mean_ptr, rstd_ptr,
               input_shape, input_stride, weight_shape, weight_stride,
               bias_shape, bias_stride, out_shape, out_stride,
               stats_shape, stats_stride, normalized_shape, eps, dtype, runtime, stream=None):
    """Compute layer normalization using aclnnLayerNorm."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_layer_norm_get_workspace is None or bindings.aclnn_layer_norm is None:
        raise RuntimeError("aclnnLayerNorm symbols not available")

    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)

    # Mean and rstd outputs are optional for this integration path.
    if mean_ptr is not None:
        mean_tensor, mean_keep = _create_tensor(bindings, stats_shape, stats_stride, "float32", mean_ptr)
    else:
        mean_tensor = None
        mean_keep = None

    if rstd_ptr is not None:
        rstd_tensor, rstd_keep = _create_tensor(bindings, stats_shape, stats_stride, "float32", rstd_ptr)
    else:
        rstd_tensor = None
        rstd_keep = None

    # Weight and bias are optional
    if weight_ptr is not None:
        weight_tensor, weight_keep = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr)
    else:
        weight_tensor = None
        weight_keep = None

    if bias_ptr is not None:
        bias_tensor, bias_keep = _create_tensor(bindings, bias_shape, bias_stride, dtype, bias_ptr)
    else:
        bias_tensor = None
        bias_keep = None

    # Create IntArray for normalized_shape
    norm_shape_array = _make_int64_array(normalized_shape)
    norm_shape_handle = bindings.acl_create_int_array(norm_shape_array, ctypes.c_uint64(len(normalized_shape)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_layer_norm_get_workspace(
            input_tensor,
            norm_shape_handle,
            ctypes.c_void_p(0) if weight_tensor is None else weight_tensor,
            ctypes.c_void_p(0) if bias_tensor is None else bias_tensor,
            ctypes.c_double(eps),
            out_tensor,
            ctypes.c_void_p(0) if mean_tensor is None else mean_tensor,
            ctypes.c_void_p(0) if rstd_tensor is None else rstd_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLayerNormGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_layer_norm(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLayerNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(norm_shape_handle)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if mean_tensor is not None:
            bindings.acl_destroy_tensor(mean_tensor)
        if rstd_tensor is not None:
            bindings.acl_destroy_tensor(rstd_tensor)
        if weight_tensor is not None:
            bindings.acl_destroy_tensor(weight_tensor)
        if bias_tensor is not None:
            bindings.acl_destroy_tensor(bias_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def embedding(weight_ptr, indices_ptr, out_ptr, weight_shape, weight_stride,
              indices_shape, indices_stride, out_shape, out_stride,
              weight_dtype, indices_dtype, runtime, stream=None):
    """Compute embedding lookup using aclnnEmbedding."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_embedding_get_workspace is None or bindings.aclnn_embedding is None:
        raise RuntimeError("aclnnEmbedding symbols not available")

    weight_tensor, weight_keep = _create_tensor(bindings, weight_shape, weight_stride, weight_dtype, weight_ptr)
    indices_tensor, indices_keep = _create_tensor(bindings, indices_shape, indices_stride, indices_dtype, indices_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, weight_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_embedding_get_workspace(
            weight_tensor,
            indices_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEmbeddingGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_embedding(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEmbedding failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(weight_tensor)
        bindings.acl_destroy_tensor(indices_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)




def gather(self_ptr, index_ptr, out_ptr,
           self_shape, self_stride, self_dtype,
           index_shape, index_stride, index_dtype,
           out_shape, out_stride, out_dtype,
           dim, runtime, stream=None):
    """Gather elements along dim using "aclnnGather"."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_gather_get_workspace is None or bindings.aclnn_gather is None:
        raise RuntimeError("aclnnGather symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    index_tensor, index_keep = _create_tensor(bindings, index_shape, index_stride, index_dtype, index_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_gather_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            index_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGatherGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_gather(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGather failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def masked_select(self_ptr, mask_ptr, out_ptr,
                  self_shape, self_stride, self_dtype,
                  mask_shape, mask_stride, mask_dtype,
                  out_shape, out_stride, out_dtype,
                  runtime, stream=None):
    """Masked select using "aclnnMaskedSelect"."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_masked_select_get_workspace is None or bindings.aclnn_masked_select is None:
        raise RuntimeError("aclnnMaskedSelect symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    mask_tensor, mask_keep = _create_tensor(bindings, mask_shape, mask_stride, mask_dtype, mask_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_masked_select_get_workspace(
            self_tensor,
            mask_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaskedSelectGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_masked_select(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaskedSelect failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(mask_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def constant_pad_nd(self_ptr, out_ptr,
                    self_shape, self_stride, self_dtype,
                    pad_widths, value,
                    out_shape, out_stride, out_dtype,
                    runtime, stream=None):
    """Pad tensor with constant value using aclnnConstantPadNd."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_constant_pad_nd_get_workspace is None or bindings.aclnn_constant_pad_nd is None:
        raise RuntimeError("aclnnConstantPadNd symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)

    pad_arr = _make_int64_array(tuple(int(x) for x in pad_widths))
    pad_handle = bindings.acl_create_int_array(pad_arr, ctypes.c_uint64(len(pad_widths)))
    if not pad_handle:
        raise RuntimeError("aclCreateIntArray returned null")

    value_scalar, value_keep = _create_scalar(bindings, value, self_dtype)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_constant_pad_nd_get_workspace(
            self_tensor,
            pad_handle,
            value_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnConstantPadNdGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_constant_pad_nd(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnConstantPadNd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_int_array(pad_handle)
        bindings.acl_destroy_scalar(value_scalar)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, pad_arr, value_keep)


def gather_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_gather_get_workspace, bindings.aclnn_gather])
    except Exception:
        return False


def masked_select_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_masked_select_get_workspace, bindings.aclnn_masked_select])
    except Exception:
        return False

def mean_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_mean_get_workspace, bindings.aclnn_mean])
    except Exception:
        return False


def softmax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_softmax_get_workspace, bindings.aclnn_softmax])
    except Exception:
        return False


def log_softmax_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_log_softmax_get_workspace, bindings.aclnn_log_softmax])
    except Exception:
        return False


def gelu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_gelu_get_workspace, bindings.aclnn_gelu])
    except Exception:
        return False


def layer_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_layer_norm_get_workspace, bindings.aclnn_layer_norm])
    except Exception:
        return False


def embedding_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_embedding_get_workspace, bindings.aclnn_embedding])
    except Exception:
        return False


def silu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_silu_get_workspace, bindings.aclnn_silu])
    except Exception:
        return False


def leaky_relu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_leaky_relu_get_workspace, bindings.aclnn_leaky_relu])
    except Exception:
        return False


def elu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_elu_get_workspace, bindings.aclnn_elu])
    except Exception:
        return False


def mish_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_mish_get_workspace, bindings.aclnn_mish])
    except Exception:
        return False


def prelu_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_prelu_get_workspace, bindings.aclnn_prelu])
    except Exception:
        return False


def batch_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_batch_norm_get_workspace, bindings.aclnn_batch_norm])
    except Exception:
        return False


def group_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_group_norm_get_workspace, bindings.aclnn_group_norm])
    except Exception:
        return False


def gather_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_gather_get_workspace, bindings.aclnn_gather])
    except Exception:
        return False


def constant_pad_nd_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_constant_pad_nd_get_workspace, bindings.aclnn_constant_pad_nd])
    except Exception:
        return False


def silu(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    """Compute SiLU activation using aclnnSilu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_silu_get_workspace is None or bindings.aclnn_silu is None:
        raise RuntimeError("aclnnSilu symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_silu_get_workspace(
            self_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSiluGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_silu(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSilu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def leaky_relu(self_ptr, out_ptr, shape, stride, dtype, negative_slope, runtime, stream=None):
    """Compute LeakyReLU activation using aclnnLeakyRelu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_leaky_relu_get_workspace is None or bindings.aclnn_leaky_relu is None:
        raise RuntimeError("aclnnLeakyRelu symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)

    # Create scalar for negative_slope
    slope_bytes = _scalar_bytes(negative_slope, dtype)
    slope_scalar = bindings.acl_create_scalar(slope_bytes, _dtype_to_acl(dtype))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_leaky_relu_get_workspace(
            self_tensor,
            slope_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLeakyReluGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_leaky_relu(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLeakyRelu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_scalar(slope_scalar)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def elu(self_ptr, out_ptr, shape, stride, dtype, alpha, runtime, stream=None):
    """Compute ELU activation using aclnnElu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_elu_get_workspace is None or bindings.aclnn_elu is None:
        raise RuntimeError("aclnnElu symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)

    # Create scalars for alpha, scale, input_scale (scale and input_scale are typically 1.0)
    alpha_bytes = _scalar_bytes(alpha, dtype)
    alpha_scalar = bindings.acl_create_scalar(alpha_bytes, _dtype_to_acl(dtype))
    scale_bytes = _scalar_bytes(1.0, dtype)
    scale_scalar = bindings.acl_create_scalar(scale_bytes, _dtype_to_acl(dtype))
    input_scale_bytes = _scalar_bytes(1.0, dtype)
    input_scale_scalar = bindings.acl_create_scalar(input_scale_bytes, _dtype_to_acl(dtype))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_elu_get_workspace(
            self_tensor,
            alpha_scalar,
            scale_scalar,
            input_scale_scalar,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEluGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_elu(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnElu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_scalar(alpha_scalar)
        bindings.acl_destroy_scalar(scale_scalar)
        bindings.acl_destroy_scalar(input_scale_scalar)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def mish(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    """Compute Mish activation using aclnnMish."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_mish_get_workspace is None or bindings.aclnn_mish is None:
        raise RuntimeError("aclnnMish symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_mish_get_workspace(
            self_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMishGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_mish(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMish failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def prelu(self_ptr, weight_ptr, out_ptr, shape, stride, weight_shape, weight_stride, dtype, runtime, stream=None):
    """Compute PReLU activation using aclnnPrelu."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_prelu_get_workspace is None or bindings.aclnn_prelu is None:
        raise RuntimeError("aclnnPrelu symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    weight_tensor, weight_keep = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_prelu_get_workspace(
            self_tensor,
            weight_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnPreluGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_prelu(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnPrelu failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(weight_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def batch_norm(input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr, out_ptr,
               input_shape, input_stride, weight_shape, weight_stride, bias_shape, bias_stride,
               running_mean_shape, running_mean_stride, running_var_shape, running_var_stride,
               out_shape, out_stride, training, momentum, eps, dtype, runtime, stream=None,
               ext_save_mean_ptr=None, ext_save_invstd_ptr=None):
    """Compute batch normalization using aclnnBatchNorm."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_batch_norm_get_workspace is None or bindings.aclnn_batch_norm is None:
        raise RuntimeError("aclnnBatchNorm symbols not available")

    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)

    # Optional tensors
    weight_tensor = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr)[0] if weight_ptr else None
    bias_tensor = _create_tensor(bindings, bias_shape, bias_stride, dtype, bias_ptr)[0] if bias_ptr else None
    running_mean_tensor = _create_tensor(bindings, running_mean_shape, running_mean_stride, dtype, running_mean_ptr)[0] if running_mean_ptr else None
    running_var_tensor = _create_tensor(bindings, running_var_shape, running_var_stride, dtype, running_var_ptr)[0] if running_var_ptr else None

    # Allocate auxiliary output tensors (saveMean, saveInvstd) required by ACLNN.
    # ACLNN does not accept NULL for these even if we don't need the values.
    # Shape is (C,) where C is the number of channels; dtype is always float32.
    # IMPORTANT: Must use allocator (not acl.rt.malloc) to avoid memory corruption.
    C = input_shape[1] if len(input_shape) >= 2 else 1
    aux_shape = (C,)
    aux_stride = (1,)
    aux_dtype = "float32"
    aux_itemsize = 4  # float32

    # Import here to avoid circular dependency
    from . import runtime as npu_runtime_module
    _own_save_ptrs = ext_save_mean_ptr is None
    if _own_save_ptrs:
        save_mean_ptr = npu_runtime_module._alloc_device(C * aux_itemsize, runtime=runtime)
        save_invstd_ptr = npu_runtime_module._alloc_device(C * aux_itemsize, runtime=runtime)
    else:
        save_mean_ptr = ext_save_mean_ptr
        save_invstd_ptr = ext_save_invstd_ptr

    save_mean_tensor, save_mean_keep = _create_tensor(bindings, aux_shape, aux_stride, aux_dtype, save_mean_ptr)
    save_invstd_tensor, save_invstd_keep = _create_tensor(bindings, aux_shape, aux_stride, aux_dtype, save_invstd_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_batch_norm_get_workspace(
            input_tensor,
            weight_tensor,
            bias_tensor,
            running_mean_tensor,
            running_var_tensor,
            ctypes.c_bool(training),
            ctypes.c_double(momentum),
            ctypes.c_double(eps),
            out_tensor,
            save_mean_tensor,
            save_invstd_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBatchNormGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_batch_norm(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBatchNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if weight_tensor:
            bindings.acl_destroy_tensor(weight_tensor)
        if bias_tensor:
            bindings.acl_destroy_tensor(bias_tensor)
        if running_mean_tensor:
            bindings.acl_destroy_tensor(running_mean_tensor)
        if running_var_tensor:
            bindings.acl_destroy_tensor(running_var_tensor)
        if save_mean_tensor:
            bindings.acl_destroy_tensor(save_mean_tensor)
        if save_invstd_tensor:
            bindings.acl_destroy_tensor(save_invstd_tensor)
        if save_mean_ptr is not None and _own_save_ptrs:
            runtime.defer_free(save_mean_ptr)
        if save_invstd_ptr is not None and _own_save_ptrs:
            runtime.defer_free(save_invstd_ptr)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def group_norm(input_ptr, weight_ptr, bias_ptr, out_ptr,
               input_shape, input_stride, weight_shape, weight_stride, bias_shape, bias_stride,
               out_shape, out_stride, num_groups, eps, dtype, runtime, stream=None):
    """Compute group normalization using aclnnGroupNorm."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_group_norm_get_workspace is None or bindings.aclnn_group_norm is None:
        raise RuntimeError("aclnnGroupNorm symbols not available")

    # Extract N, C, HxW from input shape
    N = input_shape[0]
    C = input_shape[1]
    HxW = 1
    for dim in input_shape[2:]:
        HxW *= dim

    # Use NCHW format for input/output tensors (required on Ascend 910B for norm ops)
    input_fmt = _ACL_FORMAT_NCHW if len(input_shape) >= 4 else _ACL_FORMAT_ND
    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, input_fmt)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, input_fmt)

    # Optional tensors
    weight_tensor = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr)[0] if weight_ptr else None
    bias_tensor = _create_tensor(bindings, bias_shape, bias_stride, dtype, bias_ptr)[0] if bias_ptr else None

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    # Allocate auxiliary output tensors (meanOut, rstdOut) required by ACLNN.
    # ACLNN does not accept NULL for these even if we don't need the values.
    # Shape is (N, num_groups); dtype is always float32.
    # IMPORTANT: Must use allocator (not acl.rt.malloc) to avoid memory corruption.
    aux_shape = (N, num_groups)
    aux_stride = (num_groups, 1)
    aux_dtype = "float32"
    aux_itemsize = 4  # float32
    aux_numel = N * num_groups

    from . import runtime as npu_runtime_module
    mean_out_ptr = npu_runtime_module._alloc_device(aux_numel * aux_itemsize, runtime=runtime)
    rstd_out_ptr = npu_runtime_module._alloc_device(aux_numel * aux_itemsize, runtime=runtime)

    mean_out_tensor, mean_out_keep = _create_tensor(bindings, aux_shape, aux_stride, aux_dtype, mean_out_ptr)
    rstd_out_tensor, rstd_out_keep = _create_tensor(bindings, aux_shape, aux_stride, aux_dtype, rstd_out_ptr)

    try:
        ret = bindings.aclnn_group_norm_get_workspace(
            input_tensor,
            weight_tensor,
            bias_tensor,
            ctypes.c_int64(N),
            ctypes.c_int64(C),
            ctypes.c_int64(HxW),
            ctypes.c_int64(num_groups),
            ctypes.c_double(eps),
            out_tensor,
            mean_out_tensor,
            rstd_out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGroupNormGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_group_norm(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGroupNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if weight_tensor:
            bindings.acl_destroy_tensor(weight_tensor)
        if bias_tensor:
            bindings.acl_destroy_tensor(bias_tensor)
        bindings.acl_destroy_tensor(mean_out_tensor)
        bindings.acl_destroy_tensor(rstd_out_tensor)
        runtime.defer_free(mean_out_ptr)
        runtime.defer_free(rstd_out_ptr)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def dropout_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_dropout_gen_mask_get_workspace,
            bindings.aclnn_dropout_gen_mask,
            bindings.aclnn_dropout_do_mask_get_workspace,
            bindings.aclnn_dropout_do_mask,
        ])
    except Exception:
        return False


def inplace_normal_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_inplace_normal_get_workspace, bindings.aclnn_inplace_normal])
    except Exception:
        return False


def _align_up(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def dropout_gen_mask(shape, p, seed, offset, mask_ptr, mask_numel, runtime, stream=None):
    """Generate dropout mask using aclnnDropoutGenMask.

    The mask is bit-packed: output shape is (align(numel, 128) / 8,) with dtype uint8.
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_dropout_gen_mask_get_workspace is None:
        raise RuntimeError("aclnnDropoutGenMask symbols not available")

    # Create shape IntArray using aclCreateIntArray
    shape_data = _make_int64_array(shape)
    shape_arr = bindings.acl_create_int_array(shape_data, ctypes.c_uint64(len(shape)))

    # Create output tensor: bit-packed mask
    mask_shape = (mask_numel,)
    mask_stride = (1,)
    mask_tensor, mask_keep = _create_tensor(bindings, mask_shape, mask_stride, "uint8", mask_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_dropout_gen_mask_get_workspace(
            shape_arr,
            ctypes.c_double(p),
            ctypes.c_int64(seed),
            ctypes.c_int64(offset),
            mask_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDropoutGenMaskGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_dropout_gen_mask(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDropoutGenMask failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(mask_tensor)
        if shape_arr:
            bindings.acl_destroy_int_array(shape_arr)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def dropout_do_mask(input_ptr, mask_ptr, out_ptr, shape, stride, dtype, mask_numel, p, runtime, stream=None):
    """Apply dropout mask using aclnnDropoutDoMask."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_dropout_do_mask_get_workspace is None:
        raise RuntimeError("aclnnDropoutDoMask symbols not available")

    input_tensor, input_keep = _create_tensor(bindings, shape, stride, dtype, input_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)

    # mask is bit-packed uint8
    mask_shape = (mask_numel,)
    mask_stride = (1,)
    mask_tensor, mask_keep = _create_tensor(bindings, mask_shape, mask_stride, "uint8", mask_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_dropout_do_mask_get_workspace(
            input_tensor,
            mask_tensor,
            ctypes.c_double(p),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDropoutDoMaskGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_dropout_do_mask(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDropoutDoMask failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_tensor(mask_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def inplace_normal(self_ptr, shape, stride, dtype, mean, std, seed, offset, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_normal_get_workspace is None or bindings.aclnn_inplace_normal is None:
        raise RuntimeError("aclnnInplaceNormal symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_inplace_normal_get_workspace(
            self_tensor,
            ctypes.c_float(mean),
            ctypes.c_float(std),
            ctypes.c_int64(seed),
            ctypes.c_int64(offset),
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceNormalGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_inplace_normal(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceNormal failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def inplace_uniform(self_ptr, shape, stride, dtype, low, high, seed, offset, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_uniform_get_workspace is None or bindings.aclnn_inplace_uniform is None:
        raise RuntimeError("aclnnInplaceUniform symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_inplace_uniform_get_workspace(
            self_tensor,
            ctypes.c_double(low),
            ctypes.c_double(high),
            ctypes.c_int64(seed),
            ctypes.c_int64(offset),
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceUniformGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_inplace_uniform(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceUniform failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)


def inplace_fill_scalar(self_ptr, shape, stride, dtype, value, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_fill_scalar_get_workspace is None or bindings.aclnn_inplace_fill_scalar is None:
        raise RuntimeError("aclnnInplaceFillScalar symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    value_scalar, value_keep = _create_scalar(bindings, value, dtype)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_inplace_fill_scalar_get_workspace(
            self_tensor,
            value_scalar,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceFillScalarGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_inplace_fill_scalar(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceFillScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_scalar(value_scalar)
        bindings.acl_destroy_tensor(self_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, value_keep)


def inplace_copy(dst_ptr, src_ptr, dst_shape, dst_stride, dst_dtype, src_shape, src_stride, src_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_inplace_copy_get_workspace is None or bindings.aclnn_inplace_copy is None:
        raise RuntimeError("aclnnInplaceCopy symbols not available")

    dst_tensor, dst_keep = _create_tensor(bindings, dst_shape, dst_stride, dst_dtype, dst_ptr)
    src_tensor, src_keep = _create_tensor(bindings, src_shape, src_stride, src_dtype, src_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_inplace_copy_get_workspace(
            dst_tensor,
            src_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceCopyGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_inplace_copy(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceCopy failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(dst_tensor)
        bindings.acl_destroy_tensor(src_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (dst_keep, src_keep)


def erfinv(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_erfinv_get_workspace is None or bindings.aclnn_erfinv is None:
        raise RuntimeError("aclnnErfinv symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_erfinv_get_workspace(
            self_tensor,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnErfinvGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_erfinv(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnErfinv failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep)


def linalg_qr(self_ptr, q_ptr, r_ptr, self_shape, self_stride, q_shape, q_stride, r_shape, r_stride, dtype, mode, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_linalg_qr_get_workspace is None or bindings.aclnn_linalg_qr is None:
        raise RuntimeError("aclnnLinalgQr symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    q_tensor, q_keep = _create_tensor(bindings, q_shape, q_stride, dtype, q_ptr)
    r_tensor, r_keep = _create_tensor(bindings, r_shape, r_stride, dtype, r_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    try:
        ret = bindings.aclnn_linalg_qr_get_workspace(
            self_tensor,
            ctypes.c_int64(int(mode)),
            q_tensor,
            r_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLinalgQrGetWorkspaceSize failed: {ret}")

        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr

        ret = bindings.aclnn_linalg_qr(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLinalgQr failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(q_tensor)
        bindings.acl_destroy_tensor(r_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, q_keep, r_keep)


# ---------------------------------------------------------------------------
# Symbol checkers for new indexing ops
# ---------------------------------------------------------------------------

def masked_fill_scalar_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_masked_fill_scalar_get_workspace,
            bindings.aclnn_inplace_masked_fill_scalar,
        ])
    except Exception:
        return False


def index_copy_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_index_copy_get_workspace,
            bindings.aclnn_inplace_index_copy,
        ])
    except Exception:
        return False


def index_fill_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_index_fill_get_workspace,
            bindings.aclnn_inplace_index_fill,
        ])
    except Exception:
        return False


def index_add_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_index_add_get_workspace,
            bindings.aclnn_index_add,
        ])
    except Exception:
        return False


def scatter_add_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_scatter_add_get_workspace,
            bindings.aclnn_scatter_add,
        ])
    except Exception:
        return False


def masked_scatter_symbols_ok():
    try:
        bindings = get_bindings()
        return all([
            bindings.aclnn_inplace_masked_scatter_get_workspace,
            bindings.aclnn_inplace_masked_scatter,
        ])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Wrapper functions for new indexing ops
# ---------------------------------------------------------------------------

def inplace_masked_fill_scalar(self_ptr, self_shape, self_stride, self_dtype,
                               mask_ptr, mask_shape, mask_stride, mask_dtype,
                               value, runtime, stream=None):
    """aclnnInplaceMaskedFillScalar — in-place masked fill with scalar."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not masked_fill_scalar_symbols_ok():
        raise RuntimeError("aclnnInplaceMaskedFillScalar symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    mask_tensor, mask_keep = _create_tensor(bindings, mask_shape, mask_stride, mask_dtype, mask_ptr)
    scalar, scalar_keep = _create_scalar(bindings, value, self_dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_inplace_masked_fill_scalar_get_workspace(
            self_tensor, mask_tensor, scalar,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceMaskedFillScalarGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_inplace_masked_fill_scalar(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceMaskedFillScalar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(mask_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, mask_keep, scalar_keep)


def inplace_index_copy(self_ptr, self_shape, self_stride, self_dtype,
                       dim, index_ptr, index_shape, index_stride, index_dtype,
                       source_ptr, source_shape, source_stride, source_dtype,
                       runtime, stream=None):
    """aclnnInplaceIndexCopy — in-place index copy along dim."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not index_copy_symbols_ok():
        raise RuntimeError("aclnnInplaceIndexCopy symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    index_tensor, index_keep = _create_tensor(bindings, index_shape, index_stride, index_dtype, index_ptr)
    source_tensor, source_keep = _create_tensor(bindings, source_shape, source_stride, source_dtype, source_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_inplace_index_copy_get_workspace(
            self_tensor, ctypes.c_int64(dim), index_tensor, source_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceIndexCopyGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_inplace_index_copy(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceIndexCopy failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        bindings.acl_destroy_tensor(source_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, index_keep, source_keep)


def inplace_index_fill(self_ptr, self_shape, self_stride, self_dtype,
                       dim, index_ptr, index_shape, index_stride, index_dtype,
                       value, runtime, stream=None):
    """aclnnInplaceIndexFill — in-place index fill with scalar."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not index_fill_symbols_ok():
        raise RuntimeError("aclnnInplaceIndexFill symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    index_tensor, index_keep = _create_tensor(bindings, index_shape, index_stride, index_dtype, index_ptr)
    scalar, scalar_keep = _create_scalar(bindings, value, self_dtype)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_inplace_index_fill_get_workspace(
            self_tensor, ctypes.c_int64(dim), index_tensor, scalar,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceIndexFillGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_inplace_index_fill(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceIndexFill failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, index_keep, scalar_keep)


def index_add(self_ptr, self_shape, self_stride, self_dtype,
              dim, index_ptr, index_shape, index_stride, index_dtype,
              source_ptr, source_shape, source_stride, source_dtype,
              alpha, out_ptr, out_shape, out_stride, out_dtype,
              runtime, stream=None):
    """aclnnIndexAdd — index add along dim with alpha."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not index_add_symbols_ok():
        raise RuntimeError("aclnnIndexAdd symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    index_tensor, index_keep = _create_tensor(bindings, index_shape, index_stride, index_dtype, index_ptr)
    source_tensor, source_keep = _create_tensor(bindings, source_shape, source_stride, source_dtype, source_ptr)
    alpha_scalar, alpha_keep = _create_scalar(bindings, alpha, self_dtype)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_index_add_get_workspace(
            self_tensor, ctypes.c_int64(dim), index_tensor, source_tensor,
            alpha_scalar, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnIndexAddGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_index_add(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnIndexAdd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        bindings.acl_destroy_tensor(source_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, index_keep, source_keep, alpha_keep, out_keep)


def scatter_add_op(self_ptr, self_shape, self_stride, self_dtype,
                   dim, index_ptr, index_shape, index_stride, index_dtype,
                   src_ptr, src_shape, src_stride, src_dtype,
                   out_ptr, out_shape, out_stride, out_dtype,
                   runtime, stream=None):
    """aclnnScatterAdd — scatter add along dim."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not scatter_add_symbols_ok():
        raise RuntimeError("aclnnScatterAdd symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    index_tensor, index_keep = _create_tensor(bindings, index_shape, index_stride, index_dtype, index_ptr)
    src_tensor, src_keep = _create_tensor(bindings, src_shape, src_stride, src_dtype, src_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_scatter_add_get_workspace(
            self_tensor, ctypes.c_int64(dim), index_tensor, src_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnScatterAddGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_scatter_add(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnScatterAdd failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(index_tensor)
        bindings.acl_destroy_tensor(src_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, index_keep, src_keep, out_keep)


def inplace_masked_scatter(self_ptr, self_shape, self_stride, self_dtype,
                           mask_ptr, mask_shape, mask_stride, mask_dtype,
                           source_ptr, source_shape, source_stride, source_dtype,
                           runtime, stream=None):
    """aclnnInplaceMaskedScatter — in-place masked scatter."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not masked_scatter_symbols_ok():
        raise RuntimeError("aclnnInplaceMaskedScatter symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    mask_tensor, mask_keep = _create_tensor(bindings, mask_shape, mask_stride, mask_dtype, mask_ptr)
    source_tensor, source_keep = _create_tensor(bindings, source_shape, source_stride, source_dtype, source_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_inplace_masked_scatter_get_workspace(
            self_tensor, mask_tensor, source_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceMaskedScatterGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_inplace_masked_scatter(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnInplaceMaskedScatter failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(mask_tensor)
        bindings.acl_destroy_tensor(source_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, mask_keep, source_keep)


# ---------------------------------------------------------------------------
# aclnnVar
# ---------------------------------------------------------------------------
def var_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_var_get_workspace, bindings.aclnn_var])
    except Exception:
        return False


def var(self_ptr, out_ptr, shape, stride, dtype, dims, unbiased, keepdim,
        out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not var_symbols_ok():
        raise RuntimeError("aclnnVar symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)

    if dims:
        dim_array = _make_int64_array(dims)
        dim_handle = bindings.acl_create_int_array(dim_array, ctypes.c_uint64(len(dims)))
    else:
        dim_array = _make_int64_array(list(range(len(shape))))
        dim_handle = bindings.acl_create_int_array(dim_array, ctypes.c_uint64(len(shape)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_var_get_workspace(
            self_tensor,
            dim_handle,
            ctypes.c_bool(unbiased),
            ctypes.c_bool(keepdim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnVarGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_var(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnVar failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(dim_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep)


# ---------------------------------------------------------------------------
# aclnnNorm
# ---------------------------------------------------------------------------
def norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_norm_get_workspace, bindings.aclnn_norm])
    except Exception:
        return False


def norm(self_ptr, out_ptr, shape, stride, dtype, p, dims, keepdim,
         out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not norm_symbols_ok():
        raise RuntimeError("aclnnNorm symbols not available")

    from ..._dtype import float32 as f32
    out_dtype = dtype if getattr(dtype, 'is_floating_point', True) else f32

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)

    p_scalar, p_keep = _create_scalar(bindings, float(p), dtype)

    if dims is not None:
        if isinstance(dims, int):
            dims = [dims]
        dim_array = _make_int64_array(dims)
        dim_handle = bindings.acl_create_int_array(dim_array, ctypes.c_uint64(len(dims)))
    else:
        dim_array = _make_int64_array(list(range(len(shape))))
        dim_handle = bindings.acl_create_int_array(dim_array, ctypes.c_uint64(len(shape)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_norm_get_workspace(
            self_tensor,
            p_scalar,
            dim_handle,
            ctypes.c_bool(keepdim),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnNormGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_norm(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(dim_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, p_keep)


# ---------------------------------------------------------------------------
# aclnnProd / aclnnProdDim
# ---------------------------------------------------------------------------
def prod_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_prod_get_workspace, bindings.aclnn_prod])
    except Exception:
        return False


def prod_dim_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_prod_dim_get_workspace, bindings.aclnn_prod_dim])
    except Exception:
        return False


def prod(self_ptr, out_ptr, shape, stride, dtype, dim, keepdim,
         out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    acl_dtype = ctypes.c_int32(_dtype_to_acl(dtype))

    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None

    if dim is not None:
        if not prod_dim_symbols_ok():
            raise RuntimeError("aclnnProdDim symbols not available")
        d = dim if dim >= 0 else dim + len(shape)
        try:
            ret = bindings.aclnn_prod_dim_get_workspace(
                self_tensor,
                ctypes.c_int64(d),
                ctypes.c_bool(keepdim),
                acl_dtype,
                out_tensor,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnProdDimGetWorkspaceSize failed: {ret}")
            if workspace_size.value:
                workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                workspace = workspace_ptr
            ret = bindings.aclnn_prod_dim(
                ctypes.c_void_p(0 if workspace is None else int(workspace)),
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnProdDim failed: {ret}")
            _maybe_sync(runtime)
        finally:
            _defer_executor(executor)
            bindings.acl_destroy_tensor(self_tensor)
            bindings.acl_destroy_tensor(out_tensor)
            if workspace is not None:
                runtime.defer_free(workspace)
            _ = (self_keep, out_keep)
    else:
        if not prod_symbols_ok():
            raise RuntimeError("aclnnProd symbols not available")
        try:
            ret = bindings.aclnn_prod_get_workspace(
                self_tensor,
                acl_dtype,
                out_tensor,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnProdGetWorkspaceSize failed: {ret}")
            if workspace_size.value:
                workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                workspace = workspace_ptr
            ret = bindings.aclnn_prod(
                ctypes.c_void_p(0 if workspace is None else int(workspace)),
                ctypes.c_uint64(workspace_size.value),
                executor,
                ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
            )
            if ret != 0:
                raise RuntimeError(f"aclnnProd failed: {ret}")
            _maybe_sync(runtime)
        finally:
            _defer_executor(executor)
            bindings.acl_destroy_tensor(self_tensor)
            bindings.acl_destroy_tensor(out_tensor)
            if workspace is not None:
                runtime.defer_free(workspace)
            _ = (self_keep, out_keep)


# ---------------------------------------------------------------------------
# aclnnFloorDivide
# ---------------------------------------------------------------------------
def floor_divide_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_floor_divide_get_workspace, bindings.aclnn_floor_divide])
    except Exception:
        return False


def floor_divide(self_ptr, other_ptr, out_ptr,
                 self_shape, self_stride, other_shape, other_stride,
                 out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not floor_divide_symbols_ok():
        raise RuntimeError("aclnnFloorDivide symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_floor_divide_get_workspace(
            self_tensor, other_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnFloorDivideGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_floor_divide(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnFloorDivide failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, other_keep, out_keep)


# ---------------------------------------------------------------------------
# aclnnRmsNorm
# ---------------------------------------------------------------------------
def rms_norm_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_rms_norm_get_workspace, bindings.aclnn_rms_norm])
    except Exception:
        return False


def rms_norm(x_ptr, gamma_ptr, eps, y_ptr, rstd_ptr,
             x_shape, x_stride, gamma_shape, gamma_stride,
             y_shape, y_stride, rstd_shape, rstd_stride,
             dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not rms_norm_symbols_ok():
        raise RuntimeError("aclnnRmsNorm symbols not available")

    x_tensor, x_keep = _create_tensor(bindings, x_shape, x_stride, dtype, x_ptr)
    gamma_tensor, gamma_keep = _create_tensor(bindings, gamma_shape, gamma_stride, dtype, gamma_ptr)
    y_tensor, y_keep = _create_tensor(bindings, y_shape, y_stride, dtype, y_ptr)
    rstd_tensor, rstd_keep = _create_tensor(bindings, rstd_shape, rstd_stride, dtype, rstd_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_rms_norm_get_workspace(
            x_tensor, gamma_tensor,
            ctypes.c_double(eps),
            y_tensor, rstd_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRmsNormGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_rms_norm(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRmsNorm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(x_tensor)
        bindings.acl_destroy_tensor(gamma_tensor)
        bindings.acl_destroy_tensor(y_tensor)
        bindings.acl_destroy_tensor(rstd_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (x_keep, gamma_keep, y_keep, rstd_keep)


# ---------------------------------------------------------------------------
# aclnnConvolution
# ---------------------------------------------------------------------------
def convolution_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_convolution_get_workspace, bindings.aclnn_convolution])
    except Exception:
        return False


def convolution(input_ptr, weight_ptr, bias_ptr,
                input_shape, input_stride, weight_shape, weight_stride,
                bias_shape, bias_stride, dtype,
                stride, padding, dilation, transposed, output_padding, groups,
                out_ptr, out_shape, out_stride,
                runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not convolution_symbols_ok():
        raise RuntimeError("aclnnConvolution symbols not available")

    # aclnnConvolution requires NCHW format (not ND) for input/weight/output tensors.
    # cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) is required for fp32 on Ascend910B.
    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)
    weight_tensor, weight_keep = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)

    bias_tensor = None
    bias_keep = None
    if bias_ptr is not None:
        bias_tensor, bias_keep = _create_tensor(bindings, bias_shape, bias_stride, dtype, bias_ptr, _ACL_FORMAT_NCHW)

    stride_arr = _make_int64_array(list(stride))
    stride_handle = bindings.acl_create_int_array(stride_arr, ctypes.c_uint64(len(stride)))
    padding_arr = _make_int64_array(list(padding))
    padding_handle = bindings.acl_create_int_array(padding_arr, ctypes.c_uint64(len(padding)))
    dilation_arr = _make_int64_array(list(dilation))
    dilation_handle = bindings.acl_create_int_array(dilation_arr, ctypes.c_uint64(len(dilation)))
    output_padding_arr = _make_int64_array(list(output_padding))
    output_padding_handle = bindings.acl_create_int_array(output_padding_arr, ctypes.c_uint64(len(output_padding)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_convolution_get_workspace(
            input_tensor,
            weight_tensor,
            ctypes.c_void_p(0) if bias_tensor is None else bias_tensor,
            stride_handle,
            padding_handle,
            dilation_handle,
            ctypes.c_bool(transposed),
            output_padding_handle,
            ctypes.c_int64(groups),
            out_tensor,
            ctypes.c_int8(1),  # cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) required for fp32 on Ascend910B
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnConvolutionGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_convolution(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnConvolution failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(stride_handle)
        bindings.acl_destroy_int_array(padding_handle)
        bindings.acl_destroy_int_array(dilation_handle)
        bindings.acl_destroy_int_array(output_padding_handle)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(weight_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if bias_tensor is not None:
            bindings.acl_destroy_tensor(bias_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (input_keep, weight_keep, out_keep, bias_keep,
             stride_arr, padding_arr, dilation_arr, output_padding_arr)


# ---------------------------------------------------------------------------
# aclnnMaxPool
# ---------------------------------------------------------------------------
def max_pool_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_max_pool_get_workspace, bindings.aclnn_max_pool])
    except Exception:
        return False


def max_pool(self_ptr, out_ptr, shape, stride_t, dtype,
             kernel_shape, strides, pads, dilations, ceil_mode,
             out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not max_pool_symbols_ok():
        raise RuntimeError("aclnnMaxPool symbols not available")

    # aclnnMaxPool requires NCHW format. Note: only fp16 dtype is supported on Ascend910B.
    # Callers must cast fp32 input to fp16 before calling this function.
    self_tensor, self_keep = _create_tensor(bindings, shape, stride_t, dtype, self_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)

    ks_arr = _make_int64_array(list(kernel_shape))
    ks_handle = bindings.acl_create_int_array(ks_arr, ctypes.c_uint64(len(kernel_shape)))
    st_arr = _make_int64_array(list(strides))
    st_handle = bindings.acl_create_int_array(st_arr, ctypes.c_uint64(len(strides)))
    # pads for MaxPool is [pH, pW, pH, pW] (4 elements for 2D)
    pads_arr = _make_int64_array(list(pads))
    pads_handle = bindings.acl_create_int_array(pads_arr, ctypes.c_uint64(len(pads)))
    dl_arr = _make_int64_array(list(dilations))
    dl_handle = bindings.acl_create_int_array(dl_arr, ctypes.c_uint64(len(dilations)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_max_pool_get_workspace(
            self_tensor,
            ks_handle,
            st_handle,
            ctypes.c_int64(0),  # autoPad = explicit
            pads_handle,
            dl_handle,
            ctypes.c_int64(1 if ceil_mode else 0),
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxPoolGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_max_pool(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxPool failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(ks_handle)
        bindings.acl_destroy_int_array(st_handle)
        bindings.acl_destroy_int_array(pads_handle)
        bindings.acl_destroy_int_array(dl_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, ks_arr, st_arr, pads_arr, dl_arr)


# ---------------------------------------------------------------------------
# aclnnMaxPool2dWithMask — fp32/fp16-capable, used on Ascend910B
# ---------------------------------------------------------------------------
def max_pool2d_with_mask_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_max_pool2d_with_mask_get_workspace,
                    bindings.aclnn_max_pool2d_with_mask])
    except Exception:
        return False


def max_pool2d_with_mask(self_ptr, out_ptr, mask_ptr,
                         shape, stride_t, dtype,
                         kernel_size, strides, padding, dilations, ceil_mode,
                         out_shape, out_stride, mask_shape, mask_stride,
                         runtime, stream=None):
    """MaxPool2d via aclnnMaxPool2dWithMask, which supports fp32/fp16 on Ascend910B.

    The mask output is an int8 tensor used internally for backward; callers
    typically discard it for forward-only use.
    """
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not max_pool2d_with_mask_symbols_ok():
        raise RuntimeError("aclnnMaxPool2dWithMask symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, shape, stride_t, dtype, self_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)
    # mask indices are int8
    mask_tensor, mask_keep = _create_tensor(bindings, mask_shape, mask_stride, "int8", mask_ptr, _ACL_FORMAT_NCHW)

    ks_arr = _make_int64_array(list(kernel_size))
    ks_handle = bindings.acl_create_int_array(ks_arr, ctypes.c_uint64(len(kernel_size)))
    st_arr = _make_int64_array(list(strides))
    st_handle = bindings.acl_create_int_array(st_arr, ctypes.c_uint64(len(strides)))
    pd_arr = _make_int64_array(list(padding))
    pd_handle = bindings.acl_create_int_array(pd_arr, ctypes.c_uint64(len(padding)))
    dl_arr = _make_int64_array(list(dilations))
    dl_handle = bindings.acl_create_int_array(dl_arr, ctypes.c_uint64(len(dilations)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_max_pool2d_with_mask_get_workspace(
            self_tensor,
            ks_handle,
            st_handle,
            pd_handle,
            dl_handle,
            ctypes.c_bool(ceil_mode),
            out_tensor,
            mask_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxPool2dWithMaskGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_max_pool2d_with_mask(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxPool2dWithMask failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(ks_handle)
        bindings.acl_destroy_int_array(st_handle)
        bindings.acl_destroy_int_array(pd_handle)
        bindings.acl_destroy_int_array(dl_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_tensor(mask_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, mask_keep, ks_arr, st_arr, pd_arr, dl_arr)


# ---------------------------------------------------------------------------
# aclnnAvgPool2d
# ---------------------------------------------------------------------------
def avg_pool2d_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_avg_pool2d_get_workspace, bindings.aclnn_avg_pool2d])
    except Exception:
        return False


def avg_pool2d(self_ptr, out_ptr, shape, stride_t, dtype,
               kernel_size, strides, paddings, ceil_mode, count_include_pad,
               divisor_override, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not avg_pool2d_symbols_ok():
        raise RuntimeError("aclnnAvgPool2d symbols not available")

    # aclnnAvgPool2d requires NCHW format; cubeMathType=1 needed for fp32 on Ascend910B.
    self_tensor, self_keep = _create_tensor(bindings, shape, stride_t, dtype, self_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)

    ks_arr = _make_int64_array(list(kernel_size))
    ks_handle = bindings.acl_create_int_array(ks_arr, ctypes.c_uint64(len(kernel_size)))
    st_arr = _make_int64_array(list(strides))
    st_handle = bindings.acl_create_int_array(st_arr, ctypes.c_uint64(len(strides)))
    pd_arr = _make_int64_array(list(paddings))
    pd_handle = bindings.acl_create_int_array(pd_arr, ctypes.c_uint64(len(paddings)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_avg_pool2d_get_workspace(
            self_tensor,
            ks_handle,
            st_handle,
            pd_handle,
            ctypes.c_bool(ceil_mode),
            ctypes.c_bool(count_include_pad),
            ctypes.c_int64(divisor_override if divisor_override is not None else 0),
            ctypes.c_int8(1),  # cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) required for fp32 on Ascend910B
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAvgPool2dGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_avg_pool2d(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAvgPool2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(ks_handle)
        bindings.acl_destroy_int_array(st_handle)
        bindings.acl_destroy_int_array(pd_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, ks_arr, st_arr, pd_arr)


# ---------------------------------------------------------------------------
# aclnnAdaptiveAvgPool2d
# ---------------------------------------------------------------------------
def adaptive_avg_pool2d_symbols_ok():
    try:
        bindings = get_bindings()
        return all([bindings.aclnn_adaptive_avg_pool2d_get_workspace, bindings.aclnn_adaptive_avg_pool2d])
    except Exception:
        return False


def adaptive_avg_pool2d(self_ptr, out_ptr, shape, stride_t, dtype,
                        output_size, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not adaptive_avg_pool2d_symbols_ok():
        raise RuntimeError("aclnnAdaptiveAvgPool2d symbols not available")

    # aclnnAdaptiveAvgPool2d requires NCHW format.
    self_tensor, self_keep = _create_tensor(bindings, shape, stride_t, dtype, self_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)

    os_arr = _make_int64_array(list(output_size))
    os_handle = bindings.acl_create_int_array(os_arr, ctypes.c_uint64(len(output_size)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_adaptive_avg_pool2d_get_workspace(
            self_tensor,
            os_handle,
            out_tensor,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAdaptiveAvgPool2dGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_adaptive_avg_pool2d(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAdaptiveAvgPool2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(os_handle)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (self_keep, out_keep, os_arr)


# ===========================================================================
# Backward wrapper functions
# ===========================================================================


def softmax_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_softmax_backward_get_workspace, b.aclnn_softmax_backward])
    except Exception:
        return False


def softmax_backward(grad_ptr, output_ptr, out_ptr, shape, grad_stride, output_stride, out_stride,
                     dtype, dim, runtime, stream=None):
    """aclnnSoftmaxBackward(gradOutput, output, dim, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    grad_tensor, grad_keep = _create_tensor(bindings, shape, grad_stride, dtype, grad_ptr)
    output_tensor, output_keep = _create_tensor(bindings, shape, output_stride, dtype, output_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_softmax_backward_get_workspace(
            grad_tensor, output_tensor, ctypes.c_int64(dim), out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSoftmaxBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_softmax_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSoftmaxBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(output_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, output_keep, out_keep)


def gelu_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_gelu_backward_get_workspace, b.aclnn_gelu_backward])
    except Exception:
        return False


def gelu_backward(grad_ptr, self_ptr, out_ptr, shape, grad_stride, self_stride, out_stride,
                  dtype, runtime, stream=None):
    """aclnnGeluBackward(gradOutput, self, gradInput)"""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    grad_tensor, grad_keep = _create_tensor(bindings, shape, grad_stride, dtype, grad_ptr)
    self_tensor, self_keep = _create_tensor(bindings, shape, self_stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_gelu_backward_get_workspace(
            grad_tensor, self_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGeluBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_gelu_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGeluBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, self_keep, out_keep)


def layer_norm_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_layer_norm_backward_get_workspace, b.aclnn_layer_norm_backward])
    except Exception:
        return False


def layer_norm_backward(grad_ptr, input_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
                        grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
                        input_shape, input_stride, stats_shape, stats_stride,
                        weight_shape, weight_stride, bias_shape, bias_stride,
                        normalized_shape, dtype, runtime, stream=None):
    """aclnnLayerNormBackward with outputMask (aclBoolArray)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    out_stride = _contiguous_stride(input_shape)
    grad_tensor, grad_keep = _create_tensor(bindings, input_shape, input_stride, dtype, grad_ptr)
    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr)
    mean_tensor, mean_keep = _create_tensor(bindings, stats_shape, stats_stride, "float32", mean_ptr)
    rstd_tensor, rstd_keep = _create_tensor(bindings, stats_shape, stats_stride, "float32", rstd_ptr)
    grad_input_tensor, gi_keep = _create_tensor(bindings, input_shape, out_stride, dtype, grad_input_ptr)

    weight_tensor = None
    if weight_ptr is not None:
        weight_tensor, _ = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr)
    bias_tensor = None
    if bias_ptr is not None:
        bias_tensor, _ = _create_tensor(bindings, bias_shape, bias_stride, dtype, bias_ptr)
    gw_tensor = None
    if grad_weight_ptr is not None:
        gw_tensor, _ = _create_tensor(bindings, weight_shape, weight_stride, dtype, grad_weight_ptr)
    gb_tensor = None
    if grad_bias_ptr is not None:
        gb_tensor, _ = _create_tensor(bindings, bias_shape, bias_stride, dtype, grad_bias_ptr)

    norm_shape_array = _make_int64_array(normalized_shape)
    norm_shape_handle = bindings.acl_create_int_array(norm_shape_array, ctypes.c_uint64(len(normalized_shape)))

    output_mask = [True, grad_weight_ptr is not None, grad_bias_ptr is not None]
    mask_arr = _make_bool_array(output_mask)
    mask_handle = bindings.acl_create_bool_array(mask_arr, ctypes.c_uint64(3))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_layer_norm_backward_get_workspace(
            grad_tensor, input_tensor, norm_shape_handle,
            mean_tensor, rstd_tensor,
            ctypes.c_void_p(0) if weight_tensor is None else weight_tensor,
            ctypes.c_void_p(0) if bias_tensor is None else bias_tensor,
            mask_handle,
            grad_input_tensor,
            ctypes.c_void_p(0) if gw_tensor is None else gw_tensor,
            ctypes.c_void_p(0) if gb_tensor is None else gb_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLayerNormBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_layer_norm_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnLayerNormBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_int_array(norm_shape_handle)
        bindings.acl_destroy_bool_array(mask_handle)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(mean_tensor)
        bindings.acl_destroy_tensor(rstd_tensor)
        bindings.acl_destroy_tensor(grad_input_tensor)
        if weight_tensor is not None:
            bindings.acl_destroy_tensor(weight_tensor)
        if bias_tensor is not None:
            bindings.acl_destroy_tensor(bias_tensor)
        if gw_tensor is not None:
            bindings.acl_destroy_tensor(gw_tensor)
        if gb_tensor is not None:
            bindings.acl_destroy_tensor(gb_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, input_keep, mean_keep, rstd_keep, gi_keep)


def convolution_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_convolution_backward_get_workspace, b.aclnn_convolution_backward])
    except Exception:
        return False


def convolution_backward(grad_ptr, input_ptr, weight_ptr,
                         grad_shape, grad_stride, input_shape, input_stride,
                         weight_shape, weight_stride, dtype,
                         bias_sizes, stride, padding, dilation, transposed, output_padding, groups,
                         output_mask,
                         grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
                         gi_shape, gi_stride, gw_shape, gw_stride, gb_shape, gb_stride,
                         runtime, stream=None):
    """aclnnConvolutionBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    grad_tensor, grad_keep = _create_tensor(bindings, grad_shape, grad_stride, dtype, grad_ptr, _ACL_FORMAT_NCHW)
    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)
    weight_tensor, weight_keep = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr, _ACL_FORMAT_NCHW)

    gi_tensor = None
    if grad_input_ptr is not None:
        gi_tensor, _ = _create_tensor(bindings, gi_shape, gi_stride, dtype, grad_input_ptr, _ACL_FORMAT_NCHW)
    gw_tensor = None
    if grad_weight_ptr is not None:
        gw_tensor, _ = _create_tensor(bindings, gw_shape, gw_stride, dtype, grad_weight_ptr, _ACL_FORMAT_NCHW)
    gb_tensor = None
    if grad_bias_ptr is not None:
        gb_tensor, _ = _create_tensor(bindings, gb_shape, gb_stride, dtype, grad_bias_ptr, _ACL_FORMAT_NCHW)

    bias_arr = _make_int64_array(list(bias_sizes)) if bias_sizes else None
    bias_handle = bindings.acl_create_int_array(bias_arr, ctypes.c_uint64(len(bias_sizes))) if bias_sizes else None

    stride_arr = _make_int64_array(list(stride))
    stride_handle = bindings.acl_create_int_array(stride_arr, ctypes.c_uint64(len(stride)))
    padding_arr = _make_int64_array(list(padding))
    padding_handle = bindings.acl_create_int_array(padding_arr, ctypes.c_uint64(len(padding)))
    dilation_arr = _make_int64_array(list(dilation))
    dilation_handle = bindings.acl_create_int_array(dilation_arr, ctypes.c_uint64(len(dilation)))
    output_padding_arr = _make_int64_array(list(output_padding))
    output_padding_handle = bindings.acl_create_int_array(output_padding_arr, ctypes.c_uint64(len(output_padding)))
    mask_arr = _make_bool_array([bool(v) for v in output_mask])
    mask_handle = bindings.acl_create_bool_array(mask_arr, ctypes.c_uint64(len(output_mask)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_convolution_backward_get_workspace(
            grad_tensor, input_tensor, weight_tensor,
            ctypes.c_void_p(0) if bias_handle is None else bias_handle,
            stride_handle, padding_handle, dilation_handle,
            ctypes.c_bool(transposed),
            output_padding_handle,
            ctypes.c_int64(groups),
            mask_handle,
            ctypes.c_int8(1),  # cubeMathType=1 for Ascend910B
            ctypes.c_void_p(0) if gi_tensor is None else gi_tensor,
            ctypes.c_void_p(0) if gw_tensor is None else gw_tensor,
            ctypes.c_void_p(0) if gb_tensor is None else gb_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnConvolutionBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_convolution_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnConvolutionBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(weight_tensor)
        if gi_tensor is not None:
            bindings.acl_destroy_tensor(gi_tensor)
        if gw_tensor is not None:
            bindings.acl_destroy_tensor(gw_tensor)
        if gb_tensor is not None:
            bindings.acl_destroy_tensor(gb_tensor)
        if bias_handle is not None:
            bindings.acl_destroy_int_array(bias_handle)
        bindings.acl_destroy_int_array(stride_handle)
        bindings.acl_destroy_int_array(padding_handle)
        bindings.acl_destroy_int_array(dilation_handle)
        bindings.acl_destroy_int_array(output_padding_handle)
        bindings.acl_destroy_bool_array(mask_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, input_keep, weight_keep)


def batch_norm_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_batch_norm_backward_get_workspace, b.aclnn_batch_norm_backward])
    except Exception:
        return False


def batch_norm_backward(grad_ptr, input_ptr, weight_ptr,
                        running_mean_ptr, running_var_ptr,
                        save_mean_ptr, save_invstd_ptr,
                        grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
                        grad_shape, grad_stride, input_shape, input_stride,
                        weight_shape, weight_stride,
                        rm_shape, rm_stride, rv_shape, rv_stride,
                        sm_shape, sm_stride, si_shape, si_stride,
                        gi_shape, gi_stride, gw_shape, gw_stride, gb_shape, gb_stride,
                        training, eps, output_mask, dtype, runtime, stream=None):
    """aclnnBatchNormBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    grad_tensor, grad_keep = _create_tensor(bindings, grad_shape, grad_stride, dtype, grad_ptr, _ACL_FORMAT_NCHW)
    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)

    weight_tensor = None
    if weight_ptr is not None:
        weight_tensor, _ = _create_tensor(bindings, weight_shape, weight_stride, dtype, weight_ptr)
    rm_tensor = None
    if running_mean_ptr is not None:
        rm_tensor, _ = _create_tensor(bindings, rm_shape, rm_stride, dtype, running_mean_ptr)
    rv_tensor = None
    if running_var_ptr is not None:
        rv_tensor, _ = _create_tensor(bindings, rv_shape, rv_stride, dtype, running_var_ptr)

    sm_tensor, sm_keep = _create_tensor(bindings, sm_shape, sm_stride, "float32", save_mean_ptr)
    si_tensor, si_keep = _create_tensor(bindings, si_shape, si_stride, "float32", save_invstd_ptr)

    gi_tensor, gi_keep = _create_tensor(bindings, gi_shape, gi_stride, dtype, grad_input_ptr, _ACL_FORMAT_NCHW)
    gw_tensor = None
    if grad_weight_ptr is not None:
        gw_tensor, _ = _create_tensor(bindings, gw_shape, gw_stride, dtype, grad_weight_ptr)
    gb_tensor = None
    if grad_bias_ptr is not None:
        gb_tensor, _ = _create_tensor(bindings, gb_shape, gb_stride, dtype, grad_bias_ptr)

    mask_arr = _make_bool_array([bool(v) for v in output_mask])
    mask_handle = bindings.acl_create_bool_array(mask_arr, ctypes.c_uint64(len(output_mask)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_batch_norm_backward_get_workspace(
            grad_tensor, input_tensor,
            ctypes.c_void_p(0) if weight_tensor is None else weight_tensor,
            ctypes.c_void_p(0) if rm_tensor is None else rm_tensor,
            ctypes.c_void_p(0) if rv_tensor is None else rv_tensor,
            sm_tensor, si_tensor,
            ctypes.c_bool(training), ctypes.c_double(eps),
            mask_handle,
            gi_tensor,
            ctypes.c_void_p(0) if gw_tensor is None else gw_tensor,
            ctypes.c_void_p(0) if gb_tensor is None else gb_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBatchNormBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_batch_norm_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnBatchNormBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(input_tensor)
        if weight_tensor is not None:
            bindings.acl_destroy_tensor(weight_tensor)
        if rm_tensor is not None:
            bindings.acl_destroy_tensor(rm_tensor)
        if rv_tensor is not None:
            bindings.acl_destroy_tensor(rv_tensor)
        bindings.acl_destroy_tensor(sm_tensor)
        bindings.acl_destroy_tensor(si_tensor)
        bindings.acl_destroy_tensor(gi_tensor)
        if gw_tensor is not None:
            bindings.acl_destroy_tensor(gw_tensor)
        if gb_tensor is not None:
            bindings.acl_destroy_tensor(gb_tensor)
        bindings.acl_destroy_bool_array(mask_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, input_keep, sm_keep, si_keep, gi_keep)


def embedding_dense_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_embedding_dense_backward_get_workspace, b.aclnn_embedding_dense_backward])
    except Exception:
        return False


def embedding_dense_backward(grad_ptr, indices_ptr, grad_weight_ptr,
                             grad_shape, grad_stride, indices_shape, indices_stride,
                             gw_shape, gw_stride, grad_dtype, indices_dtype,
                             num_weights, padding_idx, scale_grad_by_freq,
                             runtime, stream=None):
    """aclnnEmbeddingDenseBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    grad_tensor, grad_keep = _create_tensor(bindings, grad_shape, grad_stride, grad_dtype, grad_ptr)
    indices_tensor, idx_keep = _create_tensor(bindings, indices_shape, indices_stride, indices_dtype, indices_ptr)
    gw_tensor, gw_keep = _create_tensor(bindings, gw_shape, gw_stride, grad_dtype, grad_weight_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_embedding_dense_backward_get_workspace(
            grad_tensor, indices_tensor,
            ctypes.c_int64(num_weights),
            ctypes.c_int64(padding_idx if padding_idx is not None else -1),
            ctypes.c_bool(scale_grad_by_freq),
            gw_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEmbeddingDenseBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_embedding_dense_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEmbeddingDenseBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(indices_tensor)
        bindings.acl_destroy_tensor(gw_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, idx_keep, gw_keep)


# ---------------------------------------------------------------
# max_pool2d backward
# ---------------------------------------------------------------

def max_pool2d_with_mask_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_max_pool2d_with_mask_backward_get_workspace,
                    b.aclnn_max_pool2d_with_mask_backward])
    except Exception:
        return False


def max_pool2d_with_mask_backward(grad_ptr, input_ptr, mask_ptr, grad_input_ptr,
                                   grad_shape, grad_stride, input_shape, input_stride,
                                   mask_shape, mask_stride,
                                   gi_shape, gi_stride,
                                   kernel_size, strides, padding, dilation, ceil_mode,
                                   dtype, runtime, stream=None):
    """aclnnMaxPool2dWithMaskBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    grad_tensor, grad_keep = _create_tensor(bindings, grad_shape, grad_stride, dtype, grad_ptr, _ACL_FORMAT_NCHW)
    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)
    mask_tensor, mask_keep = _create_tensor(bindings, mask_shape, mask_stride, "int8", mask_ptr)
    gi_tensor, gi_keep = _create_tensor(bindings, gi_shape, gi_stride, dtype, grad_input_ptr, _ACL_FORMAT_NCHW)

    ks_arr = _make_int64_array(list(kernel_size))
    ks_handle = bindings.acl_create_int_array(ks_arr, ctypes.c_uint64(len(kernel_size)))
    st_arr = _make_int64_array(list(strides))
    st_handle = bindings.acl_create_int_array(st_arr, ctypes.c_uint64(len(strides)))
    pad_arr = _make_int64_array(list(padding))
    pad_handle = bindings.acl_create_int_array(pad_arr, ctypes.c_uint64(len(padding)))
    dil_arr = _make_int64_array(list(dilation))
    dil_handle = bindings.acl_create_int_array(dil_arr, ctypes.c_uint64(len(dilation)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_max_pool2d_with_mask_backward_get_workspace(
            grad_tensor, input_tensor, mask_tensor,
            ks_handle, st_handle, pad_handle, dil_handle,
            ctypes.c_bool(ceil_mode),
            gi_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_max_pool2d_with_mask_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMaxPool2dWithMaskBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(mask_tensor)
        bindings.acl_destroy_tensor(gi_tensor)
        bindings.acl_destroy_int_array(ks_handle)
        bindings.acl_destroy_int_array(st_handle)
        bindings.acl_destroy_int_array(pad_handle)
        bindings.acl_destroy_int_array(dil_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, input_keep, mask_keep, gi_keep)


# ---------------------------------------------------------------
# avg_pool2d backward
# ---------------------------------------------------------------

def avg_pool2d_backward_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_avg_pool2d_backward_get_workspace, b.aclnn_avg_pool2d_backward])
    except Exception:
        return False


def avg_pool2d_backward(grad_ptr, input_ptr, grad_input_ptr,
                         grad_shape, grad_stride, input_shape, input_stride,
                         gi_shape, gi_stride,
                         kernel_size, strides, padding,
                         ceil_mode, count_include_pad, divisor_override,
                         dtype, runtime, stream=None):
    """aclnnAvgPool2dBackward."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    grad_tensor, grad_keep = _create_tensor(bindings, grad_shape, grad_stride, dtype, grad_ptr, _ACL_FORMAT_NCHW)
    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)
    gi_tensor, gi_keep = _create_tensor(bindings, gi_shape, gi_stride, dtype, grad_input_ptr, _ACL_FORMAT_NCHW)

    ks_arr = _make_int64_array(list(kernel_size))
    ks_handle = bindings.acl_create_int_array(ks_arr, ctypes.c_uint64(len(kernel_size)))
    st_arr = _make_int64_array(list(strides))
    st_handle = bindings.acl_create_int_array(st_arr, ctypes.c_uint64(len(strides)))
    pad_arr = _make_int64_array(list(padding))
    pad_handle = bindings.acl_create_int_array(pad_arr, ctypes.c_uint64(len(padding)))

    # divisor_override is int64_t (0 means no override)
    div_val = divisor_override if divisor_override is not None else 0

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_avg_pool2d_backward_get_workspace(
            grad_tensor, input_tensor,
            ks_handle, st_handle, pad_handle,
            ctypes.c_bool(ceil_mode), ctypes.c_bool(count_include_pad),
            ctypes.c_int64(div_val),
            ctypes.c_int8(1),  # cubeMathType=1 for Ascend910B
            gi_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAvgPool2dBackwardGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_avg_pool2d_backward(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAvgPool2dBackward failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(grad_tensor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(gi_tensor)
        bindings.acl_destroy_int_array(ks_handle)
        bindings.acl_destroy_int_array(st_handle)
        bindings.acl_destroy_int_array(pad_handle)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (grad_keep, input_keep, gi_keep)


# ---------------------------------------------------------------
# rms_norm backward (grad)
# ---------------------------------------------------------------

def rms_norm_grad_symbols_ok():
    try:
        b = get_bindings()
        return all([b.aclnn_rms_norm_grad_get_workspace, b.aclnn_rms_norm_grad])
    except Exception:
        return False


def rms_norm_grad(dy_ptr, x_ptr, rstd_ptr, gamma_ptr,
                   dx_ptr, dgamma_ptr,
                   dy_shape, dy_stride, x_shape, x_stride,
                   rstd_shape, rstd_stride,
                   gamma_shape, gamma_stride,
                   dx_shape, dx_stride, dgamma_shape, dgamma_stride,
                   dtype, runtime, stream=None):
    """aclnnRmsNormGrad(dy, x, rstd, gamma, dx, dgamma)."""
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()

    dy_tensor, dy_keep = _create_tensor(bindings, dy_shape, dy_stride, dtype, dy_ptr)
    x_tensor, x_keep = _create_tensor(bindings, x_shape, x_stride, dtype, x_ptr)
    rstd_tensor, rstd_keep = _create_tensor(bindings, rstd_shape, rstd_stride, dtype, rstd_ptr)
    gamma_tensor, gamma_keep = _create_tensor(bindings, gamma_shape, gamma_stride, dtype, gamma_ptr)
    dx_tensor, dx_keep = _create_tensor(bindings, dx_shape, dx_stride, dtype, dx_ptr)
    dgamma_tensor, dgamma_keep = _create_tensor(bindings, dgamma_shape, dgamma_stride, dtype, dgamma_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_rms_norm_grad_get_workspace(
            dy_tensor, x_tensor, rstd_tensor, gamma_tensor,
            dx_tensor, dgamma_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRmsNormGradGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_rms_norm_grad(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRmsNormGrad failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(dy_tensor)
        bindings.acl_destroy_tensor(x_tensor)
        bindings.acl_destroy_tensor(rstd_tensor)
        bindings.acl_destroy_tensor(gamma_tensor)
        bindings.acl_destroy_tensor(dx_tensor)
        bindings.acl_destroy_tensor(dgamma_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (dy_keep, x_keep, rstd_keep, gamma_keep, dx_keep, dgamma_keep)


# ---------------------------------------------------------------
# P1 ops: reciprocal, addmm, einsum, upsample_nearest2d,
#          upsample_bilinear2d, one_hot
# ---------------------------------------------------------------

def reciprocal_symbols_ok():
    b = get_bindings()
    return b.aclnn_reciprocal_get_workspace is not None and b.aclnn_reciprocal is not None

def reciprocal(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    return _unary_call(bindings, "aclnnReciprocal",
                       bindings.aclnn_reciprocal_get_workspace, bindings.aclnn_reciprocal,
                       self_ptr, out_ptr, shape, stride, dtype, runtime, stream)


def addmm_symbols_ok():
    b = get_bindings()
    return b.aclnn_addmm_get_workspace is not None and b.aclnn_addmm is not None

def addmm(self_ptr, mat1_ptr, mat2_ptr, out_ptr,
          self_shape, self_stride, self_dtype,
          mat1_shape, mat1_stride,
          mat2_shape, mat2_stride,
          out_shape, out_stride,
          beta, alpha, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not addmm_symbols_ok():
        raise RuntimeError("aclnnAddmm symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    mat1_tensor, m1_keep = _create_tensor(bindings, mat1_shape, mat1_stride, self_dtype, mat1_ptr)
    mat2_tensor, m2_keep = _create_tensor(bindings, mat2_shape, mat2_stride, self_dtype, mat2_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, self_dtype, out_ptr)

    beta_scalar, beta_buf = _create_scalar(bindings, beta, self_dtype)
    alpha_scalar, alpha_buf = _create_scalar(bindings, alpha, self_dtype)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_addmm_get_workspace(
            self_tensor, mat1_tensor, mat2_tensor,
            beta_scalar, alpha_scalar,
            out_tensor, ctypes.c_int8(1),  # cubeMathType=1
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAddmmGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_addmm(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnAddmm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(mat1_tensor)
        bindings.acl_destroy_tensor(mat2_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_scalar(beta_scalar)
        bindings.acl_destroy_scalar(alpha_scalar)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, m1_keep, m2_keep, out_keep, beta_buf, alpha_buf)


def einsum_symbols_ok():
    b = get_bindings()
    return b.aclnn_einsum_get_workspace is not None and b.aclnn_einsum is not None

def einsum(tensor_ptrs, shapes, strides, dtypes, equation,
           out_ptr, out_shape, out_stride, out_dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not einsum_symbols_ok():
        raise RuntimeError("aclnnEinsum symbols not available")

    tensor_list, tensor_keeps = _create_tensor_list(bindings, tensor_ptrs, shapes, strides, dtypes)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)

    eq_bytes = equation.encode('utf-8') + b'\x00'
    eq_buf = ctypes.create_string_buffer(eq_bytes)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_einsum_get_workspace(
            ctypes.c_void_p(tensor_list),
            eq_buf,
            out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEinsumGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_einsum(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnEinsum failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        for tensor, keep in tensor_keeps:
            bindings.acl_destroy_tensor(tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (tensor_keeps, out_keep, eq_buf)


def upsample_nearest2d_symbols_ok():
    b = get_bindings()
    return b.aclnn_upsample_nearest2d_get_workspace is not None and b.aclnn_upsample_nearest2d is not None

def upsample_nearest2d(input_ptr, out_ptr, input_shape, input_stride, dtype,
                       output_size, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_nearest2d_symbols_ok():
        raise RuntimeError("aclnnUpsampleNearest2d symbols not available")

    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)

    size_array = _make_int64_array(list(output_size))
    size_handle = bindings.acl_create_int_array(size_array, ctypes.c_uint64(len(output_size)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_upsample_nearest2d_get_workspace(
            input_tensor, size_handle, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnUpsampleNearest2dGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_upsample_nearest2d(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnUpsampleNearest2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (input_keep, out_keep, size_array)


def upsample_bilinear2d_symbols_ok():
    b = get_bindings()
    return b.aclnn_upsample_bilinear2d_get_workspace is not None and b.aclnn_upsample_bilinear2d is not None

def upsample_bilinear2d(input_ptr, out_ptr, input_shape, input_stride, dtype,
                        output_size, align_corners, scales_h, scales_w,
                        out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not upsample_bilinear2d_symbols_ok():
        raise RuntimeError("aclnnUpsampleBilinear2d symbols not available")

    input_tensor, input_keep = _create_tensor(bindings, input_shape, input_stride, dtype, input_ptr, _ACL_FORMAT_NCHW)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr, _ACL_FORMAT_NCHW)

    size_array = _make_int64_array(list(output_size))
    size_handle = bindings.acl_create_int_array(size_array, ctypes.c_uint64(len(output_size)))

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_upsample_bilinear2d_get_workspace(
            input_tensor, size_handle,
            ctypes.c_bool(align_corners),
            ctypes.c_double(scales_h if scales_h is not None else 0.0),
            ctypes.c_double(scales_w if scales_w is not None else 0.0),
            out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnUpsampleBilinear2dGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_upsample_bilinear2d(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnUpsampleBilinear2d failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(input_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (input_keep, out_keep, size_array)


def one_hot_symbols_ok():
    b = get_bindings()
    return b.aclnn_one_hot_get_workspace is not None and b.aclnn_one_hot is not None

def one_hot(self_ptr, on_ptr, off_ptr, out_ptr,
            self_shape, self_stride, self_dtype,
            on_shape, on_stride, on_dtype,
            off_shape, off_stride, off_dtype,
            out_shape, out_stride, out_dtype,
            num_classes, axis, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not one_hot_symbols_ok():
        raise RuntimeError("aclnnOneHot symbols not available")

    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, self_dtype, self_ptr)
    on_tensor, on_keep = _create_tensor(bindings, on_shape, on_stride, on_dtype, on_ptr)
    off_tensor, off_keep = _create_tensor(bindings, off_shape, off_stride, off_dtype, off_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)

    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_one_hot_get_workspace(
            self_tensor,
            ctypes.c_int64(num_classes),
            on_tensor, off_tensor,
            ctypes.c_int64(axis),
            out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnOneHotGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_one_hot(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnOneHot failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(on_tensor)
        bindings.acl_destroy_tensor(off_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, on_keep, off_keep, out_keep)


# Dot product (vector dot vector -> scalar)
def dot(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_dot_get_workspace is None or bindings.aclnn_dot is None:
        raise RuntimeError("aclnnDot symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_dot_get_workspace(
            self_tensor, other_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDotGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_dot(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnDot failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


# Matrix-vector multiplication
def mv(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
       out_shape, out_stride, dtype, cube_math_type, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_mv_get_workspace is None or bindings.aclnn_mv is None:
        raise RuntimeError("aclnnMv symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_mv_get_workspace(
            self_tensor, other_tensor, out_tensor,
            ctypes.c_int8(cube_math_type),
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMvGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_mv(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMv failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


# Ger (outer product): vector outer vector -> matrix
def ger(self_ptr, other_ptr, out_ptr, self_shape, self_stride, other_shape, other_stride,
        out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_ger_get_workspace is None or bindings.aclnn_ger is None:
        raise RuntimeError("aclnnGer symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, self_shape, self_stride, dtype, self_ptr)
    other_tensor, other_keep = _create_tensor(bindings, other_shape, other_stride, dtype, other_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_ger_get_workspace(
            self_tensor, other_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGerGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_ger(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnGer failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(other_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, other_keep, out_keep)


# Global median (reduces all elements to scalar)
def median(self_ptr, out_ptr, shape, stride, dtype, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_median_get_workspace is None or bindings.aclnn_median is None:
        raise RuntimeError("aclnnMedian symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_median_get_workspace(
            self_tensor, out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMedianGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_median(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMedian failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep)


# Median along a dimension
def median_dim(self_ptr, out_ptr, indices_ptr,
               shape, stride, dtype,
               out_shape, out_stride,
               dim, keepdim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_median_dim_get_workspace is None or bindings.aclnn_median_dim is None:
        raise RuntimeError("aclnnMedianDim symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    indices_tensor, indices_keep = _create_tensor(bindings, out_shape, out_stride, "int64", indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_median_dim_get_workspace(
            self_tensor,
            ctypes.c_int64(dim),
            ctypes.c_bool(keepdim),
            out_tensor,
            indices_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMedianDimGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_median_dim(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMedianDim failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_tensor(indices_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep, indices_keep)


# Kthvalue
def kthvalue(self_ptr, out_ptr, indices_ptr,
             shape, stride, dtype,
             out_shape, out_stride,
             k, dim, keepdim, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_kthvalue_get_workspace is None or bindings.aclnn_kthvalue is None:
        raise RuntimeError("aclnnKthvalue symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    indices_tensor, indices_keep = _create_tensor(bindings, out_shape, out_stride, "int64", indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_kthvalue_get_workspace(
            self_tensor,
            ctypes.c_int64(k),
            ctypes.c_int64(dim),
            ctypes.c_bool(keepdim),
            out_tensor,
            indices_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnKthvalueGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_kthvalue(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnKthvalue failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_tensor(indices_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep, indices_keep)


# SearchSorted
def search_sorted(sorted_sequence_ptr, values_ptr, out_ptr,
                  sorted_sequence_shape, sorted_sequence_stride,
                  values_shape, values_stride, out_shape, out_stride,
                  dtype, out_int32, right, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_search_sorted_get_workspace is None or bindings.aclnn_search_sorted is None:
        raise RuntimeError("aclnnSearchSorted symbols not available")
    sorted_tensor, sorted_keep = _create_tensor(bindings, sorted_sequence_shape, sorted_sequence_stride, dtype, sorted_sequence_ptr)
    values_tensor, values_keep = _create_tensor(bindings, values_shape, values_stride, dtype, values_ptr)
    out_dtype = "int32" if out_int32 else "int64"
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, out_dtype, out_ptr)
    # sorter is not used (nullptr)
    sorter_ptr = ctypes.c_void_p(0)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_search_sorted_get_workspace(
            sorted_tensor, values_tensor,
            ctypes.c_bool(out_int32),
            ctypes.c_bool(right),
            sorter_ptr,
            out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSearchSortedGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_search_sorted(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnSearchSorted failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(sorted_tensor)
        bindings.acl_destroy_tensor(values_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (sorted_keep, values_keep, out_keep)


# Unique
def unique(self_ptr, out_ptr, inverse_indices_ptr,
           shape, stride, dtype,
           out_shape, out_stride,
           inverse_shape, inverse_stride,
           sorted, return_inverse,
           runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_unique_get_workspace is None or bindings.aclnn_unique is None:
        raise RuntimeError("aclnnUnique symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    inverse_tensor, inverse_keep = _create_tensor(bindings, inverse_shape, inverse_stride, "int64", inverse_indices_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_unique_get_workspace(
            self_tensor,
            ctypes.c_bool(sorted),
            ctypes.c_bool(return_inverse),
            out_tensor,
            inverse_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnUniqueGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_unique(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnUnique failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        bindings.acl_destroy_tensor(inverse_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep, inverse_keep)


# Randperm
def randperm(n, out_ptr, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_randperm_get_workspace is None or bindings.aclnn_randperm is None:
        raise RuntimeError("aclnnRandperm symbols not available")
    shape = (n,)
    stride = (1,)
    out_tensor, out_keep = _create_tensor(bindings, shape, stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        import random
        seed = random.randint(0, 2**31 - 1)
        offset = 0
        ret = bindings.aclnn_randperm_get_workspace(
            ctypes.c_int64(n),
            ctypes.c_int64(seed),
            ctypes.c_int64(offset),
            out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRandpermGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_randperm(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnRandperm failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (out_keep,)


# Flatten (ACLNN version always produces 2D output based on axis)
def flatten(self_ptr, out_ptr, shape, stride, dtype, axis, out_shape, out_stride, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if bindings.aclnn_flatten_get_workspace is None or bindings.aclnn_flatten is None:
        raise RuntimeError("aclnnFlatten symbols not available")
    self_tensor, self_keep = _create_tensor(bindings, shape, stride, dtype, self_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_flatten_get_workspace(
            self_tensor, ctypes.c_int64(axis), out_tensor,
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnFlattenGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_flatten(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value), executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnFlatten failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(self_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_raw_free(workspace)
        _ = (self_keep, out_keep)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n
