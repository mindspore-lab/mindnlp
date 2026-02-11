import ctypes
import os


_LIB_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
    "/usr/local/Ascend/latest/aarch64-linux/lib64",
)

_LIBS = (
    "libaclnn_ops_infer.so",
    "libaclnn_math.so",
)


def _load_libs():
    libs = []
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
    return libs


def is_available():
    try:
        _load_libs()
        return True
    except Exception:
        return False
