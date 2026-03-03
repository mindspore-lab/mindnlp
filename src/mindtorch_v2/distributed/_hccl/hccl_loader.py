import ctypes
import os
import threading

_HCCL_LIB = None
_HCCL_LOCK = threading.Lock()

_CANDIDATE_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/lib64",
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
    "/usr/local/Ascend/latest/lib64",
    "/usr/local/Ascend/latest/aarch64-linux/lib64",
)


def ensure_hccl():
    global _HCCL_LIB
    if _HCCL_LIB is not None:
        return _HCCL_LIB
    with _HCCL_LOCK:
        if _HCCL_LIB is not None:
            return _HCCL_LIB
        for d in _CANDIDATE_DIRS:
            path = os.path.join(d, "libhccl.so")
            if os.path.exists(path):
                _HCCL_LIB = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return _HCCL_LIB
        raise RuntimeError(
            "libhccl.so not found. Searched: " + ", ".join(_CANDIDATE_DIRS)
        )
