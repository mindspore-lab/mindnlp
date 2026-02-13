import ctypes
import os
import sys
import threading


_ACL_READY = False
_ACL_MODULE = None
_ACL_LOCK = threading.Lock()

_CANDIDATE_LIB_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/lib64",
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
    "/usr/local/Ascend/ascend-toolkit/latest/opp/lib64",
    "/usr/local/Ascend/latest/lib64",
    "/usr/local/Ascend/driver/lib64/driver",
)


_CANDIDATE_PYTHON_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
    "/usr/local/Ascend/latest/python/site-packages",
)

_PRELOAD_LIBS = (
    "libascendcl.so",
    "libascendcl_impl.so",
    "libopapi.so",
)


def _existing_dirs():
    return [path for path in _CANDIDATE_LIB_DIRS if os.path.isdir(path)]



def _existing_python_dirs():
    return [path for path in _CANDIDATE_PYTHON_DIRS if os.path.isdir(path)]


def _append_python_path(paths):
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def _prepend_ld_library_path(paths):
    if not paths:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(paths)
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


def _preload_libs(paths):
    for base in paths:
        for lib in _PRELOAD_LIBS:
            candidate = os.path.join(base, lib)
            if os.path.exists(candidate):
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)


def _import_acl():
    import acl

    return acl


def ensure_acl():
    global _ACL_READY, _ACL_MODULE
    if _ACL_READY:
        return _ACL_MODULE
    with _ACL_LOCK:
        if _ACL_READY:
            return _ACL_MODULE
        paths = _existing_dirs()
        _prepend_ld_library_path(paths)
        _preload_libs(paths)
        try:
            _ACL_MODULE = _import_acl()
        except ModuleNotFoundError:
            _append_python_path(_existing_python_dirs())
            _ACL_MODULE = _import_acl()
        _ACL_READY = True
        return _ACL_MODULE
