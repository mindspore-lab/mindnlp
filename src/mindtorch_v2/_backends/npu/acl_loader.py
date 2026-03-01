import ctypes
import glob
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

_LATEST_TOOLKIT_ROOT = "/usr/local/Ascend/ascend-toolkit/latest"
_LATEST_OPP_DIR = "/usr/local/Ascend/ascend-toolkit/latest/opp"

_PRELOAD_LIBS = (
    "libascend_protobuf.so",
    "libascendcl.so",
)


def _existing_dirs():
    return [path for path in _CANDIDATE_LIB_DIRS if os.path.isdir(path)]



def _existing_python_dirs():
    return [path for path in _CANDIDATE_PYTHON_DIRS if os.path.isdir(path)]


def _append_python_path(paths):
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def _align_ascend_env_to_latest():
    # Some shells inject stale paths like ".../laster".
    # Force a single coherent latest toolkit view for ACL/ACLNN.
    if os.path.isdir(_LATEST_TOOLKIT_ROOT):
        os.environ["ASCEND_TOOLKIT_HOME"] = _LATEST_TOOLKIT_ROOT
        os.environ["ASCEND_HOME_PATH"] = _LATEST_TOOLKIT_ROOT
    if os.path.isdir(_LATEST_OPP_DIR):
        os.environ["ASCEND_OPP_PATH"] = _LATEST_OPP_DIR


def _sanitize_ld_library_path(value):
    if not value:
        return ""
    keep = []
    for item in value.split(":"):
        if not item:
            continue
        # Remove toolkit stub paths so real GE/runtime libs are always preferred.
        if "/runtime/lib64/stub" in item:
            continue
        keep.append(item)
    return ":".join(keep)


def _prepend_ld_library_path(paths):
    if not paths:
        return
    existing = _sanitize_ld_library_path(os.environ.get("LD_LIBRARY_PATH", ""))
    prefix = ":".join(paths)
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


def _preload_libs(paths):
    # Ensure protobuf/provider libs are globally visible before loading ACLNN deps.
    for base in paths:
        candidate = os.path.join(base, "libascend_protobuf.so")
        if os.path.exists(candidate):
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break
        for match in sorted(glob.glob(candidate + "*")):
            if not os.path.isfile(match):
                continue
            ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
            break

    for base in paths:
        candidate = os.path.join(base, "libplatform.so")
        if os.path.exists(candidate):
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break

    for base in paths:
        for lib in _PRELOAD_LIBS:
            candidate = os.path.join(base, lib)
            if os.path.exists(candidate):
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                continue
            if lib == "libascend_protobuf.so":
                for match in sorted(glob.glob(candidate + "*")):
                    if not os.path.isfile(match):
                        continue
                    ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
                    break


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
        _align_ascend_env_to_latest()
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
