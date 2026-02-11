import os
import numpy as np
import acl

# Constants mirrored from acl_engine/tools/constant.py
ACL_ERROR_CODE = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_FORMAT_ND = 2

ACL_DTYPE = {
    "float": 0,
    "float16": 1,
    "int64": 9,
}


class _Runtime:
    def __init__(self):
        self.initialized = False
        self.device_id = 0
        self.context = None
        self.stream = None

    def init(self, device_id=0):
        if self.initialized:
            return
        ret = acl.init()
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.init failed: {ret}")
        ret = acl.rt.set_device(device_id)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.set_device failed: {ret}")
        self.context, ret = acl.rt.create_context(device_id)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.create_context failed: {ret}")
        self.stream, ret = acl.rt.create_stream()
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.create_stream failed: {ret}")
        self.device_id = device_id
        self.initialized = True

    def finalize(self):
        if not self.initialized:
            return
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        self.stream = None
        self.context = None
        self.initialized = False


_runtime = _Runtime()
_MODEL_DIR = None

_CANDIDATE_MODEL_DIRS = (
    "/usr/local/Ascend/ascend-toolkit/latest/opp",
    "/home/lvyufeng/lvyufeng/acl_engine",
)


def is_available():
    try:
        _runtime.init(0)
        return True
    except Exception:
        return False


def _probe_model_dirs():
    global _MODEL_DIR
    _runtime.init(0)
    for path in _CANDIDATE_MODEL_DIRS:
        if not os.path.isdir(path):
            continue
        ret = acl.op.set_model_dir(path)
        if ret != ACL_ERROR_CODE:
            continue
        _MODEL_DIR = path
        return True
    _MODEL_DIR = None
    return False


def _model_dir():
    if _MODEL_DIR is None:
        _probe_model_dirs()
    if _MODEL_DIR is None:
        raise RuntimeError("NPU op model dir not initialized")
    return _MODEL_DIR


def _dtype_to_acl(dtype):
    if dtype == "float32":
        return ACL_DTYPE["float"]
    if dtype == "float16":
        return ACL_DTYPE["float16"]
    if dtype == "int64":
        return ACL_DTYPE["int64"]
    raise ValueError(f"Unsupported dtype for NPU: {dtype}")


def _alloc_device(size):
    dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc failed: {ret}")
    return dev_ptr


def _memcpy_h2d(dst, size, src_ptr):
    ret = acl.rt.memcpy(dst, size, src_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy H2D failed: {ret}")


def _memcpy_d2h(dst_ptr, size, src):
    ret = acl.rt.memcpy(dst_ptr, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.memcpy D2H failed: {ret}")


def _create_desc(dtype, shape):
    return acl.create_tensor_desc(_dtype_to_acl(dtype), list(shape), ACL_FORMAT_ND)


def _create_buffer(dev_ptr, size):
    return acl.create_data_buffer(dev_ptr, size)


def _execute_op(op_type, in_descs, in_bufs, out_descs, out_bufs, attr):
    ret = acl.op.update_params(op_type, in_descs, out_descs, attr)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.op.update_params failed: {ret}")
    ret = acl.op.execute_v2(op_type, in_descs, in_bufs, out_descs, out_bufs, attr, _runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.op.execute_v2 failed: {ret}")
    ret = acl.rt.synchronize_stream(_runtime.stream)
    if ret != ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")


class NpuStorage:
    def __init__(self, device_ptr, shape, dtype):
        self.device_ptr = device_ptr
        self.shape = tuple(shape)
        self.dtype = dtype


class NpuTensor:
    def __init__(self, storage, shape, stride):
        self.storage = storage
        self.shape = tuple(shape)
        self.stride = tuple(stride)


# Minimal op implementations using execute_v2 + op models from ASCEND_OPP_PATH.

def add(a, b):
    _runtime.init(0)
    op_type = "Add"
    in_descs = [
        _create_desc(a.storage.dtype, a.storage.shape),
        _create_desc(b.storage.dtype, b.storage.shape),
    ]
    out_descs = [_create_desc(a.storage.dtype, a.storage.shape)]
    out_size = acl.get_tensor_desc_size(out_descs[0])
    out_ptr = _alloc_device(out_size)
    in_bufs = [
        _create_buffer(a.storage.device_ptr, acl.get_tensor_desc_size(in_descs[0])),
        _create_buffer(b.storage.device_ptr, acl.get_tensor_desc_size(in_descs[1])),
    ]
    out_bufs = [_create_buffer(out_ptr, out_size)]
    attr = acl.op.create_attr()
    _execute_op(op_type, in_descs, in_bufs, out_descs, out_bufs, attr)
    return NpuTensor(NpuStorage(out_ptr, a.storage.shape, a.storage.dtype), a.storage.shape, a.stride)
