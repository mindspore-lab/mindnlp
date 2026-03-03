from ._backends.npu import runtime as npu_runtime
from ._backends.npu import aclnn


def _npu_probe_model_dirs():
    return npu_runtime._probe_model_dirs()


def _npu_model_dir():
    return npu_runtime._model_dir()


def _npu_aclnn_available():
    return aclnn.is_available()


def _npu_aclnn_symbols_ok():
    return aclnn.symbols_ok()

def _npu_aclnn_ones_zero_ok():
    return aclnn.ones_zero_symbols_ok()

def _npu_device_count():
    return npu_runtime.device_count()
