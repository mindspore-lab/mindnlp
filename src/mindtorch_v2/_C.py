from ._backends import ascend
from ._backends import ascend_ctypes


def _npu_probe_model_dirs():
    return ascend._probe_model_dirs()


def _npu_model_dir():
    return ascend._model_dir()


def _npu_aclnn_available():
    return ascend_ctypes.is_available()


def _npu_aclnn_symbols_ok():
    return ascend_ctypes.symbols_ok()
