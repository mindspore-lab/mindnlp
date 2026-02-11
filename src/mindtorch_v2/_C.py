from ._backends import ascend


def _npu_probe_model_dirs():
    return ascend._probe_model_dirs()
