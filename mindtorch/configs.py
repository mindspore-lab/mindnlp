import os
from packaging import version
import mindspore
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

SOC = MSContext.get_instance().get_ascend_soc_version()
DEVICE_TARGET = mindspore.get_context('device_target')
SUPPORT_BF16 = DEVICE_TARGET == 'Ascend' and SOC not in ['ascend910', 'ascend310b']
ON_A1 = SOC == 'ascend910'
ON_A2 = SOC in ['ascend910b', 'ascend910_93']
ON_ORANGE_PI = '310b' in SOC
DEFAULT_DTYPE = mindspore.float32
MS27 = '.'.join(mindspore.__version__.split('.')[:2]) >= '2.7'

# OP backend select
USE_PYBOOST = True
CPU_USE_NUMPY_OP = bool(os.environ.get('CPU_USE_NUMPY', False))

def set_pyboost(mode: bool):
    """set global pyboost"""
    global USE_PYBOOST
    USE_PYBOOST = mode

def use_pyboost():
    """set global pyboost"""
    return USE_PYBOOST

def set_cpu_use_numpy(mode: bool):
    """set global pyboost"""
    global CPU_USE_NUMPY_OP
    CPU_USE_NUMPY_OP = mode

def cpu_use_numpy():
    """set global pyboost"""
    return CPU_USE_NUMPY_OP