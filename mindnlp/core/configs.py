from packaging import version
import mindspore
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

SOC = MSContext.get_instance().get_ascend_soc_version()
DEVICE_TARGET = mindspore.get_context('device_target')
SUPPORT_BF16 = SOC in ["ascend910b", "ascend910_93"]
ON_ORANGE_PI = '310b' in SOC
USE_PYBOOST = DEVICE_TARGET == 'Ascend'
DEFAULT_DTYPE = mindspore.float32
MS27 = '.'.join(mindspore.__version__.split('.')[:2]) >= '2.7'


def set_pyboost(mode: bool):
    """set global pyboost"""
    global USE_PYBOOST
    USE_PYBOOST = mode

def use_pyboost():
    """set global pyboost"""
    return USE_PYBOOST