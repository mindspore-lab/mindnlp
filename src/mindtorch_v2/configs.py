"""Configuration constants for mindtorch_v2 compatibility."""

import mindspore

# Device target - detect from MindSpore context
try:
    DEVICE_TARGET = mindspore.get_context('device_target')
except Exception:
    DEVICE_TARGET = "CPU"

# SoC type - detect from MindSpore if on Ascend
SOC = None
if DEVICE_TARGET == 'Ascend':
    try:
        from mindspore._c_expression import MSContext
        SOC = MSContext.get_instance().get_ascend_soc_version()
    except Exception:
        pass

# Hardware feature flags
SUPPORT_BF16 = DEVICE_TARGET == 'Ascend' and SOC is not None
ON_A1 = SOC == 'ascend910' if SOC else False
ON_ORANGE_PI = False
