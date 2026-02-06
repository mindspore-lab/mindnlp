"""Configuration constants for mindtorch_v2 compatibility.

Auto-detects device target from MindSpore context at import time.
"""

import os
import mindspore

# Device target - auto-detect from MindSpore context
try:
    DEVICE_TARGET = mindspore.get_context('device_target')
except Exception:
    DEVICE_TARGET = "CPU"

# SoC version for Ascend devices
SOC = None
if DEVICE_TARGET == 'Ascend':
    try:
        from mindspore._c_expression import MSContext
        SOC = MSContext.get_instance().get_ascend_soc_version()
    except Exception:
        pass

# Hardware feature flags
# 910A (ascend910) doesn't support BF16
ON_A1 = SOC == 'ascend910' if SOC else False
ON_A2 = SOC in ['ascend910b', 'ascend910_93'] if SOC else False
ON_ORANGE_PI = '310b' in SOC if SOC else False

# BF16 only supported on newer Ascend chips (not 910A, not 310B)
SUPPORT_BF16 = DEVICE_TARGET == 'Ascend' and not ON_A1 and not ON_ORANGE_PI

# Set memory allocation config for 910A to show real memory in npu-smi
if ON_A1:
    os.environ.setdefault("MS_ALLOC_CONF", "enable_vmm:True,vmm_align_size:2MB")
