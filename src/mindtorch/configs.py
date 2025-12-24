import os
import warnings
import mindspore
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

SOC = MSContext.get_instance().get_ascend_soc_version()
DEVICE_TARGET = mindspore.get_context('device_target')
SUPPORT_BF16 = DEVICE_TARGET == 'Ascend' and SOC not in ['ascend910', 'ascend310b']
ON_A1 = SOC == 'ascend910'
ON_A2 = SOC in ['ascend910b', 'ascend910_93']
ON_ORANGE_PI = '310b' in SOC
DEFAULT_DTYPE = mindspore.float32
FLASH_ATTN_MASK_VALID = int(os.environ.get('FLASH_ATTN_MASK_VALID', 1))

if ON_A1 or DEVICE_TARGET == 'GPU':
    warnings.warn('MindSpore on GPU/910A do not support bfloat16, use float16 instead.')
    mindspore.bfloat16 = mindspore.float16


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value

# OP backend select
ENABLE_DISPATCH = parse_flag_from_env('ENABLE_DISPATCH', True)
ENABLE_PYBOOST = parse_flag_from_env('ENABLE_PYBOOST', True)
CPU_USE_NUMPY_OP = parse_flag_from_env('CPU_USE_NUMPY', DEVICE_TARGET != 'CPU')
ENABLE_FLASH_ATTENTION = parse_flag_from_env('ENABLE_FLASH_ATTENTION', False)
CAPTURE_INF_NAN = parse_flag_from_env('CAPTURE_INF_NAN', False)
