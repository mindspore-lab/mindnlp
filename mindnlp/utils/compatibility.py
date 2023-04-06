"""utils for mindspore backward compatibility."""
import mindspore
from packaging import version

MIN_COMPATIBLE_VERSION = '1.8.1'
MAX_GRAPH_FIRST_VERSION = '1.12.0'
API_COMPATIBLE_VERSION = '1.10.1'

MS_VERSION = mindspore.__version__
MS_VERSION = MS_VERSION.replace('rc', '')

less_min_minddata_compatible = version.parse(MS_VERSION) <= version.parse(MIN_COMPATIBLE_VERSION)
less_min_compatible = version.parse(MS_VERSION) < version.parse(MIN_COMPATIBLE_VERSION)
less_min_pynative_first = version.parse(MS_VERSION) <= version.parse(MAX_GRAPH_FIRST_VERSION)
less_min_api_compatible = version.parse(MS_VERSION) <= version.parse(API_COMPATIBLE_VERSION)

__all__ = [
    'less_min_compatible',
    'less_min_pynative_first',
    'less_min_api_compatible',
    'less_min_minddata_compatible'
]
