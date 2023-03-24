"""utils for mindspore backward compatibility."""
import mindspore
from packaging import version

MIN_COMPATIBLE_VERSION = '1.8.1'
MAX_GRAPH_FIRST_VERSION = '1.10.0'

less_min_compatible = version.parse(mindspore.__version__) < version.parse(MIN_COMPATIBLE_VERSION)
less_min_pynative_first = version.parse(mindspore.__version__) <= version.parse(MAX_GRAPH_FIRST_VERSION)

__all__ = ['less_min_compatible', 'less_min_pynative_first']
