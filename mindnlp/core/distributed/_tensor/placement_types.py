"""
NOTE: core.distributed._tensor has been moved to core.distributed.tensor.
The imports here are purely for backward compatibility. We will remove these
imports in a few releases

TODO: throw warnings when this module imported
"""

from core.distributed.tensor._dtensor_spec import *  # noqa: F401, F403
from core.distributed.tensor.placement_types import *  # noqa: F401, F403
