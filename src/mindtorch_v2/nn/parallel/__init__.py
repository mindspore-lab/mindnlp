"""Parallel module package for mindtorch_v2.

Provides DataParallel and DistributedDataParallel compatibility classes.
"""

from ..modules.parallel import DataParallel, DistributedDataParallel

__all__ = ['DataParallel', 'DistributedDataParallel']
