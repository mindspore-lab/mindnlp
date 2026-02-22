from .distributed import DistributedDataParallel
from .data_parallel import DataParallel, data_parallel

__all__ = ["DistributedDataParallel", "DataParallel", "data_parallel"]
