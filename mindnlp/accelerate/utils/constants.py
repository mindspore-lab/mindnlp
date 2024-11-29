"""constants"""
import os
from .dataclasses import DistributedType

def detect_accelerate_distributed_type():
    """
    detect distributed_type

    Returns:
        _type_: According to the factors such as the available parallel software and hardware environment of the current system and the user-specified parallel scheme,
          the optimal parallel strategy is comprehensively decided in different situations.
    """
    if os.environ.get("MULTI_NPU_DP", None) == "true": 
        return DistributedType.MULTI_NPU_DP
    if os.environ.get("ACCELERATE_USE_MINDFORMERS", "false") == "true": 
        return DistributedType.MINDFORMERS
    else:
        return DistributedType.NO

accelerate_distributed_type = detect_accelerate_distributed_type()
    