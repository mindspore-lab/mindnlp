"""global constants for mindnlp"""
import os
import psutil

# from .devices import _is_Ascend_npu_avaliable, _avaliable_Ascend_npus_count #TODU: if use acl
from .dataclasses import DistributedType



def detect_actual_distributed_type():
    """
    the actual_distributed_type isn't the distributed_type users wanted in the startup command, such as:
        1. NPU is available, specified 'msrun' ==> NPU
        2. mschrun specifies parallel npu, but npu is not available ==> cpu execution (reasonable)
        3. NPU is available, but the user python x.py start without specifying the information of the number of port cards to initialize the communication, and the actual_distributed_type is CPU
        .etc

    Returns:
        _type_: According to the factors such as the available parallel software and hardware environment of the current system and the user-specified parallel scheme,
          the optimal parallel strategy is comprehensively decided in different situations.
    """
    if os.environ.get("MULTI_NPU_DATA_PARALLEL", None) == "true": 
        # TODO: 暂时用环境变量 MULTI_NPU_DATA_PARALLEL 作为开关，讨论是否改为这个取代 DistributedType.MINDFORMERS 作为兜底策略
        return DistributedType.MULTI_NPU_DATA_PARALLEL
    if os.environ.get("ACCELERATE_USE_MINDFORMERS", "false") == "true": 
        # TODO: 在原有逻辑中，没有配置环境变量的情况下默认使用 DistributedType.MINDFORMERS 。这里是否需要删掉
        return DistributedType.MINDFORMERS
    else:
        return DistributedType.NO
    
_actual_distributed_type = detect_actual_distributed_type()
    