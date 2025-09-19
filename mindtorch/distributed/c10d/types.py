import mindtorch
from mindtorch import Tensor
from typing import Optional, List
from datetime import timedelta


class _SupplementBase:
    def __del__(self):
        pass


class NCCLPreMulSumSupplement(_SupplementBase):
    def __init__(self, factor):
        if isinstance(factor, float):
            self.double_factor = factor
            self.tensor_factor = None
        elif isinstance(factor, Tensor):
            self.double_factor = 0.0
            assert factor.numel() == 1, "Tensor must have exactly one element."
            self.tensor_factor = factor
        else:
            raise ValueError("factor must be either a float or a Tensor")


class ReduceOp:
    SUM = 'sum'
    AVG = 1
    PRODUCT = 'prod'
    MIN = 'min'
    MAX = 'max'
    BAND = 5  # Bitwise AND
    BOR = 6   # Bitwise OR
    BXOR = 7  # Bitwise XOR
    PREMUL_SUM = 8  # Multiply by a user-supplied constant before summing.
    UNUSED = 9

    def __init__(self, op: Optional[int] = None, supplement: Optional[_SupplementBase] = None):
        if op is None:
            self.op = self.SUM
        else:
            if op == self.PREMUL_SUM:
                raise ValueError("Use `make_ncc_premul_sum` to create an instance of ReduceOp with PREMUL_SUM")
            self.op = op
        
        if supplement:
            self.supplement = supplement
        else:
            self.supplement = None

    def __eq__(self, other):
        if isinstance(other, int):
            return self.op == other
        elif isinstance(other, ReduceOp):
            return self.op == other.op
        return False

    def __int__(self):
        return self.op


def make_ncc_premul_sum(factor):
    rop = ReduceOp()
    rop.op = ReduceOp.PREMUL_SUM
    rop.supplement = NCCLPreMulSumSupplement(factor)
    return rop


kUnsetTimeout = timedelta(milliseconds=-1)


class BroadcastOptions:
    def __init__(self):
        self.rootRank = 0
        self.rootTensor = 0
        self.timeout = kUnsetTimeout
        self.asyncOp = True


class AllreduceOptions:
    def __init__(self):
        self.reduceOp = ReduceOp.SUM
        self.timeout = kUnsetTimeout
        self.sparseIndices = None


class AllreduceCoalescedOptions(AllreduceOptions):
    pass


class ReduceOptions:
    def __init__(self):
        self.reduceOp = ReduceOp.SUM
        self.rootRank = 0
        self.rootTensor = 0
        self.timeout = kUnsetTimeout


class AllgatherOptions:
    def __init__(self):
        self.timeout = kUnsetTimeout
        self.asyncOp = True


class GatherOptions:
    def __init__(self):
        self.rootRank = 0
        self.timeout = kUnsetTimeout


class ScatterOptions:
    def __init__(self):
        self.rootRank = 0
        self.timeout = kUnsetTimeout
        self.asyncOp = True


class ReduceScatterOptions:
    def __init__(self):
        self.reduceOp = ReduceOp.SUM
        self.timeout = kUnsetTimeout
        self.asyncOp = True


class AllToAllOptions:
    def __init__(self):
        self.timeout = kUnsetTimeout


class BarrierOptions:
    def __init__(self):
        self.device_ids = []
        self.timeout = kUnsetTimeout
        self.device = None


class DistributedBackendOptions:
    def __init__(self, store, group_rank, group_size, timeout, group_id, global_ranks_in_group):
        self.store = store
        self.group_rank = group_rank
        self.group_size = group_size
        self.timeout = timeout
        self.group_id = group_id
        self.global_ranks_in_group = global_ranks_in_group
