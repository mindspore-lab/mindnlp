from enum import IntEnum


class RedOpType(IntEnum):
    SUM = 0
    PRODUCT = 1
    MAX = 2
    MIN = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    PREMUL_SUM = 8
    UNUSED = 9


class ReduceOp:
    SUM = RedOpType.SUM
    PRODUCT = RedOpType.PRODUCT
    MAX = RedOpType.MAX
    MIN = RedOpType.MIN
    BAND = RedOpType.BAND
    BOR = RedOpType.BOR
    BXOR = RedOpType.BXOR
    AVG = RedOpType.AVG
    PREMUL_SUM = RedOpType.PREMUL_SUM
    UNUSED = RedOpType.UNUSED

    RedOpType = RedOpType

    def __init__(self, op=None):
        if op is not None:
            self._op = RedOpType(op)
        else:
            self._op = RedOpType.SUM

    def __int__(self):
        return int(self._op)

    def __eq__(self, other):
        if isinstance(other, ReduceOp):
            return self._op == other._op
        if isinstance(other, (int, RedOpType)):
            return int(self._op) == int(other)
        return NotImplemented

    def __hash__(self):
        return hash(self._op)

    def __repr__(self):
        return f"<ReduceOp.{self._op.name}: {self._op.value}>"


# Deprecated alias matching PyTorch
class reduce_op:
    SUM = ReduceOp.SUM
    PRODUCT = ReduceOp.PRODUCT
    MAX = ReduceOp.MAX
    MIN = ReduceOp.MIN
