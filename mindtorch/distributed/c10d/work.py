import mindtorch
from mindtorch import Tensor
from enum import Enum
from typing import List, Optional, Callable
import time
import threading


class OpType(Enum):
    BROADCAST = 0
    ALLREDUCE = 1
    ALLREDUCE_COALESCED = 2
    REDUCE = 3
    ALLGATHER = 4
    _ALLGATHER_BASE = 5
    ALLGATHER_COALESCED = 6
    GATHER = 7
    SCATTER = 8
    REDUCE_SCATTER = 9
    ALLTOALL_BASE = 10
    ALLTOALL = 11
    SEND = 12
    RECV = 13
    RECVANYSOURCE = 14
    BARRIER = 15
    _REDUCE_SCATTER_BASE = 16
    COALESCED = 17
    _ALLREDUCE_SPARSE = 18
    UNKNOWN = 100


class WorkResult(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    COMM_ERROR = 2
    UNKNOWN = 100


kNoTimeout = 0  # Default to 0 for no timeout


def op_type_to_string(op_type: OpType) -> str:
    """Converts OpType to human-readable string."""
    return op_type.name


def is_p2p_op(op_type: OpType, batch_p2p=False) -> bool:
    """Determines if the operation is point-to-point."""
    if batch_p2p:
        return False
    return op_type in {OpType.SEND, OpType.RECV, OpType.RECVANYSOURCE}


class Work:
    def __init__(self, rank=-1, op_type=OpType.UNKNOWN, profiling_title=None, input_tensors=None):
        self.rank = rank
        self.op_type = op_type
        self.completed = False
        self.exception = None
        self.cv = threading.Condition()
        self.record_function_end_callback = None
        self.input_tensors = input_tensors

        if profiling_title is not None:
            # Simulate the profiling functionality in Python (simplified)
            self.record_function_end_callback = lambda: print(f"Profiling {profiling_title} ended.")

    def is_completed(self) -> bool:
        """Non-blocking check for completion."""
        with self.cv:
            return self.completed

    def is_success(self) -> bool:
        """Returns True if work is successful."""
        with self.cv:
            return self.exception is None

    def exception(self):
        """Returns the exception if work was unsuccessful."""
        with self.cv:
            return self.exception

    def source_rank(self) -> int:
        """Returns source rank for recv-from-any work."""
        raise NotImplementedError("sourceRank() is only available for recv or recv-from-any operations.")

    def result(self) -> List[Tensor]:
        """Returns result tensors."""
        raise NotImplementedError("Result not implemented for this operation.")

    def synchronize(self):
        """Ensure synchronization of operations on output tensors."""
        if self.is_completed() and self.is_success():
            if self.record_function_end_callback:
                self.record_function_end_callback()

    def wait(self, timeout=kNoTimeout) -> bool:
        """Wait for completion of the work."""
        with self.cv:
            if timeout == kNoTimeout:
                self.cv.wait_for(lambda: self.completed)
            else:
                self.cv.wait(timeout)
                if not self.completed:
                    raise TimeoutError("Operation timed out!")
            if self.exception:
                raise self.exception
            return self.is_success()

    def abort(self):
        """Aborts the work."""
        raise NotImplementedError("Abort is not implemented.")

    def get_future(self):
        """Returns a Future object associated with the work."""
        raise NotImplementedError("getFuture is not implemented.")

    def get_future_result(self):
        """Returns a Future object that marks success or failure."""
        raise NotImplementedError("getFutureResult is not implemented.")

    def finish(self, exception=None):
        """Complete the work and notify waiting threads."""
        with self.cv:
            self.completed = True
            self.exception = exception
            if self.record_function_end_callback:
                self.record_function_end_callback()
            self.cv.notify_all()

    def finish_and_throw(self, exception):
        """Finish work and throw exception."""
        with self.cv:
            self.completed = True
            self.exception = exception
            if self.record_function_end_callback:
                self.record_function_end_callback()
            if self.exception:
                raise self.exception

    def get_duration(self) -> float:
        """Get the duration of the work."""
        raise NotImplementedError("This backend doesn't support getDuration.")

    def get_sequence_number(self) -> int:
        """Get the sequence number for the work."""
        raise NotImplementedError("This backend doesn't support getSequenceNumber.")

    @staticmethod
    def create_from_future(future):
        """Create a Work object from a Future."""
        return FutureWrappingWork(future)


class FutureWrappingWork(Work):
    def __init__(self, fut):
        super().__init__()
        self._fut = fut

    def is_completed(self) -> bool:
        """Checks if the future is completed."""
        return self._fut.completed()

    def is_success(self) -> bool:
        """Checks if the future has succeeded."""
        return self._fut.has_value()

    def exception(self):
        """Returns exception if any."""
        return self._fut.exception_ptr()

    def source_rank(self) -> int:
        raise NotImplementedError("FutureWrappingWork::sourceRank() not implemented")

    def result(self) -> List[Tensor]:
        return self._fut.value().to_py_object_holder().extract_tensors()

    def wait(self, timeout=kNoTimeout) -> bool:
        """Waits for the future to complete."""
        if timeout != kNoTimeout:
            raise NotImplementedError("Timeout handling not implemented for FutureWrappingWork.")
        self._fut.wait()
        return True

    def abort(self):
        raise NotImplementedError("abort not implemented for FutureWrappingWork.")

    def get_future(self):
        return self._fut


class WorkInfo:
    def __init__(self, op_type: OpType, seq: int, time_started, time_finished, active_duration):
        self.op_type = op_type
        self.seq = seq
        self.time_started = time_started
        self.time_finished = time_finished
        self.active_duration = active_duration
