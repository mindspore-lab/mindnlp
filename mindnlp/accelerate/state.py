"""accelerate state"""
import os
from functools import partial
from contextlib import contextmanager
from typing import Callable, Any
from mindspore import communication
try:
    from mindspore.communication.comm_func import barrier
except:
    barrier = None

from .utils import (
    DistributedType, is_mindformers_available
)

SharedDict = dict


# Lambda function that does nothing
def do_nothing(*args, **kwargs):
    return None


class PartialState:
    _shared_state = SharedDict()
    _know_attrs = [
        "_cpu",
        "_mixed_precision",
        "_shared_state",
        "backend",
        "debug",
        "device",
        "distributed_type",
        "fork_launched",
        "local_process_index",
        "num_processes",
        "process_index",
    ]

    def __init__(self, **kwargs):
        self.__dict__ = self._shared_state
        self._prepare_backend()

        if self.backend == "hccl":
            self.num_processes = communication.get_group_size()
            self.process_index = communication.get_rank()

    def __repr__(self) -> str:
        return (
            f"Distributed environment: {self.distributed_type}{('  Backend: ' + self.backend) if self.backend else ''}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
        )

    @staticmethod
    def _reset_state():
        """Resets `_shared_state`, is used internally and should not be called"""
        PartialState._shared_state.clear()

    @property
    def initialized(self) -> bool:
        """Returns whether the `PartialState` has been initialized"""
        return self._shared_state

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return self.distributed_type != DistributedType.NO and self.num_processes > 1

    @property
    def is_last_process(self) -> bool:
        """Returns whether the current process is the last one"""
        return self.process_index == self.num_processes - 1

    @property
    def is_main_process(self) -> bool:
        """Returns whether the current process is the main process"""
        return (
            self.process_index == 0 if self.distributed_type != DistributedType.MINDFORMERS else self.is_last_process
        )

    @property
    def num_processes(self):
        """Returns num process"""
        return self.num_processes

    @property
    def process_index(self):
        """Returns process index"""
        return self.process_index

    @property
    def is_local_main_process(self) -> bool:
        """Returns whether the current process is the main process on the local node"""
        return (
            self.local_process_index == 0
            if self.distributed_type != DistributedType.MINDFORMERS
            else self.is_last_process
        )

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> import time
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> if state.is_main_process:
        ...     time.sleep(2)
        >>> else:
        ...     print("I'm waiting for the main process to finish its sleep...")
        >>> state.wait_for_everyone()
        >>> # Should print on every process at the same time
        >>> print("Everyone is here")
        ```
        """
        if self.distributed_type in (
                DistributedType.MINDFORMERS,
        ):
            barrier()

    def _goes_first(self, is_main: bool):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.main_process_first():
        ...     # This will be printed first by process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.process_index}")
        ```
        """
        yield from self._goes_first(self.is_main_process)

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> with state.local_main_process_first():
        ...     # This will be printed first by local process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {state.local_process_index}")
        ```
        """
        yield from self._goes_first(self.is_local_main_process)

    def on_main_process(self, function: Callable[..., Any] = None):
        """
        Decorator that only runs the decorated function on the main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()


        >>> @state.on_main_process
        ... def print_something():
        ...     print("This will be printed by process 0 only.")


        >>> print_something()
        "This will be printed by process 0 only"
        ```
        """
        if not self.initialized:
            raise ValueError("The `PartialState` or `Accelerator` must be initialized before calling this function.")
        if self.is_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_local_main_process(self, function: Callable[..., Any] = None):
        """
        Decorator that only runs the decorated function on the local main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_local_main_process
        def print_something():
            print("This will be printed by process 0 only on each server.")


        print_something()
        # On server 1:
        "This will be printed by process 0 only"
        # On server 2:
        "This will be printed by process 0 only"
        ```
        """
        if self.is_local_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_last_process(self, function: Callable[..., Any]):
        """
        Decorator that only runs the decorated function on the last process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_last_process
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 3"
        ```
        """
        if self.is_last_process or not self.use_distributed:
            return function
        return do_nothing

    def on_process(self, function: Callable[..., Any] = None, process_index: int = None):
        """
        Decorator that only runs the decorated function on the process with the given index.

        Args:
            function (`Callable`, `optional`):
                The function to decorate.
            process_index (`int`, `optional`):
                The index of the process on which to run the function.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_process(process_index=2)
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_process, process_index=process_index)
        if (self.process_index == process_index) or (not self.use_distributed):
            return function
        return do_nothing

    def on_local_process(self, function: Callable[..., Any] = None, local_process_index: int = None):
        """
        Decorator that only runs the decorated function on the process with the given index on the current node.

        Args:
            function (`Callable`, *optional*):
                The function to decorate.
            local_process_index (`int`, *optional*):
                The index of the local process on which to run the function.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_process(local_process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.local_process_index}")


        print_something()
        # On server 1:
        "Printed on process 2"
        # On server 2:
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_local_process, local_process_index=local_process_index)
        if (self.local_process_index == local_process_index) or (not self.use_distributed):
            return function
        return do_nothing

    def print(self, *args, **kwargs):
        if self.is_local_main_process:
            print(*args, **kwargs)

    def _prepare_backend(self):
        # now mindformers only
        if is_mindformers_available():
            self.backend = "hccl"
            self.distributed_type = DistributedType.MINDFORMERS

    @num_processes.setter
    def num_processes(self, value):
        self._num_processes = value

    @process_index.setter
    def process_index(self, value):
        self._process_index = value


class AcceleratorState:
    _shared_state = SharedDict()
    _know_attrs = PartialState._know_attrs + [
        "mindformers_plugin"
    ]

    def __init__(self, mindformers_plugin=None, **kwargs):
        self.__dict__ = self._shared_state
        if PartialState._shared_state:
            PartialState(**kwargs)
        self.__dict__.update(PartialState._shared_state)

        if os.environ.get("ACCELERATE_USE_MINDFORMERS", "false") == "true":
            self.distributed_type = DistributedType.MINDFORMERS
            self.mindformers_plugin = mindformers_plugin

        PartialState._shared_state["distributed_type"] = self.distributed_type

    def __repr__(self):
        return PartialState().__repr__()

    @property
    def initialized(self) -> bool:
        return self._shared_state != PartialState._shared_state

    @staticmethod
    def _reset_state(reset_partial_state: bool = False):
        """Resets `_shared_state`, is used internally and should not be called"""
        AcceleratorState._shared_state.clear()
        if reset_partial_state:
            PartialState._reset_state()

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return PartialState().use_distributed

    @property
    def is_last_process(self) -> bool:
        """Returns whether the current process is the last one"""
        return PartialState().is_last_process

    @property
    def is_main_process(self) -> bool:
        """Returns whether the current process is the main process"""
        return PartialState().is_main_process

    @property
    def is_local_main_process(self) -> bool:
        """Returns whether the current process is the main process on the local node"""
        return PartialState().is_local_main_process

    @property
    def num_processes(self):
        """Returns num process"""
        return PartialState().num_processes

    @property
    def process_index(self):
        """Returns process index"""
        return PartialState().process_index

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.
        """
        PartialState().wait_for_everyone()

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        with PartialState().main_process_first():
            yield

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        with PartialState().local_main_process_first():
            yield

    def print(self, *args, **kwargs):
        PartialState().print(*args, **kwargs)
