# coding=utf-8
# Copyright 2020 Optuna, Hugging Face
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=unused-import
""" Logging utilities."""

import functools
import logging
import os
import sys
import threading
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from logging import captureWarnings as _captureWarnings
from typing import Optional

from tqdm import auto as tqdm_lib


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING

_tqdm_active = True


def _get_default_logging_level():
    """
    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        logging.getLogger().warning(
            f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
            f"has to be one of: { ', '.join(log_levels.keys()) }"
        )
    return _default_log_level


def _get_library_name() -> str:
    """
    Returns the name of the library based on the module name.
    
    Returns:
        str: The name of the library extracted from the module name.
    
    """
    return __name__.split(".")[0] # pylint: disable=use-maxsplit-arg


def _get_library_root_logger() -> logging.Logger:
    """
    Retrieves the root logger for the library.
    
    Returns:
        A logging.Logger object representing the root logger for the library.
    
    Raises:
        None.
    """
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    """
    This function configures the root logger for the library.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None
    """
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        # set defaults based on https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        # if logging level is debug, we add pathname and lineno to formatter for easy debugging
        if os.getenv("TRANSFORMERS_VERBOSITY", None) == "detail":
            formatter = logging.Formatter("[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s")
            _default_handler.setFormatter(formatter)

        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    """
    Resets the root logger of the library to its default state.
    
    Args:
        None
    
    Returns:
        None. The function does not return any value.
    
    Raises:
        None
    """
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_log_levels_dict():
    """
    Returns a dictionary of log levels.
    
    Returns:
        dict: A dictionary containing log levels and their corresponding values.
    """
    return log_levels


def captureWarnings(capture):
    """
    Calls the `captureWarnings` method from the logging library to enable management of the warnings emitted by the
    `warnings` library.

    Read more about this method here:
    https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module

    All warnings will be logged through the `py.warnings` logger.

    Careful: this method also adds a handler to this logger if it does not already have one, and updates the logging
    level of that logger to the library's root logger.
    """
    logger = get_logger("py.warnings")

    if not logger.handlers:
        logger.addHandler(_default_handler)

    logger.setLevel(_get_library_root_logger().level)

    _captureWarnings(capture)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 Transformers has following logging levels:

    - 50: `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
    - 40: `transformers.logging.ERROR`
    - 30: `transformers.logging.WARNING` or `transformers.logging.WARN`
    - 20: `transformers.logging.INFO`
    - 10: `transformers.logging.DEBUG`

    </Tip>"""
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
            - `transformers.logging.ERROR`
            - `transformers.logging.WARNING` or `transformers.logging.WARN`
            - `transformers.logging.INFO`
            - `transformers.logging.DEBUG`
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: logging.Handler) -> None:
    """adds a handler to the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    """removes given handler from the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """
    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace Transformers's default handler to
    prevent double logging if the root logger has been configured.
    """
    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)


def reset_format() -> None:
    """
    Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var TRANSFORMERS_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv("NO_ADVISORY_WARNINGS", False) # pylint: disable=invalid-envvar-default
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)


logging.Logger.warning_advice = warning_advice


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""
    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of the EmptyTqdm class.
        
        Args:
            self: The instance of the EmptyTqdm class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        self._iterator = args[0] if args else None

    def __iter__(self):
        """
        This method implements the iterator protocol for the EmptyTqdm class.
        
        Args:
            self: EmptyTqdm object. The instance of the EmptyTqdm class for which the iterator is being created.
        
        Returns:
            None. This method returns an iterator object that iterates over the _iterator attribute of the EmptyTqdm instance.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""
        def empty_fn(*args, **kwargs):
            return
        return empty_fn

    def __enter__(self):
        """
        __enter__
        
        Args:
            self: EmptyTqdm
                The self parameter refers to the current instance of the EmptyTqdm class.
        
        Returns:
            None
                This method returns None.
        
        Raises:
            No exceptions are raised by this method.
        """
        return self

    def __exit__(self, type_, value, traceback):
        """
        __exit__ method in the EmptyTqdm class.
        
        Args:
            self: EmptyTqdm object
                The instance of the EmptyTqdm class.
            type_: type
                The type of the exception. It represents the type of the exception being handled.
            value: exception
                The exception that was raised. It represents the actual exception object.
            traceback: traceback
                The traceback object. It represents the traceback information associated with the exception.
        
        Returns:
            None
            This method does not return any value.
        
        Raises:
            This method does not raise any exceptions explicitly.
        """
        return


class _tqdm_cls:

    """_tqdm_cls is a Python class that provides functionality for managing the progress of tasks. It includes methods for calling the class, setting a lock, and getting a lock. This class is designed to work
in conjunction with the tqdm_lib module for displaying progress bars during iterative processes. When _tqdm_active is True, the class uses methods from the tqdm_lib.tqdm module to handle progress tracking.
Otherwise, it falls back to using an EmptyTqdm instance for progress tracking. The set_lock method allows users to specify a lock for thread safety, and the get_lock method retrieves the current lock if one
has been set."""
    def __call__(self, *args, **kwargs):
        """
        This method __call__ in the class _tqdm_cls is used to conditionally return either a tqdm object or an EmptyTqdm object based on the _tqdm_active flag.
        
        Args:
            self (object): The instance of the _tqdm_cls class. It is used to access the attributes and methods of the class.
        
        Returns:
            None: This method does not explicitly return any value. It returns either a tqdm object or an EmptyTqdm object based on the _tqdm_active flag.
        
        Raises:
            No specific exceptions are raised by this method under normal circumstances. However, if there are issues related to the instantiation of tqdm objects or EmptyTqdm objects, standard Python
exceptions may be raised.
        """
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        """
        Method to set the lock for the _tqdm_cls instance.
        
        Args:
            self (_tqdm_cls): The instance of the _tqdm_cls class.
                This parameter is required to access the instance and set the lock.
                It is of type _tqdm_cls and represents the instance on which the lock is being set.
        
        Returns:
            None: This method does not return any value. The lock is set within the instance itself.
        
        Raises:
            No specific exceptions are raised by this method.
            However, if _tqdm_active is False, the method will not set the lock and will return without any further action.
        """
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        """
        This method is used to retrieve the lock used by the _tqdm_cls class.
        
        Args:
            self (object): The instance of the _tqdm_cls class.
            
        Returns:
            None: This method does not return any value.
        
        Raises:
            N/A
        """
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active # pylint: disable=global-variable-not-assigned
    return bool(_tqdm_active)


def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True


def disable_progress_bar():
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False
