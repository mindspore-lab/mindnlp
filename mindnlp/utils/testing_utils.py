# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Utils for test cases."""
import collections
import contextlib
import doctest
import functools
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import asyncio
from collections.abc import Mapping
from collections import defaultdict

from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch

import urllib3
import numpy as np

import mindspore
from mindnlp.utils import logging as mindnlp_logging

from .import_utils import (
    is_pytest_available,
    is_mindspore_available,
    is_essentia_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_pyctcdecode_available,
    is_safetensors_available,
    is_sentencepiece_available,
    is_soundfile_availble,
    is_tokenizers_available,
    is_pytesseract_available,
    is_vision_available,
    is_g2p_en_available,
    is_levenshtein_available,
    is_nltk_available,
    is_ftfy_available
)
from .generic import strtobool

if is_pytest_available():
    from _pytest.doctest import (
        Module,
        _get_checker,
        _get_continue_on_failure,
        _get_runner,
        _is_mocked,
        _patch_unwrap_mock_aware,
        get_optionflags,
        import_path,
    )
    from _pytest.config import create_terminal_writer
    from _pytest.outcomes import skip
    from pytest import DoctestItem
else:
    Module = object
    DoctestItem = object

if is_mindspore_available():
    from mindspore import ops


DUMMY_UNKNOWN_IDENTIFIER = "julien-c/dummy-unknown"
SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"

def is_pipeline_test(test_case):
    """
    Decorator marking a test as a pipeline test. If RUN_PIPELINE_TESTS is set to a falsy value, those tests will be
    skipped.
    """
    if not _run_pipeline_tests:
        return unittest.skip("test is pipeline test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pipeline_test()(test_case)

def parse_flag_from_env(key, default=False):
    """
    Parses a flag value from the environment variable.
    
    Args:
        key (str): The name of the environment variable to retrieve the flag value from.
        default (bool, optional): The default flag value to return if the environment variable is not set. Defaults to False.
    
    Returns:
        bool: The parsed flag value. Returns the default value if the environment variable is not set or if its value cannot be parsed.
    
    Raises:
        ValueError: If the environment variable value is set but cannot be parsed as a boolean ('yes' or 'no').
    
    Note:
        The flag value is retrieved from the environment variable specified by `key`. If the environment variable is not set, the default value is returned. If the environment variable value is set, it is
parsed as a boolean using the `strtobool` function from the `distutils.util` module. If the parsing fails, a `ValueError` is raised with a descriptive error message indicating that the value must be either
'yes' or 'no'.
    """
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError as exc:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.") from exc
    return _value

_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_too_slow_tests = parse_flag_from_env("RUN_TOO_SLOW", default=False)
_run_pipeline_tests = parse_flag_from_env("RUN_PIPELINE_TESTS", default=True)

def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)

def tooslow(test_case):
    """
    Decorator marking a test as too slow.

    Slow tests are skipped while they're in the process of being fixed. No test should stay tagged as "tooslow" as
    these will not be tested by the CI.

    """
    return unittest.skipUnless(_run_too_slow_tests, "test is too slow")(test_case)

def parse_int_from_env(key, default=None):
    """Parses an integer value from the specified environment variable.
    
    Args:
        key (str): The name of the environment variable to retrieve the integer value from.
        default (int, optional): The default integer value to return if the environment variable is not set or cannot be converted to an integer. Defaults to None.
    
    Returns:
        int or None: The integer value parsed from the environment variable or the default value if provided. Returns None if the environment variable is not set and no default value is specified.
    
    Raises:
        ValueError: If the value retrieved from the environment variable cannot be converted to an integer.
    """
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError as exc:
            raise ValueError(f"If set, {key} must be a int.") from exc
    return _value


def require_ftfy(test_case):
    """
    Decorator marking a test that requires ftfy. These tests are skipped when ftfy isn't installed.
    """
    return unittest.skipUnless(is_ftfy_available(), "test requires ftfy")(test_case)


def require_levenshtein(test_case):
    """
    Decorator marking a test that requires Levenshtein.

    These tests are skipped when Levenshtein isn't installed.

    """
    return unittest.skipUnless(is_levenshtein_available(), "test requires Levenshtein")(test_case)


def require_nltk(test_case):
    """
    Decorator marking a test that requires NLTK.

    These tests are skipped when NLTK isn't installed.

    """
    return unittest.skipUnless(is_nltk_available(), "test requires NLTK")(test_case)


def require_vision(test_case):
    """
    Decorator marking a test that requires the vision dependencies. These tests are skipped when torchaudio isn't
    installed.
    """
    return unittest.skipUnless(is_vision_available(), "test requires vision")(test_case)

def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers. These tests are skipped when 🤗 Tokenizers isn't installed.
    """
    return unittest.skipUnless(is_tokenizers_available(), "test requires tokenizers")(test_case)

def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_sentencepiece_available(), "test requires SentencePiece")(test_case)

def require_mindspore(test_case):
    """
    Decorator marking a test that requires MindSpore.

    These tests are skipped when MindSpore isn't installed.

    """
    return unittest.skipUnless(is_mindspore_available(), "test requires MindSpore")(test_case)

def require_mindspore_gpu(test_case):
    """Decorator marking a test that requires CUDA and MindSpore."""
    return unittest.skipUnless(mindspore.get_context('device_target') == "GPU", "test requires CUDA")(test_case)

def require_mindspore_npu(test_case):
    """Decorator marking a test that requires CANN and MindSpore."""
    return unittest.skipUnless(mindspore.get_context('device_target') == "Ascend", "test requires CANN")(test_case)


def require_librosa(test_case):
    """
    Decorator marking a test that requires librosa
    """
    return unittest.skipUnless(is_librosa_available(), "test requires librosa")(test_case)

def require_essentia(test_case):
    """
    Decorator marking a test that requires essentia
    """
    return unittest.skipUnless(is_essentia_available(), "test requires essentia")(test_case)

def require_pretty_midi(test_case):
    """
    Decorator marking a test that requires pretty_midi
    """
    return unittest.skipUnless(is_pretty_midi_available(), "test requires pretty_midi")(test_case)

def require_scipy(test_case):
    """
    Decorator marking a test that requires Scipy. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_scipy_available(), "test requires Scipy")(test_case)

def require_pyctcdecode(test_case):
    """
    Decorator marking a test that requires pyctcdecode
    """
    return unittest.skipUnless(is_pyctcdecode_available(), "test requires pyctcdecode")(test_case)

def require_safetensors(test_case):
    """
    Decorator marking a test that requires safetensors. These tests are skipped when safetensors isn't installed.
    """
    return unittest.skipUnless(is_safetensors_available(), "test requires safetensors")(test_case)

def require_pytesseract(test_case):
    """
    Decorator marking a test that requires pytesseract
    """
    return unittest.skipUnless(is_pytesseract_available(), "test requires pytesseract")(test_case)

def require_g2p_en(test_case):
    """
    Decorator marking a test that requires pytesseract
    """
    return unittest.skipUnless(is_g2p_en_available(), "test requires g2p-en")(test_case)


def cmd_exists(cmd):
    """
    Check if a command exists in the system PATH.
    
    Args:
        cmd (str): The name of the command to check for existence in the system PATH.
    
    Returns:
        None: Returns None if the command exists in the system PATH, otherwise returns False.
    
    Raises:
        None.
    """
    return shutil.which(cmd) is not None
#
# Helper functions for dealing with testing text outputs
# The original code came from:
# https://github.com/fastai/fastai/blob/master/tests/utils/text.py


# When any function contains print() calls that get overwritten, like progress bars,
# a special care needs to be applied, since under pytest -s captured output (capsys
# or contextlib.redirect_stdout) contains any temporary printed strings, followed by
# \r's. This helper function ensures that the buffer will contain the same output
# with and without -s in pytest, by turning:
# foo bar\r tar mar\r final message
# into:
# final message
# it can handle a single string or a multiline buffer
def apply_print_resets(buf):
    """
    Apply print resets by removing any characters before the last carriage return in the given buffer.
    
    Args:
        buf (str): The input buffer containing text data.
        
    Returns:
        None. The function modifies the buffer in place.
    
    Raises:
        None.
    """
    return re.sub(r"^.*\r", "", buf, 0, re.M)


def assert_screenout(out, what):
    """
    This function asserts the presence of a specified string within the provided output.
    
    Args:
        out (str): The output string to be checked for the presence of the specified string.
        what (str): The string to be searched for within the output.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        AssertionError: If the specified string 'what' is not found within the output string 'out'.
    """
    out_pr = apply_print_resets(out).lower()
    match_str = out_pr.find(what.lower())
    assert match_str != -1, f"expecting to find {what} in output: f{out_pr}"


class CaptureStd:
    """
    Context manager to capture:

        - stdout: replay it, clean it up and make it available via `obj.out`
        - stderr: replay it and make it available via `obj.err`

    Args:
        out (`bool`, *optional*, defaults to `True`): Whether to capture stdout or not.
        err (`bool`, *optional*, defaults to `True`): Whether to capture stderr or not.
        replay (`bool`, *optional*, defaults to `True`): Whether to replay or not.
            By default each captured stream gets replayed back on context's exit, so that one can see what the test was
            doing. If this is a not wanted behavior and the captured data shouldn't be replayed, pass `replay=False` to
            disable this feature.

    Examples:

    ```python
    # to capture stdout only with auto-replay
    with CaptureStdout() as cs:
        print("Secret message")
    assert "message" in cs.out

    # to capture stderr only with auto-replay
    import sys

    with CaptureStderr() as cs:
        print("Warning: ", file=sys.stderr)
    assert "Warning" in cs.err

    # to capture both streams with auto-replay
    with CaptureStd() as cs:
        print("Secret message")
        print("Warning: ", file=sys.stderr)
    assert "message" in cs.out
    assert "Warning" in cs.err

    # to capture just one of the streams, and not the other, with auto-replay
    with CaptureStd(err=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    # but best use the stream-specific subclasses

    # to capture without auto-replay
    with CaptureStd(replay=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    ```"""
    def __init__(self, out=True, err=True, replay=True):
        """Initialize a CaptureStd object.
        
        Args:
            self (CaptureStd): The instance of the CaptureStd class.
            out (bool): Flag indicating whether to capture stdout. Default is True.
            err (bool): Flag indicating whether to capture stderr. Default is True.
            replay (bool): Flag indicating whether to replay captured output. Default is True.
        
        Returns:
            None
        
        Raises:
            None
        
        This method initializes a CaptureStd object with the given parameters. The 'out' parameter determines whether to capture stdout, while the 'err' parameter determines whether to capture stderr. By
default, both 'out' and 'err' are set to True. If 'out' is True, a StringIO object is created to capture stdout. If 'out' is False, stdout is not captured and the 'out' attribute is set to 'not capturing
stdout'. The same logic applies to 'err' and stderr.
        
        The 'replay' parameter determines whether the captured output should be replayed. By default, 'replay' is set to True.
        
        Note: If 'out' or 'err' is set to True, but the CaptureStd context is not finished yet (i.e., __exit__ is not called), an error message is set to the corresponding attribute indicating that the context
was called too early.
        """
        self.replay = replay

        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

    def __enter__(self):
        """
        The '__enter__' method is used as a context manager to redirect the standard output and standard error streams to the provided buffers.
        
        Args:
            self: An instance of the 'CaptureStd' class.
        
        Returns:
            None. This method does not return any value explicitly.
        
        Raises:
            None.
        """
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self

    def __exit__(self, *exc):
        """
        This method __exit__ is called automatically when exiting a 'with' block that uses the CaptureStd context manager.
        
        Args:
            self: An instance of the CaptureStd class that represents the current context manager. It is used to access the attributes and buffers within the context manager.
        
        Returns:
            None. The method does not explicitly return a value.
        
        Raises:
            This method does not raise any exceptions explicitly. However, exceptions may be raised if there are errors during the execution of the code within the method.
        """
        if self.out_buf:
            sys.stdout = self.out_old
            captured = self.out_buf.getvalue()
            if self.replay:
                sys.stdout.write(captured)
            self.out = apply_print_resets(captured)

        if self.err_buf:
            sys.stderr = self.err_old
            captured = self.err_buf.getvalue()
            if self.replay:
                sys.stderr.write(captured)
            self.err = captured

    def __repr__(self):
        """
        Returns a string representation of the CaptureStd object.
        
        Args:
            self: The instance of the CaptureStd class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        Description:
            The __repr__ method is called when the repr() function is used on an instance of the CaptureStd class. It generates a string representation of the object, which includes the captured stdout and
stderr outputs, if any. The generated string representation is returned by the method.
        
            This method checks if the 'out_buf' attribute of the CaptureStd object is not empty. If it is not empty, the captured stdout output is added to the message string. Similarly, if the 'err_buf'
attribute is not empty, the captured stderr output is added to the message string. The final message string is then returned by the method.
        
            Note that the stdout and stderr outputs are represented as 'stdout: <output>' and 'stderr: <output>' respectively in the message string.
        
        Example Usage:
            capture = CaptureStd()
            capture.capture_stdout('Hello, world!')
            capture.capture_stderr('Oops, an error occurred.')
            repr_str = repr(capture)
            print(repr_str)
            # Output: "stdout: Hello, world!\nstderr: Oops, an error occurred.\n"
        """
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


# in tests it's the best to capture only the stream that's wanted, otherwise
# it's easy to miss things, so unless you need to capture both streams, use the
# subclasses below (less typing). Or alternatively, configure `CaptureStd` to
# disable the stream you don't need to test.


class CaptureStdout(CaptureStd):
    """Same as CaptureStd but captures only stdout"""
    def __init__(self, replay=True):
        """
        Initializes an instance of the CaptureStdout class.
        
        Args:
            self: The instance of the class.
            replay (bool): A boolean flag indicating whether the captured output should be replayed. 
                           Defaults to True. If set to True, the captured output will be replayed.
                           If set to False, the captured output will not be replayed.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """Same as CaptureStd but captures only stderr"""
    def __init__(self, replay=True):
        """
        Initializes an instance of the CaptureStderr class.
        
        Args:
            self (CaptureStderr): The current object.
            replay (bool): Indicates whether to replay the captured stderr output. Default is True.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
     Context manager to capture `logging` streams

     Args:
         logger: 'logging` logger object

     Returns:
         The captured output is available via `self.out`

     Example:

     ```python
     >>> from transformers import logging
     >>> from transformers.testing_utils import CaptureLogger

     >>> msg = "Testing 1, 2, 3"
     >>> logging.set_verbosity_info()
     >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart")
     >>> with CaptureLogger(logger) as cl:
     ...     logger.info(msg)
     >>> assert cl.out, msg + "\n"
     ```
     """
    def __init__(self, logger):
        """
        Initializes a new instance of the CaptureLogger class.
        
        Args:
            self: The instance of the class.
            logger: An object representing the logger to be used for capturing logs. It should be an instance of a logger class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        """
        This method is an implementation of the context manager protocol for the CaptureLogger class.
        
        Args:
            self: An instance of the CaptureLogger class. It represents the current object that the method is being called upon.
        
        Returns:
            None. The method does not explicitly return any value, but it adds a handler to the logger associated with the CaptureLogger instance.
        
        Raises:
            This method does not raise any exceptions under normal circumstances. However, potential exceptions could be raised if there are issues with adding the handler to the logger, such as improper
configuration of the logging system.
        """
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        """
        This method __exit__ is called automatically when exiting a 'with' block in the CaptureLogger class.
        
        Args:
            self (CaptureLogger): An instance of the CaptureLogger class. It is used to access the logger and the captured output.
            
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not raise any exceptions explicitly. However, exceptions may be raised internally if there are issues with removing the handler or getting the captured output.
        """
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        """
        Return a string representation of the CaptureLogger object.
        
        Args:
            self (CaptureLogger): The instance of the CaptureLogger class.
        
        Returns:
            None: This method does not explicitly return any value, as it returns None.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return f"captured: {self.out}\n"


@contextlib.contextmanager
def LoggingLevel(level):
    """
    This is a context manager to temporarily change transformers modules logging level to the desired value and have it
    restored to the original setting at the end of the scope.

    Example:

    ```python
    with LoggingLevel(logging.INFO):
        AutoModel.from_pretrained("gpt2")  # calls logger.info() several times
    ```
    """
    orig_level = mindnlp_logging.get_verbosity()
    try:
        mindnlp_logging.set_verbosity(level)
        yield
    finally:
        mindnlp_logging.set_verbosity(orig_level)


@contextlib.contextmanager
# adapted from https://stackoverflow.com/a/64789046/9201239
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    Temporary add given path to `sys.path`.

    Usage :

    ```python
    with ExtendSysPath("/path/to/dir"):
        mymodule = importlib.import_module("mymodule")
    ```
    """
    path = os.fspath(path)
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


class TestCasePlus(unittest.TestCase):
    """
    This class extends *unittest.TestCase* with additional features.

    Feature 1: A set of fully resolved important file and dir path accessors.

    In tests often we need to know where things are relative to the current test file, and it's not trivial since the
    test could be invoked from more than one directory or could reside in sub-directories with different depths. This
    class solves this problem by sorting out all the basic paths and provides easy accessors to them:

    - `pathlib` objects (all fully resolved):

       - `test_file_path` - the current test file path (=`__file__`)
       - `test_file_dir` - the directory containing the current test file
       - `tests_dir` - the directory of the `tests` test suite
       - `examples_dir` - the directory of the `examples` test suite
       - `repo_root_dir` - the directory of the repository
       - `src_dir` - the directory of `src` (i.e. where the `transformers` sub-dir resides)

    - stringified paths---same as above but these return paths as strings, rather than `pathlib` objects:

       - `test_file_path_str`
       - `test_file_dir_str`
       - `tests_dir_str`
       - `examples_dir_str`
       - `repo_root_dir_str`
       - `src_dir_str`

    Feature 2: Flexible auto-removable temporary dirs which are guaranteed to get removed at the end of test.

    1. Create a unique temporary dir:

    ```python
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
    ```

    `tmp_dir` will contain the path to the created temporary dir. It will be automatically removed at the end of the
    test.


    2. Create a temporary dir of my choice, ensure it's empty before the test starts and don't
    empty it after the test.

    ```python
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
    ```

    This is useful for debug when you want to monitor a specific directory and want to make sure the previous tests
    didn't leave any data in there.

    3. You can override the first two options by directly overriding the `before` and `after` args, leading to the
        following behavior:

    `before=True`: the temporary dir will always be cleared at the beginning of the test.

    `before=False`: if the temporary dir already existed, any existing files will remain there.

    `after=True`: the temporary dir will always be deleted at the end of the test.

    `after=False`: the temporary dir will always be left intact at the end of the test.

    Note 1: In order to run the equivalent of `rm -r` safely, only subdirs of the project repository checkout are
    allowed if an explicit `tmp_dir` is used, so that by mistake no `/tmp` or similar important part of the filesystem
    will get nuked. i.e. please always pass paths that start with `./`

    Note 2: Each test can register multiple temporary dirs and they all will get auto-removed, unless requested
    otherwise.

    Feature 3: Get a copy of the `os.environ` object that sets up `PYTHONPATH` specific to the current test suite. This
    is useful for invoking external programs from the test suite - e.g. distributed training.


    ```python
    def test_whatever(self):
        env = self.get_env()
    ```"""
    def setUp(self):
        """
        Set up the necessary environment for the TestCasePlus class.
        
        Args:
            self: The instance of the TestCasePlus class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If the root directory of the repository cannot be determined from the test file path.
        
        Description:
        This method is called before each test case to set up the required environment for the TestCasePlus class. It initializes various directories and paths based on the current test file's location. The
method performs the following steps:
        
        1. Sets up a list to keep track of temporary directories that need to be cleaned up later.
        2. Retrieves the path of the test file using the inspect module.
        3. Resolves the absolute path of the test file.
        4. Determines the parent directory of the test file.
        5. Checks if there are 'src' and 'tests' directories in any of the parent directories up to three levels above the test file.
        6. If such directories are found, the loop breaks and the repository root directory is set as the temporary directory.
        7. If no valid temporary directory is found, a ValueError is raised indicating that the root directory of the repository could not be determined.
        8. Sets the paths for the 'tests', 'examples', and 'src' directories within the repository root directory.
        
        Note:
        This method assumes a specific directory structure for the repository, where 'src' and 'tests' directories exist at an appropriate level above the test file.
        
        Example usage:
            test_case = TestCasePlus()
            test_case.setUp()
        """
        # get_auto_remove_tmp_dir feature:
        self.teardown_tmp_dirs = []

        # figure out the resolved paths for repo_root, tests, examples, etc.
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self._test_file_dir = path.parents[0]
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            if (tmp_dir / "src").is_dir() and (tmp_dir / "tests").is_dir():
                break
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        self._tests_dir = self._repo_root_dir / "tests"
        self._examples_dir = self._repo_root_dir / "examples"
        self._src_dir = self._repo_root_dir / "src"

    @property
    def test_file_path(self):
        """
        Returns the test file path.
        
        Args:
            self: An instance of the TestCasePlus class.
        
        Returns:
            None. The method does not return any value.
        
        Raises:
            This method does not raise any exceptions.
        """
        return self._test_file_path

    @property
    def test_file_path_str(self):
        """
        Method to retrieve the string representation of the test file path.
        
        Args:
            self: Instance of the TestCasePlus class.
                - Type: object
                - Purpose: Represents the current instance of the class.
                - Restrictions: None
        
        Returns:
            The method returns a string representing the test file path.
                - Type: str
                - Purpose: Provides the string representation of the test file path.
        
        Raises:
            No exceptions are raised by this method.
        """
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        """
        This method retrieves the directory path where test files are located.
        
        Args:
            self: An instance of the TestCasePlus class.
                This parameter refers to the current instance of the TestCasePlus class.
        
        Returns:
            None. The method does not return any value explicitly but retrieves and returns the test file directory path.
        
        Raises:
            This method does not raise any exceptions.
        """
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        """
        Method test_file_dir_str in the class TestCasePlus.
        
        Args:
            self: Represents the instance of the class. No additional parameters are required.
        
        Returns:
            str: A string representation of the _test_file_dir attribute of the instance.
        
        Raises:
            None.
        """
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        """
        Method: tests_dir
        
        Description:
        Returns the tests directory path used by the TestCasePlus class.
        
        Args:
        - self (object): The instance of the TestCasePlus class.
        
        Returns:
        - None: This method does not return any value explicitly.
        
        Raises:
        - None
        """
        return self._tests_dir

    @property
    def tests_dir_str(self):
        """
        Returns the tests directory as a string.
        
        Args:
            self: An instance of the TestCasePlus class.
        
        Returns:
            str: The tests directory path converted to a string.
        
        Raises:
            None.
        
        This method returns the tests directory path as a string. The tests directory is obtained from the '_tests_dir' attribute of the TestCasePlus class. The returned string represents the absolute path of
the tests directory.
        
        Example usage:
            >>> test_case = TestCasePlus()
            >>> test_case.tests_dir_str()
            '/path/to/tests/directory'
        """
        return str(self._tests_dir)

    @property
    def examples_dir(self):
        """
        Method to get the examples directory path.
        
        Args:
            self: The instance of the class.
            
        Returns:
            None. The method returns the examples directory path.
        
        Raises:
            This method does not raise any exceptions.
        """
        return self._examples_dir

    @property
    def examples_dir_str(self):
        """
        Method examples_dir_str in the class TestCasePlus returns the string representation of the _examples_dir attribute.
        
        Args:
            self: An instance of the TestCasePlus class.
                Purpose: Represents the current instance of the class.
                Restrictions: None.
        
        Returns:
            str: A string representation of the _examples_dir attribute.
                Purpose: Provides a human-readable string representation of the _examples_dir attribute.
        
        Raises:
            None.
        """
        return str(self._examples_dir)

    @property
    def repo_root_dir(self):
        """
        Method to retrieve the root directory of the repository.
        
        Args:
            self (TestCasePlus): The instance of the TestCasePlus class.
                This parameter is required to access the instance attributes and methods.
        
        Returns:
            None. The method returns the value of the '_repo_root_dir' attribute of the instance.
        
        Raises:
            This method does not raise any exceptions.
        """
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        """
        Method to retrieve the repository root directory as a string.
        
        Args:
            self: The instance of the class TestCasePlus.
                This parameter is automatically passed and refers to the instance itself.
        
        Returns:
            str: A string representing the repository root directory.
                This method returns the repository root directory as a string.
        
        Raises:
            None.
        """
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        """
        Returns the source directory path for the TestCasePlus class.
        
        Args:
            self (TestCasePlus): An instance of the TestCasePlus class.
        
        Returns:
            None: The method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        return self._src_dir

    @property
    def src_dir_str(self):
        """
        Method to retrieve the source directory path as a string representation.
        
        Args:
            self: An instance of the TestCasePlus class.
                This parameter refers to the current object instance.
                It is used to access the source directory path stored in the _src_dir attribute.
        
        Returns:
            None
            This method returns the source directory path as a string. If the source directory path does not exist or is empty, None is returned.
        
        Raises:
            None
            This method does not raise any exceptions.
        """
        return str(self._src_dir)

    def get_env(self):
        """
        Return a copy of the `os.environ` object that sets up `PYTHONPATH` correctly, depending on the test suite it's
        invoked from. This is useful for invoking external programs from the test suite - e.g. distributed training.

        It always inserts `./src` first, then `./tests` or `./examples` depending on the test suite type and finally
        the preset `PYTHONPATH` if any (all full resolved paths).

        """
        env = os.environ.copy()
        paths = [self.src_dir_str]
        if "/examples" in self.test_file_dir_str:
            paths.append(self.examples_dir_str)
        else:
            paths.append(self.tests_dir_str)
        paths.append(env.get("PYTHONPATH", ""))

        env["PYTHONPATH"] = ":".join(paths)
        return env

    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
        """
        Args:
            tmp_dir (`string`, *optional*):
                if `None`:

                   - a unique temporary path will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=True` if `after` is `None`
                else:

                   - `tmp_dir` will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=False` if `after` is `None`
            before (`bool`, *optional*):
                If `True` and the `tmp_dir` already exists, make sure to empty it right away if `False` and the
                `tmp_dir` already exists, any existing files will remain there.
            after (`bool`, *optional*):
                If `True`, delete the `tmp_dir` at the end of the test if `False`, leave the `tmp_dir` and its contents
                intact at the end of the test.

        Returns:
            tmp_dir(`string`): either the same value as passed via *tmp_dir* or the path to the auto-selected tmp dir
        """
        if tmp_dir is not None:
            # defining the most likely desired behavior for when a custom path is provided.
            # this most likely indicates the debug mode where we want an easily locatable dir that:
            # 1. gets cleared out before the test (if it already exists)
            # 2. is left intact after the test
            if before is None:
                before = True
            if after is None:
                after = False

            # using provided path
            path = Path(tmp_dir).resolve()

            # to avoid nuking parts of the filesystem, only relative paths are allowed
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`"
                )

            # ensure the dir is empty to start with
            if before is True and path.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            path.mkdir(parents=True, exist_ok=True)

        else:
            # defining the most likely desired behavior for when a unique tmp path is auto generated
            # (not a debug mode), here we require a unique tmp dir that:
            # 1. is empty before the test (it will be empty in this situation anyway)
            # 2. gets fully removed after the test
            if before is None:
                before = True
            if after is None:
                after = True

            # using unique tmp dir (always empty, regardless of `before`)
            tmp_dir = tempfile.mkdtemp()

        if after is True:
            # register for deletion
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir

    def python_one_liner_max_rss(self, one_liner_str):
        """
        Runs the passed python one liner (just the code) and returns how much max cpu memory was used to run the
        program.

        Args:
            one_liner_str (`string`):
                a python one liner code that gets passed to `python -c`

        Returns:
            max cpu memory bytes used to run the program. This value is likely to vary slightly from run to run.

        Requirements:
            this helper needs `/usr/bin/time` to be installed (`apt install time`)

        Example:

        ```
        one_liner_str = 'from transformers import AutoModel; AutoModel.from_pretrained("t5-large")'
        max_rss = self.python_one_liner_max_rss(one_liner_str)
        ```
        """
        if not cmd_exists("/usr/bin/time"):
            raise ValueError("/usr/bin/time is required, install with `apt install time`")

        cmd = shlex.split(f"/usr/bin/time -f %M python -c '{one_liner_str}'")
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        # returned data is in KB so convert to bytes
        max_rss = int(cs.err.split("\n")[-2].replace("stderr: ", "")) * 1024
        return max_rss

    def tearDown(self):
        """
        Tears down the test case by cleaning up temporary directories.
        
        Args:
            self (TestCasePlus): The instance of the TestCasePlus class.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        # get_auto_remove_tmp_dir feature: remove registered temp dirs
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []


def mockenv(**kwargs):
    """
    this is a convenience wrapper, that allows this ::

    @mockenv(RUN_SLOW=True, USE_TF=False) def test_something():
        run_slow = os.getenv("RUN_SLOW", False) use_tf = os.getenv("USE_TF", False)

    """
    return mock.patch.dict(os.environ, kwargs)


# from https://stackoverflow.com/a/34333710/9201239
@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    Temporarily updates the `os.environ` dictionary in-place. Similar to mockenv

    The `os.environ` dictionary is updated in-place so that the modification is sure to work in all situations.

    Args:
      remove: Environment variables to remove.
      update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        for k in remove:
            env.pop(k, None)
        yield
    finally:
        env.update(update_after)
        for k in remove_after:
            env.pop(k)


# --- pytest conf functions --- #

# to avoid multiple invocation from tests/conftest.py and examples/conftest.py - make sure it's called only once
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="generate report files. The value of this option is used as a prefix to report names",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, ids):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:
    - tr: `terminalreporter` passed from `conftest.py`
    - ids: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should pytest do internal
    changes - also it calls default internal methods of terminalreporter which can be hijacked by various `pytest-`
    plugins and interfere.

    """
    if not ids:
        ids = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    dirs = f"reports/{ids}"
    Path(dirs).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dirs}/{k}.txt"
        for k in [
            "durations",
            "errors",
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }

    # custom durations report
    # note: there is no need to call pytest --durations=XX to get this separate report
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):
        # expecting that the reports were --tb=long (default) so we chop them off here to the last frame
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "FAILURES SHORT STACK")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # chop off the optional leading extra frames, leaving only the last one
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # note: not printing out any rep.sections to keep the report short

    # use ready-made report funcs, we are just hijacking the filehandle to log to a dedicated file each
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # note: some pytest plugins may interfere by hijacking the default `terminalreporter` (e.g.
    # pytest-instafail does that)

    # report failures with line/short/long styles
    config.option.tbstyle = "auto"  # full tb
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # short tb
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # one line per error
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # normal warnings
        tr.summary_warnings()  # final warnings

    tr.reportchars = "wPpsxXEf"  # emulate -rA (used in summary_passes() and short_test_summary())

    # Skip the `passes` report, as it starts to take more than 5 minutes, and sometimes it timeouts on CircleCI if it
    # takes > 10 minutes (as this part doesn't generate any output on the terminal).
    # (also, it seems there is no useful information in this report, and we rarely need to read it)
    # with open(report_files["passes"], "w") as f:
    #     tr._tw = create_terminal_writer(config, f)
    #     tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # restore:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


# --- distributed testing functions --- #

# adapted from https://stackoverflow.com/a/59041913/9201239
class _RunOutput:

    """
    Represents the output of a command execution, including the return code, standard output, and standard error.
    
    Attributes:
        returncode (int): The return code of the executed command.
        stdout (str): The standard output captured from the command execution.
        stderr (str): The standard error captured from the command execution.
    """
    def __init__(self, returncode, stdout, stderr):
        """
        __init__(self, returncode, stdout, stderr)
        
        Initializes the _RunOutput class instance with the provided return code, standard output, and standard error.
        
        Args:
            self (_RunOutput): The instance of the _RunOutput class.
            returncode (int): The return code from the executed command.
            stdout (str): The standard output generated by the executed command.
            stderr (str): The standard error generated by the executed command.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    """
    Docstring for _read_stream function:
    
    Args:
        stream (stream): The input stream from which the function reads data.
        callback (function): The callback function to be executed for each line read from the stream.
    
    Returns:
        None. The function does not return any value.
    
    Raises:
        No specific exceptions are raised by this function.
    """
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    """
    This function runs a subprocess and captures its standard output and error streams.
    
    Args:
    - cmd (List[str]): A list of command and arguments to be executed.
    - env (Optional[Dict[str, str]]): A dictionary of environment variables to be used for the subprocess.
    - stdin (Optional[asyncio.subprocess.StreamReader]): A stream representing the standard input for the subprocess.
    - timeout (Optional[float]): The maximum time in seconds to wait for the subprocess to complete.
    - quiet (bool): If True, suppresses the output of the subprocess.
    - echo (bool): If True, prints the command being executed.
    
    Returns:
    _RunOutput: An object containing the return code of the subprocess, its standard output, and standard error.
    
    Raises:
    - asyncio.TimeoutError: If the subprocess execution exceeds the specified timeout.
    - OSError: If an OS-related error occurs during the subprocess execution.
    - ValueError: If the provided command is invalid or the arguments are of the wrong type.
    """
    if echo:
        print("\nRunning: ", " ".join(cmd))

    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # note: there is a warning for a possible deadlock when using `wait` with huge amounts of data in the pipe
    # https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process.wait
    #
    # If it starts hanging, will need to switch to the following code. The problem is that no data
    # will be seen until it's done and if it hangs for example there will be no debug info.
    # out, err = await p.communicate()
    # return _RunOutput(p.returncode, out, err)

    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda l: tee(l, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda l: tee(l, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    """
    Args:
        cmd (List[str]): A list of strings representing the command and its arguments to be executed.
        env (Optional[Dict[str, str]]): A dictionary of environment variables to be passed to the subprocess.
        stdin (Optional[Union[str, bytes]]): The input to be passed to the subprocess.
        timeout (int): The maximum time in seconds to wait for the subprocess to complete.
        quiet (bool): If True, suppresses output from the subprocess.
        echo (bool): If True, prints the subprocess output to the console.
    
    Returns:
        _RunOutput: An object containing the output of the executed subprocess, including stdout, stderr, and returncode.
    
    Raises:
        RuntimeError: If the subprocess fails with a non-zero return code or produces no output.
    """
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # check that the subprocess actually did run and produced some output, should the test rely on
    # the remote side to do the testing
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result


def pytest_xdist_worker_id():
    """
    Returns an int value of worker's numerical id under `pytest-xdist`'s concurrent workers `pytest -n N` regime, or 0
    if `-n 1` or `pytest-xdist` isn't being used.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker = re.sub(r"^gw", "", worker, 0, re.M)
    return int(worker)


def get_torch_dist_unique_port():
    """
    Returns a port number that can be fed to `torch.distributed.launch`'s `--master_port` argument.

    Under `pytest-xdist` it adds a delta number based on a worker id so that concurrent tests don't try to use the same
    port at once.
    """
    port = 29500
    uniq_delta = pytest_xdist_worker_id()
    return port + uniq_delta


def nested_simplify(obj, decimals=3):
    """
    Simplifies an object by rounding float numbers, and downcasting tensors/numpy arrays to get simple equality test
    within tests.
    """
    if isinstance(obj, list):
        return [nested_simplify(item, decimals) for item in obj]
    if isinstance(obj, tuple):
        return tuple(nested_simplify(item, decimals) for item in obj)
    if isinstance(obj, np.ndarray):
        return nested_simplify(obj.tolist())
    if isinstance(obj, Mapping):
        return {nested_simplify(k, decimals): nested_simplify(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (str, int, np.int64)):
        return obj
    if obj is None:
        return obj
    if is_mindspore_available() and ops.is_tensor(obj):
        return nested_simplify(obj.numpy().tolist())
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, (np.int32, np.float32)):
        return nested_simplify(obj.item(), decimals)
    raise RuntimeError(f"Not supported: {type(obj)}")


def to_2tuple(x):
    """
    Converts the input value to a 2-tuple.
    
    Args:
        x: The value to be converted. It can be of any type.
    
    Returns:
        A 2-tuple with the input value. If the input value is already an iterable, it is returned as is.
        Otherwise, a 2-tuple is created with the input value repeated twice.
    
    Raises:
        None.
    
    """
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# These utils relate to ensuring the right error message is received when running scripts
class SubprocessCallException(Exception):
    """SubprocessCallException"""
def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occured while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e
    return None

class RequestCounter:
    """
    Helper class that will count all requests made online.

    Might not be robust if urllib3 changes its logging format but should be good enough for us.

    Usage:
    ```py
    with RequestCounter() as counter:
        _ = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    assert counter["GET"] == 0
    assert counter["HEAD"] == 1
    assert counter.total_calls == 1
    ```
    """
    def __enter__(self):
        """
        __enter__
        
        Args:
            self: The instance of the RequestCounter class.
        
        Returns:
            None. This method does not explicitly return a value.
        
        Raises:
            No specific exceptions are raised within this method.
        """
        self._counter = defaultdict(int)
        self.patcher = patch.object(urllib3.connectionpool.log, "debug", wraps=urllib3.connectionpool.log.debug)
        self.mock = self.patcher.start()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """
        This method '__exit__' in the class 'RequestCounter' is called upon exiting a context manager. It updates the request counters based on the logged HTTP methods.
        
        Args:
        - self: An instance of the 'RequestCounter' class. It represents the current instance of the class.
        
        Returns:
        - None: This method does not return any value.
        
        Raises:
        This method does not explicitly raise any exceptions.
        """
        for call in self.mock.call_args_list:
            log = call.args[0] % call.args[1:]
            for method in ("HEAD", "GET", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"):
                if method in log:
                    self._counter[method] += 1
                    break
        self.patcher.stop()

    def __getitem__(self, key: str) -> int:
        """
        Retrieve the count associated with the specified key from the RequestCounter.
        
        Args:
            self (RequestCounter): An instance of the RequestCounter class.
            key (str): The key for which the count needs to be retrieved. It should be a string representing the identifier of the request.
        
        Returns:
            int: The count associated with the specified key. This count indicates the number of times the request identified by the key has been made.
        
        Raises:
            KeyError: If the specified key does not exist in the RequestCounter, a KeyError is raised indicating that the count for the key cannot be retrieved.
        """
        return self._counter[key]

    @property
    def total_calls(self) -> int:
        """ 
        Method to calculate the total number of calls made to the RequestCounter instance.
        
        Args:
            self (RequestCounter): The instance of the RequestCounter class.
                This parameter is automatically passed when calling the method.
            
        Returns:
            int: The total number of calls made to the RequestCounter instance.
                It is the sum of all the values stored in the internal counter.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return sum(self._counter.values())

def is_flaky(max_attempts: int = 5, wait_before_retry: Optional[float] = None, description: Optional[str] = None):
    """
    To decorate flaky tests. They will be retried on failures.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """
    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator


def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    To run a test in a subprocess. In particular, this can avoid (GPU) memory issue.

    Args:
        test_case (`unittest.TestCase`):
            The test that will run `target_func`.
        target_func (`Callable`):
            The function implementing the actual testing logic.
        inputs (`dict`, *optional*, defaults to `None`):
            The inputs that will be passed to `target_func` through an (input) queue.
        timeout (`int`, *optional*, defaults to `None`):
            The timeout (in seconds) that will be passed to the input and output queues. If not specified, the env.
            variable `PYTEST_TIMEOUT` will be checked. If still `None`, its value will be set to `600`.
    """
    if timeout is None:
        timeout = int(os.environ.get("PYTEST_TIMEOUT", 600))

    start_methohd = "spawn"
    ctx = multiprocessing.get_context(start_methohd)

    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    # We can't send `unittest.TestCase` to the child, otherwise we get issues regarding pickle.
    input_queue.put(inputs, timeout=timeout)

    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()
    # Kill the child process if we can't get outputs from it in time: otherwise, the hanging subprocess prevents
    # the test to exit properly.
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        process.terminate()
        test_case.fail(e)
    process.join(timeout=timeout)

    if results["error"] is not None:
        test_case.fail(f'{results["error"]}')


# The following contains utils to run the documentation tests without having to overwrite any files.

# The `preprocess_string` function adds `# doctest: +IGNORE_RESULT` markers on the fly anywhere a `load_dataset` call is
# made as a print would otherwise fail the corresonding line.

# To skip cuda tests, make sure to call `SKIP_CUDA_DOCTEST=1 pytest --doctest-modules <path_to_files_to_test>


def preprocess_string(string, skip_cuda_tests):
    """Prepare a docstring or a `.md` file to be run by doctest.

    The argument `string` would be the whole file content if it is a `.md` file. For a python file, it would be one of
    its docstring. In each case, it may contain multiple python code examples. If `skip_cuda_tests` is `True` and a
    cuda stuff is detective (with a heuristic), this method will return an empty string so no doctest will be run for
    `string`.
    """
    codeblock_pattern = r"(```(?:python|py)\s*\n\s*>>> )((?:.*?\n)*?.*?```)"
    codeblocks = re.split(re.compile(codeblock_pattern, flags=re.MULTILINE | re.DOTALL), string)
    is_cuda_found = False
    for i, codeblock in enumerate(codeblocks):
        if "load_dataset(" in codeblock and "# doctest: +IGNORE_RESULT" not in codeblock:
            codeblocks[i] = re.sub(r"(>>> .*load_dataset\(.*)", r"\1 # doctest: +IGNORE_RESULT", codeblock)
        if (
            (">>>" in codeblock or "..." in codeblock)
            and re.search(r"cuda|to\(0\)|device=0", codeblock)
            and skip_cuda_tests
        ):
            is_cuda_found = True
            break

    modified_string = ""
    if not is_cuda_found:
        modified_string = "".join(codeblocks)

    return modified_string


class HfDocTestParser(doctest.DocTestParser):
    """
    Overwrites the DocTestParser from doctest to properly parse the codeblocks that are formatted with black. This
    means that there are no extra lines at the end of our snippets. The `# doctest: +IGNORE_RESULT` marker is also
    added anywhere a `load_dataset` call is made as a print would otherwise fail the corresponding line.

    Tests involving cuda are skipped base on a naive pattern that should be updated if it is not enough.
    """
    # This regular expression is used to find doctest examples in a
    # string.  It defines three groups: `source` is the source code
    # (including leading indentation and prompts); `indent` is the
    # indentation of the first (PS1) line of the source code; and
    # `want` is the expected output (including leading indentation).
    # fmt: off
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
             (?![ ]*>>>)          # Not a line starting with PS1
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:(?!```).)*        # Match any character except '`' until a '```' is found (this is specific to HF because black removes the last line)
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:\n|$)  # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE
    )
    # fmt: on

    # !!!!!!!!!!! HF Specific !!!!!!!!!!!
    skip_cuda_tests: bool = bool(os.environ.get("SKIP_CUDA_DOCTEST", False))
    # !!!!!!!!!!! HF Specific !!!!!!!!!!!

    def parse(self, string, name="<string>"):
        """
        Overwrites the `parse` method to incorporate a skip for CUDA tests, and remove logs and dataset prints before
        calling `super().parse`
        """
        string = preprocess_string(string, self.skip_cuda_tests)
        return super().parse(string, name)


class HfDoctestModule(Module):
    """
    Overwrites the `DoctestModule` of the pytest package to make sure the HFDocTestParser is used when discovering
    tests.
    """
    def collect(self) -> Iterable[DoctestItem]:
        """
        Collects doctests from the specified module.
        
        Args:
            self (HfDoctestModule): The instance of the HfDoctestModule class.
        
        Returns:
            Iterable[DoctestItem]: A collection of doctests represented as DoctestItem objects.
        
        Raises:
            ImportError: If the module cannot be imported and the 'doctest_ignore_import_errors' configuration option is not set.
            Skip: If the 'doctest_ignore_import_errors' configuration option is set and the module cannot be imported.
        """
        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456 https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct line number is returned. This will be
                reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)

                if hasattr(obj, "__wrapped__"):
                    # Get the main obj in case of it being wrapped
                    obj = inspect.unwrap(obj)

                # Type ignored because this is a private function.
                return super()._find_lineno(  # type:ignore[misc]
                    obj,
                    source_lines,
                )

            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    # Type ignored because this is a private function.
                    super()._find(  # type:ignore[misc]
                        tests, obj, name, module, source_lines, globs, seen
                    )

        if self.path.name == "conftest.py":
            module = self.config.pluginmanager._importconftest(
                self.path,
                self.config.getoption("importmode"),
                rootpath=self.config.rootpath,
            )
        else:
            try:
                module = import_path(
                    self.path,
                    root=self.config.rootpath,
                    mode=self.config.getoption("importmode"),
                )
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    skip(f"unable to import module {self.path}")
                else:
                    raise

        # !!!!!!!!!!! HF Specific !!!!!!!!!!!
        finder = MockAwareDocTestFinder(parser=HfDocTestParser())
        # !!!!!!!!!!! HF Specific !!!!!!!!!!!
        optionflags = get_optionflags(self)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )
        for test in finder.find(module, module.__name__):
            if test.examples:  # skip empty doctests and cuda
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)


def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):
    """
    Executes a device-agnostic dispatch based on the given device and dispatch table.
    
    Args:
        device (str): The device for which the dispatch is performed.
        dispatch_table (Dict[str, Callable]): A dictionary containing the dispatch functions for different devices.
    
    Returns:
        None: Returns None if the dispatch function for the given device is None.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    if device not in dispatch_table:
        return dispatch_table["default"](*args, **kwargs)

    fn = dispatch_table[device]

    # Some device agnostic functions return values. Need to guard against `None`
    # instead at user level.
    if fn is None:
        return None
    return fn(*args, **kwargs)

def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    return tests_dir

def check_json_file_has_correct_format(file_path):
    '''
    Check if the provided JSON file has the correct format.
    
    Args:
        file_path (str): The path to the JSON file to be checked.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        AssertionError: If the JSON file does not have the correct format as per the specified conditions.
        FileNotFoundError: If the specified file_path does not exist.
        UnicodeDecodeError: If the file cannot be decoded using the specified encoding.
    '''
    with open(file_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) == 1:
            # length can only be 1 if dict is empty
            assert lines[0] == "{}"
        else:
            # otherwise make sure json has correct format (at least 3 lines)
            assert len(lines) >= 3
            # each key one line, ident should be 2, min length is 3
            assert lines[0].strip() == "{"
            for _ in lines[1:-1]:
                left_indent = len(lines[1]) - len(lines[1].lstrip())
                assert left_indent == 2
            assert lines[-1].strip() == "}"

_run_staging = parse_flag_from_env("MINDNLP_CO_STAGING", default=False)

def is_staging_test(test_case):
    """
    Decorator marking a test as a staging test.

    Those tests will run using the staging environment of huggingface.co instead of the real model hub.
    """
    if not _run_staging:
        return unittest.skip("test is staging test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_staging_test()(test_case)


def require_soundfile(test_case):
    """
    Decorator marking a test that requires soundfile

    These tests are skipped when soundfile isn't installed.

    """
    return unittest.skipUnless(is_soundfile_availble(), "test requires soundfile")(
        test_case
    )
