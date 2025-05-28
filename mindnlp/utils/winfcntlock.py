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
"""fcntl replacement for Windows."""
import msvcrt
import os
import errno


# fcntl-style operation flags (subset).
LOCK_SH = 0x01  # Shared lock (mapped to read lock)
LOCK_EX = 0x02  # Exclusive lock
LOCK_NB = 0x04  # Non-blocking
LOCK_UN = 0x08  # Unlock

__all__ = [
    "flock",
    "LOCK_SH",
    "LOCK_EX",
    "LOCK_NB",
    "LOCK_UN",
]


def _get_fd(fd_or_fileobj):
    """Return an OS-level file descriptor from int or file object."""
    if isinstance(fd_or_fileobj, int):
        return fd_or_fileobj
    if hasattr(fd_or_fileobj, "fileno"):
        return fd_or_fileobj.fileno()
    raise TypeError("flock: fd must be int or file object with fileno().")


_def_rlck = hasattr(msvcrt, "LK_RLCK")  # Not available on very old runtimes.


def flock(fd, operation):
    """A minimal replacement for *nix fcntl.flock* based on *msvcrt.locking*.

    Only the patterns used inside mindnlp are implemented (exclusive lock
    and unlock, optionally combined with LOCK_NB). Shared locks are
    mapped to the *read lock* variant where the runtime provides it, or
    to an exclusive lock otherwise.
    """
    fd_int = _get_fd(fd)

    if operation & LOCK_UN:
        # Release the (single-byte) lock.
        msvcrt.locking(fd_int, msvcrt.LK_UNLCK, 1)
        return

    non_block = bool(operation & LOCK_NB)

    # Determine requested lock mode.
    if operation & LOCK_EX:
        mode = msvcrt.LK_NBLCK if non_block else msvcrt.LK_LOCK
    elif operation & LOCK_SH:
        if _def_rlck:
            mode = msvcrt.LK_NBRLCK if non_block else msvcrt.LK_RLCK
        else:
            mode = msvcrt.LK_NBLCK if non_block else msvcrt.LK_LOCK
    else:
        raise ValueError("flock(): must specify LOCK_EX or LOCK_SH")

    try:
        msvcrt.locking(fd_int, mode, 1)
    except OSError as e:
        # Translate resource conflicts into BlockingIOError for parity
        # with Unix semantics when non-blocking.
        if non_block and e.errno in (errno.EACCES, errno.EAGAIN):
            raise BlockingIOError from e
        raise

# Provide aliases expected by download.py (if any).
flock.LOCK_EX = LOCK_EX  # type: ignore
flock.LOCK_SH = LOCK_SH  # type: ignore
flock.LOCK_NB = LOCK_NB  # type: ignore
flock.LOCK_UN = LOCK_UN  # type: ignore
