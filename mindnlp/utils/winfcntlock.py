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
import win32con # pylint: disable=import-error
import pywintypes # pylint: disable=import-error
import win32file # pylint: disable=import-error


LOCK_EX = win32con.LOCKFILE_EXCLUSIVE_LOCK
LOCK_SH = 0  # The default value
LOCK_NB = win32con.LOCKFILE_FAIL_IMMEDIATELY
LOCK_UN = 0x08
__overlapped = pywintypes.OVERLAPPED()

def flock(file, flags):
    hfile = win32file._get_osfhandle(file)
    win32file.LockFileEx(hfile, flags, 0, 0xffff0000, __overlapped)

def unlock(file):
    hfile = win32file._get_osfhandle(file)
    win32file.UnlockFileEx(hfile, 0, 0xffff0000, __overlapped)
