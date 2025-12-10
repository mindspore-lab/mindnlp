# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multiple)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""
# pylint: disable=E0401
import logging
import os
from pathlib import Path
import ctypes as ct

from mindspore import context

from bitsandbytes.consts import (
    DYNAMIC_LIBRARY_SUFFIX,
    PACKAGE_DIR,
)
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs

logger = logging.getLogger(__name__)


class BNBNativeLibrary:
    _lib: ct.CDLL
    compiled_with_cuda = False

    def __init__(self, lib: ct.CDLL):
        self._lib = lib

    def __getattr__(self, item):
        return getattr(self._lib, item)


class CudaBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p
        lib.get_cusparse.restype = ct.c_void_p
        lib.cget_managed_ptr.restype = ct.c_void_p


def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    """
    Get the disk path to the CUDA BNB native library specified by the
    given CUDA specs, taking into account the `BNB_CUDA_VERSION` override environment variable.

    The library is not guaranteed to exist at the returned path.
    """
    library_name = f"libbitsandbytes_cuda{cuda_specs.cuda_version_string}"
    library_name = f"{library_name}{DYNAMIC_LIBRARY_SUFFIX}"

    override_value = os.environ.get("BNB_CUDA_VERSION")
    if override_value:
        library_name_stem, _, library_name_ext = library_name.rpartition(".")
        # `library_name_stem` will now be e.g. `libQ4M_cuda118`;
        # let's remove any trailing numbers:
        library_name_stem = library_name_stem.rstrip("0123456789")
        # `library_name_stem` will now be e.g. `libQ4M_cuda`;
        # let's tack the new version number and the original extension back on.
        library_name = f"{library_name_stem}{override_value}.{library_name_ext}"
        logger.warning(
            f"WARNING: BNB_CUDA_VERSION={override_value} environment variable detected; loading {library_name}.\n"
            "This can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\n"
            "If this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\n"
            "If you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\n"
            "For example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n",
        )

    return PACKAGE_DIR / library_name


def get_native_library_path():
    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"
    cuda_specs = get_cuda_specs()
    if cuda_specs:
        cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)
        if cuda_binary_path.exists():
            binary_path = cuda_binary_path
        else:
            logger.warning(
                "Could not find the bitsandbytes CUDA binary at %r", cuda_binary_path
            )
    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")
    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # only a CUDA-built library exposes this
        return binary_path

    logger.warning(
        "The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.",
    )
    return binary_path


try:
    lib_path = get_native_library_path()
except Exception as e:
    lib_path = None
    logger.error(f"Could not load bitsandbytes native library: {e}", exc_info=True)
    if context.get_context("device_target") == "GPU":
        logger.warning(
            """
CUDA Setup failed despite CUDA being available. Please run the following command to get more information:

python -m bitsandbytes

Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
and open an issue at: https://github.com/hypertseng/Q4M/issues
""",
        )
