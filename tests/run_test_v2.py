"""Test runner for mindtorch_v2 with NPU (Ascend) support.

Simple test runner that:
1. Installs mindtorch_v2 as torch replacement
2. Provides torch_npu compatibility
3. Patches safetensors to use Python-level loading
4. Runs pytest with TRANSFORMERS_TEST_DEVICE support

Usage:
    # Run on NPU (Ascend)
    TRANSFORMERS_TEST_DEVICE=npu python tests/run_test_v2.py -vs tests/transformers/tests/models/albert/test_modeling_albert.py

    # Run on CPU
    python tests/run_test_v2.py -vs tests/transformers/tests/models/albert/test_modeling_albert.py
"""
import os
import sys

# Add src directory to Python path - MUST be first to avoid shadowing by tests/mindtorch_v2
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
tests_path = os.path.join(project_root, 'tests')

# Remove tests directory from sys.path to avoid shadowing (tests/mindtorch_v2 would shadow src/mindtorch_v2)
while tests_path in sys.path:
    sys.path.remove(tests_path)

# Insert src_path at position 0
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)

# Install mindtorch_v2 torch proxy
from mindtorch_v2._torch_proxy import install
install()


def create_torch_npu_module():
    """Create a fake torch_npu module that delegates to mindtorch_v2.npu.

    This is needed because transformers checks for torch_npu package availability.
    """
    import types
    import importlib.util
    import torch  # This is mindtorch_v2 thanks to the proxy

    torch_npu = types.ModuleType('torch_npu')
    torch_npu.__version__ = '2.1.0'

    # Set proper module spec to avoid "torch_npu.__spec__ is None" errors
    # This is required by accelerate's is_npu_available() check
    torch_npu.__spec__ = importlib.util.spec_from_loader('torch_npu', loader=None)
    torch_npu.__file__ = __file__
    torch_npu.__path__ = []

    # Copy all functions from torch.npu to torch_npu
    for attr in dir(torch.npu):
        if not attr.startswith('_'):
            setattr(torch_npu, attr, getattr(torch.npu, attr))

    # Add NPU-specific functions needed by transformers
    def npu_fusion_attention(query, key, value, head_num, input_layout,
                              pse=None, padding_mask=None, atten_mask=None,
                              scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
                              next_tockens=0, inner_precise=1, prefix=None,
                              sparse_mode=0, actual_seq_qlen=None,
                              actual_seq_kvlen=None, gen_mask_parallel=True,
                              sync=False):
        """Stub for NPU fusion attention - falls back to standard attention."""
        # This is a stub - transformers will detect it's available and try to use it
        # We'll raise NotImplementedError to fall back to standard attention
        raise NotImplementedError("NPU fusion attention not available in mindtorch_v2")

    torch_npu.npu_fusion_attention = npu_fusion_attention

    # Add other commonly needed NPU functions
    torch_npu.npu_format_cast = lambda x, format: x  # No-op
    torch_npu.get_npu_format = lambda x: 0  # Return default format

    # Add torch_npu to sys.modules
    sys.modules['torch_npu'] = torch_npu

    return torch_npu


# Create torch_npu module for compatibility
create_torch_npu_module()


def patch_safetensors_for_mindtorch_v2():
    """Patch safetensors to use Python-level tensor loading.

    The Rust-based safe_open.get_tensor() and PySafeSlice.__getitem__
    return bool instead of tensor when using mindtorch_v2.
    This patch replaces them with pure Python implementations.
    """
    import json
    import mmap
    from typing import OrderedDict
    import numpy as np
    import safetensors
    import torch  # This is mindtorch_v2 thanks to the proxy

    MAX_HEADER_SIZE = 100 * 1000 * 1000

    _NP_TYPES = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.float16,  # Fall back to float16 for bfloat16
        "I64": np.int64,
        "U64": np.uint64,
        "I32": np.int32,
        "U32": np.uint32,
        "I16": np.int16,
        "U16": np.uint16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": bool,
    }

    _DTYPE_SIZE = {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4,
        "I64": 8, "U64": 8, "F64": 8,
    }

    class PySafeSlice:
        def __init__(self, info, bufferfile, base_ptr, buffermmap):
            self.info = info
            self.bufferfile = bufferfile
            self.buffermmap = buffermmap
            self.base_ptr = base_ptr

        @property
        def shape(self):
            return self.info["shape"]

        @property
        def dtype(self):
            return _NP_TYPES[self.info["dtype"]]

        @property
        def start_offset(self):
            return self.base_ptr + self.info["data_offsets"][0]

        def get_shape(self):
            """Return the shape of the tensor."""
            return self.info["shape"]

        def get_dtype(self):
            """Return the dtype string (e.g., 'F32')."""
            return self.info["dtype"]

        def get(self, slice_arg=None):
            nbytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
            buffer = bytearray(nbytes)
            self.bufferfile.seek(self.start_offset)
            self.bufferfile.readinto(buffer)
            array = np.frombuffer(buffer, dtype=self.dtype).reshape(self.shape)
            if slice_arg is not None:
                array = array[slice_arg]
            return torch.from_numpy(array.copy())

        def __getitem__(self, slice_arg):
            return self.get(slice_arg)

    def getSize(fileobject):
        fileobject.seek(0, 2)
        size = fileobject.tell()
        fileobject.seek(0)
        return size

    def metadata_validate(metadata):
        end = 0
        for key, info in metadata.items():
            s, e = info["data_offsets"]
            if e < s:
                raise ValueError(f"SafeTensorError::InvalidOffset({key})")
            if e > end:
                end = e
        return end

    def read_metadata(buffer):
        buffer_len = getSize(buffer)
        if buffer_len < 8:
            raise ValueError("SafeTensorError::HeaderTooSmall")

        n = np.frombuffer(buffer.read(8), dtype=np.uint64).item()

        if n > MAX_HEADER_SIZE:
            raise safetensors.SafetensorError("SafeTensorError::HeaderTooLarge")

        stop = n + 8
        if stop > buffer_len:
            raise safetensors.SafetensorError("SafeTensorError::InvalidHeaderLength")

        tensors = json.loads(buffer.read(n), object_pairs_hook=OrderedDict)
        metadata = tensors.pop("__metadata__", None)
        buffer_end = metadata_validate(tensors)

        if buffer_end + 8 + n != buffer_len:
            raise ValueError("SafeTensorError::MetadataIncompleteBuffer")

        return stop, tensors, metadata

    class fast_safe_open:
        def __init__(self, filename, framework=None, device="cpu"):
            self.filename = filename
            self.framework = framework
            self.file = open(self.filename, "rb")
            self.file_mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
            self.base, self.tensors_decs, self.__metadata__ = read_metadata(self.file)
            self.tensors = OrderedDict()
            for key, info in self.tensors_decs.items():
                self.tensors[key] = PySafeSlice(info, self.file, self.base, self.file_mmap)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def metadata(self):
            meta = self.__metadata__
            if meta is not None:
                meta["format"] = "pt"
            return meta

        def keys(self):
            return list(self.tensors.keys())

        def get_tensor(self, name):
            return self.tensors[name].get()

        def get_slice(self, name):
            return self.tensors[name]

        def offset_keys(self):
            return self.keys()

    def safe_load_file(filename, device='cpu'):
        result = {}
        with fast_safe_open(filename, framework="pt", device=device) as f:
            for k in f.keys():
                result[k] = f.get_tensor(k)
        return result

    # Apply patches
    safetensors.safe_open = fast_safe_open
    from safetensors import torch as st
    st.load_file = safe_load_file


def setup_test_skips():
    """Set up test skipping for unsupported features."""
    import torch

    # torch._dynamo is already a stub module with no-op decorators
    # torch.compile is already a no-op in the loader

    # Set up CUDA-like attributes for compatibility
    if not hasattr(torch, 'cuda'):
        import types
        torch.cuda = types.ModuleType('cuda')
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0

    # Mark SDPA as not available - mindtorch_v2 doesn't have optimized SDPA kernels
    # This will cause transformers to skip SDPA-related tests
    torch._sdpa_available = False
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # Keep the function but mark it as not the optimized version
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        def sdpa_with_warning(*args, **kwargs):
            return original_sdpa(*args, **kwargs)
        sdpa_with_warning._is_optimized = False
        torch.nn.functional.scaled_dot_product_attention = sdpa_with_warning


# Apply patches
patch_safetensors_for_mindtorch_v2()
setup_test_skips()

# Print device info
import torch
print(f"[mindtorch_v2] Default device: {torch.get_default_device()}")
print(f"[mindtorch_v2] NPU available: {torch.npu.is_available()}")
if os.environ.get('TRANSFORMERS_TEST_DEVICE'):
    print(f"[mindtorch_v2] TRANSFORMERS_TEST_DEVICE={os.environ['TRANSFORMERS_TEST_DEVICE']}")

# Run pytest (with --noconftest to avoid transformers conftest.py import issues)
import pytest
args = sys.argv[1:]
if '--noconftest' not in args:
    args.insert(0, '--noconftest')
sys.exit(pytest.main(args))
