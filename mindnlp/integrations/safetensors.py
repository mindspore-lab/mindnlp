from typing import OrderedDict
import mmap
import json
import numpy as np

import safetensors

from mindnlp import core

from core.configs import SUPPORT_BF16

if SUPPORT_BF16:
    from mindspore.common.np_dtype import bfloat16  # pylint: disable=import-error
else:
    from ml_dtypes import bfloat16

MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
PROTOCOL_VERSION = 1001
MAX_HEADER_SIZE = 100 * 1000 * 1000


_MS_TYPES = {
    "F64": core.float64,
    "F32": core.float32,
    "F16": core.float16,
    "BF16": core.bfloat16,
    "I64": core.int64,
    "U64": core.uint64,
    "I32": core.int32,
    "U32": core.uint32,
    "I16": core.int16,
    "U16": core.uint16,
    "I8": core.int8,
    "U8": core.uint8,
    "BOOL": core.bool,
}

_NP_TYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": bfloat16,
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
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}

class PySafeSlice:
    def __init__(self, info, bufferfile, base_ptr, buffermmap):
        self.info = info
        self.bufferfile = bufferfile
        self.buffermmap = buffermmap
        self.base_ptr = base_ptr

        self.start = [0 for _ in self.shape]
        self.stop = list(self.shape)
        self.step = [1 for _ in self.shape]

    @property
    def ndim(self):
        return len(self.shape)

    def get(self, *args, **kwargs):
        nbytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        offset = self.start_offset
        tensor = np.frombuffer(self.buffermmap, dtype=self.dtype, offset=offset,
                               count=nbytes // np.dtype(self.dtype).itemsize)
        tensor = tensor.reshape(self.shape)
        if not SUPPORT_BF16 and self.info["dtype"] == 'BF16':
            tensor = tensor.astype(np.float16)
        tensor = core.from_numpy(tensor)
        return tensor

    @property
    def start_offset(self):
        return self.base_ptr + self.info["data_offsets"][0]

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.info["dtype"]

    @property
    def shape(self):
        return self.info["shape"]

    @property
    def dtype(self):
        return _NP_TYPES[self.info["dtype"]]

    @property
    def nelements(self):
        return np.prod(self.info["shape"])

    @property
    def bits(self):
        return _DTYPE_SIZE[self.info["dtype"]]

    @property
    def nbytes(self):
        return self.nelements * self.bits

    def __getitem__(self, slice):
        if slice is Ellipsis:
            return self.get()
        return self.get()[slice]

def getSize(fileobject):
    fileobject.seek(0, 2)  # move the cursor to the end of the file
    size = fileobject.tell()
    fileobject.seek(0)  # move the cursor to the start of the file
    return size


def metadata_validate(metadata):
    start = 0
    for key, info in metadata.items():
        s, e = info["data_offsets"]
        if s != start or e < s:
            raise ValueError(f"SafeTensorError::InvalidOffset({key})")
        start = e
        nelements = np.prod(info["shape"])
        nbytes = nelements * _DTYPE_SIZE[info["dtype"]]
        if (e - s) != nbytes:
            raise ValueError("SafeTensorError::TensorInvalidInfo")
    return start

def read_metadata(buffer):
    buffer_len = getSize(buffer)
    if buffer_len < 8:
        raise ValueError("SafeTensorError::HeaderTooSmall")

    n = np.frombuffer(buffer.read(8), dtype=np.uint64).item()

    if n > MAX_HEADER_SIZE:
        raise ValueError("SafeTensorError::HeaderTooLarge")

    stop = n + 8
    if stop > buffer_len:
        raise ValueError("SafeTensorError::InvalidHeaderLength")

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
            self.tensors[key].key = key

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def metadata(self):
        return self.__metadata__

    def keys(self):
        return list(self.tensors.keys())

    def get_tensor(self, name):
        return self.tensors[name].get()

    def get_slice(self, name):
        return self.tensors[name]

def safe_load_file(filename):
    """
    This function safely loads a file containing state dictionary data and converts it into a dictionary of MindSpore Parameters.

    Args:
        filename (str): The path to the file containing the state dictionary data to be loaded.

    Returns:
        dict: A dictionary where keys are parameter names and values are MindSpore Parameters.

    Raises:
        FileNotFoundError: If the specified file 'filename' does not exist.
        ValueError: If the data in the file is not in the correct format to create MindSpore Parameters.
    """
    result = {}
    with fast_safe_open(filename, framework="np") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def safe_save_file(tensor_dict, filename, metadata=None):
    """
    Function to safely save a dictionary of tensors to a file.

    Args:
        tensor_dict (dict): A dictionary where keys are strings and values are numpy arrays representing tensors.
        filename (str): The name of the file where the tensor data will be saved.
        metadata (optional): Additional metadata to be saved along with the tensor data. Default is None.

    Returns:
        None. The function does not return any value explicitly.

    Raises:
        ValueError: If the input tensor_dict is not in the expected format.
        IOError: If there are issues with writing the data to the specified file.
        Exception: Any other unexpected error that may occur during the process.
    """
    tensor_dict = {k: v.asnumpy() for k, v in tensor_dict.items()}
    return safetensors.numpy.save_file(tensor_dict, filename, metadata)

safetensors.safe_open = fast_safe_open
from safetensors import torch
torch.load_file = safe_load_file
