# Copyright 2023 Huawei Technologies Co., Ltd
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
Serialization utils
"""
import os
import io
import sys
import pickle
import shutil
import zipfile
import tarfile
import pathlib
import warnings
import tempfile
import operator

from contextlib import closing, contextmanager
from enum import Enum
from typing import Dict, Union, Optional, Any, OrderedDict
from functools import reduce
from dataclasses import dataclass

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.train.serialization import _exec_save, _parse_ckpt_proto, tensor_to_np_type, tensor_to_ms_type

import safetensors
import safetensors.numpy

from mindnlp.configs import SUPPORT_BF16
from .nn import Module
from ..utils import logging


if SUPPORT_BF16:
    from mindspore.common.np_dtype import bfloat16 # pylint: disable=import-error
else:
    from ml_dtypes import bfloat16

logger = logging.get_logger(__name__)

MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
PROTOCOL_VERSION = 1001

@contextmanager
def mkdtemp():
    """
    Context manager that creates a temporary directory and provides its path.
    
    Usage:
        with mkdtemp() as path:
            # Use the temporary directory at 'path'
    
    Args:
        This function does not take any parameters.
    
    Returns:
        None.
    
    Raises:
        This function does not raise any exceptions.
    """
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)

class PyTorchFileReader:
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """
    def __init__(self, file):
        """
        Initializes a new instance of PyTorchFileReader.
        
        Args:
            self (PyTorchFileReader): The instance of the PyTorchFileReader class.
            file (str): The path to the zip file to be read.
        
        Returns:
            None. This method initializes the PyTorchFileReader instance with the provided file.
        
        Raises:
            IOError: If the file specified by the 'file' parameter does not exist or cannot be opened.
            zipfile.BadZipFile: If the file specified by the 'file' parameter is not a valid zip file.
            IndexError: If the zip file does not contain any files.
        """
        self.file = zipfile.ZipFile(file)
        self.directory = self.file.namelist()[0].split('/')[0]

    def open_record(self, name):
        """
        Opens a record file from the PyTorchFileReader directory.
        
        Args:
            self (PyTorchFileReader): The instance of the PyTorchFileReader class.
            name (str): The name of the record file to open.
        
        Returns:
            None: If the specified record file does not exist in the PyTorchFileReader directory.
        
        Raises:
            None.
        
        This method checks if the specified record file exists in the PyTorchFileReader directory. If it does, the file is opened and returned. If the file does not exist, None is returned.
        """
        filename = f"{self.directory}/{name}"
        if filename in self.file.namelist():
            return self.file.open(filename)
        return None

    def read_record(self, name):
        """
        Reads a record from a PyTorch file.
        
        Args:
            self (PyTorchFileReader): An instance of the PyTorchFileReader class.
            name (str): The name of the record to read from the PyTorch file.
        
        Returns:
            None: If the record with the specified name does not exist in the PyTorch file.
        
        Raises:
            FileNotFoundError: If the PyTorch file does not exist in the specified directory.
            IOError: If there is an error in reading the PyTorch file.
        
        """
        filename = f"{self.directory}/{name}"
        if filename in self.file.namelist():
            return self.file.read(filename)
        return None

    def has_record(self, name):
        """
        This method checks if a record with the specified name exists in the PyTorchFileReader's directory.
        
        Args:
            self (PyTorchFileReader): An instance of the PyTorchFileReader class.
            name (str): The name of the record to be checked in the directory.
        
        Returns:
            None: This method returns None.
        
        Raises:
            None
        """
        filename = f"{self.directory}/{name}"
        return filename in self.file.namelist()

    def get_all_records(
        self,
    ):
        """
        Retrieves a list of all records from the PyTorchFileReader object.
        
        Args:
            self: The PyTorchFileReader object itself.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        This method iterates through the files in the PyTorchFileReader object's directory and retrieves the names of all records. The records are then returned as a list of file names.
        
        Note:
            - The PyTorchFileReader object must be initialized with a valid directory.
            - The list of file names returned only includes the names of the files, without the directory path.
        """
        files = [name.replace(self.directory + '/' , '')for name in self.file.namelist()]
        return files

    def get_record_offset(self, name):
        """
        Returns the header offset of a specified record in a PyTorch file.
        
        Args:
            self (PyTorchFileReader): An instance of the PyTorchFileReader class.
            name (str): The name of the record for which the header offset is to be retrieved.
        
        Returns:
            None: If the specified record does not exist in the PyTorch file.
        
        Raises:
            None.
        
        This method takes in the self parameter, which is an instance of the PyTorchFileReader class. It also takes a name parameter, which represents the name of the record for which the header offset is to
be retrieved. The method checks if the specified record exists in the PyTorch file by creating the filename using the directory attribute of the PyTorchFileReader instance and the provided name. If the
filename exists in the file's namelist, the method returns the header offset of the file info associated with the filename. Otherwise, it returns None, indicating that the specified record does not exist in
the file.
        """
        filename = f"{self.directory}/{name}"
        if filename in self.file.namelist():
            return self.file.getinfo(filename).header_offset
        return None

class PyTorchFileWriter:
    def __init__(self, file):
        self.zipfile = zipfile.ZipFile(file, mode='w')
        self.written_records = set()

    def write_record(self, name, data, offset=0):
        if name in self.written_records:
            raise RuntimeError(f"Record {name} already written")
        self.written_records.add(name)
        self.zipfile.writestr(name, data)

    def write_end_of_file(self):
        pass

    def get_all_written_records(self):
        return self.written_records

class LoadEndianness(Enum):

    """
    Represents an enumeration for specifying the byte order (endianness) of a data load.
    
    This class inherits from the built-in Enum class in Python and provides a set of pre-defined constants for different byte orders. The byte order determines the arrangement of bytes in a multi-byte data
type, such as integers and floating-point numbers, when it is stored or transmitted.
    
    Attributes:
        BIG_ENDIAN: Represents the big-endian byte order where the most significant byte is stored first.
        LITTLE_ENDIAN: Represents the little-endian byte order where the least significant byte is stored first.
        NATIVE: Represents the native byte order of the underlying platform.
        NETWORK: Represents the byte order used in network byte order, which is big-endian.
    
    The LoadEndianness class allows you to easily specify the desired byte order when loading data, ensuring compatibility with the expected byte order. It provides a convenient and readable way to work with
different byte orders without the need for manual byte swapping or conversion.
    
    Usage:
        The LoadEndianness class can be used to specify the byte order when loading data from a file, network, or any other data source. Simply import the class and use the desired constant to set the byte
order.
    
    Example:
        >>> load_endianness = LoadEndianness.BIG_ENDIAN
        >>> data = load_data(source_file, byte_order=load_endianness)
        >>> print(data)
    
    Note:
        It is important to ensure that the byte order specified matches the actual byte order of the data being loaded. Using the wrong byte order can lead to incorrect interpretation of the data and produce
unexpected results.
    
    """
    NATIVE = 1
    LITTLE = 2
    BIG = 3

_default_load_endian: Optional[LoadEndianness] = None

def get_default_load_endianness() -> Optional[LoadEndianness]:
    '''
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    '''
    return _default_load_endian

def set_default_load_endianness(endianness):
    '''
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    '''
    global _default_load_endian
    if not isinstance(endianness, LoadEndianness) and endianness is not None:
        raise TypeError("Invalid argument type in function set_default_load_endianness")
    _default_load_endian = endianness

def _is_zipfile(f) -> bool:
    """
    Args:
        f (file object): The file object to be checked for being a valid zip file.
            It should be opened in binary mode and point to the beginning of the file.
    
    Returns:
        bool: Returns True if the input file is a valid zip file, otherwise False.
    
    Raises:
        No specific exceptions are raised by this function.
    """
    # This is a stricter implementation than zipfile.is_zipfile().
    # zipfile.is_zipfile() is True if the magic number appears anywhere in the
    # binary. Since we expect the files here to be generated by torch.save or
    # torch.jit.save, it's safe to only check the start bytes and avoid
    # collisions and assume the zip has only 1 file.
    # See bugs.python.org/issue28494.

    # Read the first 4 bytes of the file
    read_bytes = []
    start = f.tell()

    byte = f.read(1)
    while byte != b"":
        read_bytes.append(byte)
        if len(read_bytes) == 4:
            break
        byte = f.read(1)
    f.seek(start)

    local_header_magic_number = [b'P', b'K', b'\x03', b'\x04']
    return read_bytes == local_header_magic_number

def _check_seekable(f) -> bool:
    """
    Checks if the given file object is seekable.
    
    Args:
        f (file object): The file object to be checked for seekability.
    
    Returns:
        bool: True if the file object is seekable, False otherwise.
    
    Raises:
        UnsupportedOperation: If the file object does not support seek or tell operations.
        AttributeError: If the file object does not have the seek or tell attribute.
    """
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (str(e) + ". You can only torch.load from a file that is seekable."
                                + " Please pre-load the data into a buffer like io.BytesIO and"
                                + " try to load from it instead.")
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)
    return False

def _is_compressed_file(f) -> bool:
    """
    Checks whether the given file is a compressed file.
    
    Args:
        f (object): The file object to be checked.
    
    Returns:
        bool: Returns True if the file is compressed, False otherwise.
    
    Raises:
        None.
    
    """
    compress_modules = ['gzip']
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False

def _should_read_directly(f):
    """
    Checks if f is a file that should be read directly. It should be read
    directly if it is backed by a real file (has a fileno) and is not a
    a compressed file (e.g. gzip)
    """
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False


def _is_path(name_or_buffer):
    """
    Check if the input is a valid path.
    
    Args:
        name_or_buffer (str or pathlib.Path): A string representing a file path or a pathlib.Path object.
        
    Returns:
        None: This function does not return any value.
    
    Raises:
        None
    """
    return isinstance(name_or_buffer, (str, pathlib.Path))

def _is_torchscript_zip(zip_file):
    """
    Checks if the given zip file contains a specific record.
    
    Args:
        zip_file (object): The zip file to be checked for the presence of a specific record.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None
    """
    return 'constants.pkl' in zip_file.get_all_records()

class _opener:

    """
    Class `_opener` represents a context manager for opening files in Python. It inherits from the built-in `object` class.
    
    This class provides a convenient way to work with file-like objects by allowing them to be used within a `with` statement. The `_opener` class implements the `__init__`, `__enter__`, and `__exit__` methods.
    
    __init__(self, file_like):
        Initializes an instance of the `_opener` class.
        
        Parameters:
            - file_like: A file-like object that will be used for reading or writing operations.
            
    __enter__(self):
        Returns the file-like object passed during initialization.
        
        Returns:
            The file-like object for use within the `with` statement.
            
    __exit__(self, *args):
        Performs cleanup operations after the `with` statement block is executed.
        
        Parameters:
            - *args: Any exception arguments passed by the Python interpreter.
            
        Note:
            This method does not handle exceptions. It is designed to be used as a context manager and should be used in conjunction with a `try-except-finally` block to handle exceptions properly.
    """
    def __init__(self, file_like):
        """
        Initializes an instance of the '_opener' class.
        
        Args:
            self (object): The instance of the '_opener' class.
            file_like (object): A file-like object representing the file to be opened.
                It can be a file object, a file path, or any object with a file-like interface.
                The object must support the 'read' method.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        self.file_like = file_like

    def __enter__(self):
        """
        The '__enter__' method is a special method in the '_opener' class that is used to set up the context for an object. It is called when using the 'with' statement in Python.
        
        Args:
            self: An instance of the '_opener' class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not raise any exceptions.
        """
        return self.file_like

    def __exit__(self, *args):
        """
        Method '__exit__' in the class '_opener'.
        
        Args:
            self: The instance of the class.
                Type: object
                Purpose: Represents the instance of the class.
                Restrictions: None
        
        Returns:
            None: Indicates that the method does not return any value.
                Type: None
                Purpose: Signifies the absence of a return value.
        
        Raises:
            No specific exceptions are raised by this method.
        """


class _open_file(_opener):

    """
    _open_file represents a class that inherits from _opener and provides methods for opening and closing files.
    
    This class initializes an instance of _open_file with the given name and mode and utilizes the super() function to call the __init__ method of the _opener class with the opened file.
    
    The __exit__ method is implemented to close the file-like object when the instance is exited.
    
    Attributes:
        name (str): The name of the file to be opened.
        mode (str): The mode in which the file should be opened.
    
    Methods:
        __init__(name, mode):
            Initializes an instance of _open_file with the given name and mode.
    
        __exit__(*args):
            Closes the file-like object when the instance is exited.
    """
    def __init__(self, name, mode):
        """
        __init__
        
        Initializes an instance of the _open_file class.
        
        Args:
            self: _open_file instance
                The instance of the _open_file class.
        
            name: str
                The name of the file to be opened.
        
            mode: str
                The mode in which the file should be opened. It should be a string that represents the mode in which the file is opened. It can be 'r' for reading, 'w' for writing, or 'a' for appending. Other
modes are also supported.
        
        Returns:
            None
            This method does not return any value.
        
        Raises:
            OSError
                If an error occurs while opening the file, an OSError is raised.
        """
        super().__init__(open(name, mode))

    def __exit__(self, *args):
        """
        This method __exit__ is used in the class _open_file to handle the cleanup operations when exiting a context manager.
        
        Args:
        - self (object): The instance of the _open_file class. Represents the context manager itself.
        
        Returns:
        None. This method does not return any value explicitly.
        
        Raises:
        This method does not raise any exceptions explicitly. However, it indirectly depends on the behavior of the 'close()' method of the file-like object it operates on, which may raise exceptions related
to file I/O operations.
        """
        self.file_like.close()


class _open_buffer_reader(_opener):

    """
    A class representing an open buffer reader for reading files.
    
    This class is a subclass of _opener and provides functionality for reading files from a buffer. 
    The class's forwardor takes a buffer as input and initializes the buffer for reading. 
    It also performs a check to ensure that the buffer is seekable before proceeding with reading operations.
    """
    def __init__(self, buffer):
        """
        Initializes an instance of the '_open_buffer_reader' class.
        
        Args:
            self: The instance of the '_open_buffer_reader' class.
            buffer: The buffer object to be read. It should be a seekable object.
        
        Returns:
            None
        
        Raises:
            TypeError: If the 'buffer' parameter is not a seekable object.
        """
        super().__init__(buffer)
        _check_seekable(buffer)


class _open_buffer_writer(_opener):

    """
    _open_buffer_writer is a Python class that represents a buffered writer for file-like objects. This class inherits from the _opener class.
    
    Usage:
        The _open_buffer_writer class provides a convenient way to write data to file-like objects with buffering capabilities.
    
    Attributes:
        file_like (file-like object): The file-like object to which data will be written.
    
    Methods:
        __init__(self, file_like):
            Initializes a new instance of the _open_buffer_writer class.
            
            Args:
                file_like (file-like object): The file-like object to which data will be written.
        
        write(self, data):
            Writes the given data to the file-like object.
            
            Args:
                data (str): The data to be written.
                
            Returns:
                None
                
        flush(self):
            Flushes the buffer and writes any remaining data to the file-like object.
            
            Returns:
                None
                
        __enter__(self):
            Enters the context manager and returns the _open_buffer_writer instance.
            
            Returns:
                _open_buffer_writer: The current _open_buffer_writer instance.
                
        __exit__(self, *args):
            Exits the context manager and performs necessary cleanup operations.
            
            Args:
                *args: Variable length argument list.
                
            Returns:
                None
    """
    def __exit__(self, *args):
        """
        __exit__
        
        Args:
            self: _open_buffer_writer
                The instance of the _open_buffer_writer class.
        
        Returns:
            None: 
                This method does not return any value.
        
        Raises:
            N/A
        """
        self.file_like.flush()


def _open_file_like(name_or_buffer, mode):
    """
    Args:
        name_or_buffer (str or buffer): The name of the file or a buffer object to be opened. If a string, it represents the file path. If a buffer, it represents a memory buffer.
        mode (str): The mode in which the file or buffer should be opened. It should be either 'r' for reading or 'w' for writing.
    
    Returns:
        None: This function does not return a value.
    
    Raises:
        RuntimeError: If the mode is not 'r' or 'w'.
    """
    if _is_path(name_or_buffer):
        return _open_file(name_or_buffer, mode)
    if 'w' in mode:
        return _open_buffer_writer(name_or_buffer)
    if 'r' in mode:
        return _open_buffer_reader(name_or_buffer)
    raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")

class _open_zipfile_reader(_opener):

    """
    The _open_zipfile_reader class represents a reader for opening and reading zip files. 
    It inherits from the _opener class and provides functionality for reading zip files.
    
    Attributes:
        name_or_buffer: The name or buffer of the file to be opened.
    
    Methods:
        __init__: Initializes the _open_zipfile_reader instance, using the specified name_or_buffer to open a PyTorchFileReader.
    """
    def __init__(self, name_or_buffer) -> None:
        """
        Initializes the _open_zipfile_reader class.
        
        Args:
            self (object): The instance of the _open_zipfile_reader class.
            name_or_buffer (str or file-like object): The name of the file or a buffer object for reading the zipfile. 
                It can be a string representing the name of the file or a file-like object for reading the zipfile data. 
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - TypeError: If the name_or_buffer parameter is not a string or file-like object.
            - ValueError: If the name_or_buffer parameter is empty or invalid.
            - IOError: If there is an error reading the zipfile from the provided name_or_buffer.
        """
        super().__init__(PyTorchFileReader(name_or_buffer))

class _open_zipfile_writer_file(_opener):
    def __init__(self, name):
        self.file_stream = None
        self.name = str(name)
        try:
            self.name.encode('ascii')
        except UnicodeEncodeError:
            self.file_stream = io.FileIO(self.name, mode='w')
            super().__init__(PyTorchFileWriter(self.file_stream))
        else:
            super().__init__(PyTorchFileWriter(self.name))

    def __exit__(self, *args):
        self.file_like.write_end_of_file()
        if self.file_stream is not None:
            self.file_stream.close()

class _open_zipfile_writer_buffer(_opener):
    def __init__(self, buffer):
        if not callable(getattr(buffer, "write", None)):
            msg = f"Buffer of {str(type(buffer)).strip('<>')} has no callable attribute 'write'"
            if not hasattr(buffer, "write"):
                raise AttributeError(msg)
            raise TypeError(msg)
        self.buffer = buffer
        super().__init__(PyTorchFileWriter(buffer))

    def __exit__(self, *args):
        self.file_like.write_end_of_file()
        self.buffer.flush()

def _open_zipfile_writer(name_or_buffer):
    if _is_path(name_or_buffer):
        container = _open_zipfile_writer_file
    else:
        container = _open_zipfile_writer_buffer
    return container(name_or_buffer)

def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    '''Rebuilds a tensor based on the provided parameters.
    
    Args:
        storage (ndarray): The storage array from which the tensor is created.
        storage_offset (int): The offset in the storage array from where the tensor data starts.
        size (tuple): The size of the tensor.
        stride (tuple or None): The stride of the tensor, or None if not applicable.
        requires_grad (bool): Indicates if the tensor requires gradient computation.
        backward_hooks (list): A list of backward hooks for the tensor.
        metadata (Any, optional): Additional metadata associated with the tensor.
    
    Returns:
        None: This function does not have a return value.
    
    Raises:
        None: This function does not raise any exceptions.
    '''
    if size == ():
        num_elemets = 1
    else:
        num_elemets = reduce(operator.mul, size)
    array = storage[storage_offset: storage_offset + num_elemets]

    if array.dtype == bfloat16 and not SUPPORT_BF16:
        logger.warning_once("MindSpore do not support bfloat16 dtype, we will automaticlly convert to float16")
        array = array.astype(np.float16)

    if stride is not None and len(stride) > 1 and stride[0] == 1 and stride[1] > 1:
        stride = tuple((s * 4 for s in stride))
        array = np.lib.stride_tricks.as_strided(array, size, stride)
    else:
        order = "C"
        array = array.reshape(size, order=order)
    param = mindspore.Tensor(array)
    return param

@dataclass
class FakeParameter:

    """
    This class represents a fake parameter in Python. 
    
    The 'FakeParameter' class inherits from [insert inherited class here]. 
    
    Class Attributes:
        [List any class attributes here, if applicable]
    
    Instance Attributes:
        [List any instance attributes here, if applicable]
    
    Methods:
        [List all the methods of the class here, along with their descriptions]
    
        - [method name]: [method description]
        - [method name]: [method description]
        - ...
    
    Usage:
        [Explain how to use the 'FakeParameter' class, including any important details or considerations]
    
    Example:
        [Provide an example usage of the 'FakeParameter' class]
    
        >>> [code example]
    
    """
    storage: np.ndarray = None
    storage_offset: int = None
    size: tuple = None
    stride: tuple = None
    requires_grad: bool = None

@dataclass
class FakeStorage:

    """
    This class represents a fake storage system in Python.
    
    The 'FakeStorage' class is designed to mimic a real storage system but without any actual functionality. It serves as a placeholder or a testing tool for applications that require a storage system.
    
    Attributes:
        None.
    
    Methods:
        None.
    
    Inheritance:
        This class does not inherit from any other class.
    
    Usage:
        1. Instantiate the 'FakeStorage' class to create a fake storage object.
        2. Use the object to simulate storage-related operations without actually interacting with a real storage system.
        3. Since this class does not have any attributes or methods, its usefulness lies in its ability to stand in for a real storage system during testing or development.
    
    Note:
        - This class is not intended for production use and should only be used for testing or development purposes.
        - It is recommended to replace instances of 'FakeStorage' with a real storage system before deploying the application.
    
    Example:
        from fake_storage import FakeStorage
    
        storage = FakeStorage()
        storage.upload_file('file.txt')
        storage.download_file('file.txt')
        storage.delete_file('file.txt')
    
        The above example demonstrates how to use the 'FakeStorage' class to simulate storage operations. However, since this is a fake storage system, no files are actually uploaded, downloaded, or deleted.
    
    """
    storage: np.ndarray = None

def _rebuild_tensor_legacy(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    """
    This function rebuilds a tensor using legacy parameters.
    
    Args:
        storage (Tensor): The storage for the tensor.
        storage_offset (int): The offset within the storage.
        size (tuple): The size of the tensor.
        stride (tuple): The stride of the tensor.
        requires_grad (bool): Indicates if gradients need to be computed for the tensor.
        backward_hooks (dict): Dictionary containing backward hooks for the tensor.
        metadata (optional): Additional metadata for the tensor.
    
    Returns:
        None. This function does not return any value.
    
    Raises:
        None. This function does not raise any exceptions.
    """
    return FakeParameter(storage, storage_offset, size, stride, requires_grad)

def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    """
    This function decodes a bytes string to ASCII if it is a bytes type, otherwise returns the input string.
    
    Args:
        bytes_str (Union[bytes, str]): A bytes or string input to be decoded if it is a bytes type. If it is already a string, it will be returned as is.
    
    Returns:
        str: The decoded ASCII string if the input is of bytes type, otherwise the original string.
    
    Raises:
        None
    """
    # When using encoding='bytes' in Py3, some **internal** keys stored as
    # strings in Py2 are loaded as bytes. This function decodes them with
    # ascii encoding, one that Py3 uses by default.
    #
    # NOTE: This should only be used on internal keys (e.g., `typename` and
    #       `location` in `persistent_load` below!
    if isinstance(bytes_str, bytes):
        return bytes_str.decode('ascii')
    return bytes_str


dtype_map = {
    "HalfStorage": np.float16,
    "FloatStorage": np.float32,
    'BFloat16Storage': bfloat16,
    'LongStorage': np.int64,
    'ByteStorage': np.uint8,
    'BoolStorage': np.bool_
}

element_size_map = {
    "HalfStorage": 2,
    "FloatStorage": 3,
    'BFloat16Storage': 2,
    'LongStorage': 4,
    'ByteStorage': 1,
    'BoolStorage': 1
}

def load(f, pickle_module=pickle, *, mmap=None, **pickle_load_args):
    """
    Load a file using pickle, optionally with memory mapping.
    
    Args:
        f (file-like object or str): The file to load from. If a string is provided, it should be the filename.
        pickle_module (module): The module to use for pickling. Defaults to the standard 'pickle' module.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        ValueError: Raised if 'f' is not a string filename when using mmap argument, or if torchscript is detected in a zipfile.
        RuntimeError: Raised if mmap argument is used without files saved with `torch.save(_use_new_zipfile_serialization=True)`.
    """
    if pickle_module is None:
        pickle_module = pickle

    # make flipping default BC-compatible
    if mmap is None:
        mmap = False

    if 'encoding' not in pickle_load_args:
        pickle_load_args['encoding'] = 'utf-8'

    with _open_file_like(f, 'rb') as opened_file:
        if _is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            overall_storage = None
            with _open_zipfile_reader(opened_file, ) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    raise ValueError('do not support torchscript now')
                if mmap:
                    if not isinstance(f, str):
                        raise ValueError("f must be a string filename in order to use mmap argument")
                    overall_storage = f

                return _load(opened_zipfile,
                             pickle_module,
                             overall_storage=overall_storage,
                             **pickle_load_args)
        if mmap:
            raise RuntimeError("mmap can only be used with files saved with ",
                               "`torch.save(_use_new_zipfile_serialization=True), "
                               "please torch.save your checkpoint with this option in order to use mmap.")

        return _legacy_load(opened_file, pickle_module, **pickle_load_args)

def _legacy_load(f, pickle_module, **pickle_load_args):
    """
    Args:
        f (file-like object): The file-like object containing the serialized data to be loaded.
        pickle_module (module): The module used for unpickling the serialized data.
        
    Returns:
        None. This function does not return any value.
        
    Raises:
        ValueError: Raised if legacy load for MindSpore is not supported.
        RuntimeError: Raised if an unknown saved id type is encountered during deserialization.
        RuntimeError: Raised if the magic number in the file does not match the expected value.
        RuntimeError: Raised if the protocol version in the file does not match the expected value.
        RuntimeError: Raised if there is an issue with the file-like object compatibility with torch.load.
    """
    deserialized_objects: Dict[int, Any] = {}

    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]

        def find_class(self, mod_name, name):
            if name == '_rebuild_tensor_v2':
                name = '_rebuild_tensor_legacy'
            if mod_name == 'torch._utils':
                return eval(name)
            if mod_name == 'torch':
                return str(name)
            return super().find_class(mod_name, name)

    def legacy_load(f):
        deserialized_objects: Dict[int, Any] = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                # Ignore containers that don't have any sources saved
                return saved_id[0]
            return deserialized_objects[int(saved_id)]

        with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, \
                mkdtemp() as tmpdir:
            raise ValueError('do not support legacy load for MindSpore.')

    deserialized_objects = {}

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == 'module':
            # Ignore containers that don't have any sources saved
            return data[0]
        if typename == 'storage':
            storage_type, root_key, location, numel, view_metadata = data
            location = _maybe_decode_ascii(location)

            if root_key not in deserialized_objects:
                typed_storage = FakeStorage(np.empty(numel, dtype_map[storage_type]))
                deserialized_objects[root_key] = typed_storage
            else:
                typed_storage = deserialized_objects[root_key]

            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                if view_key not in deserialized_objects:
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[view_key] = typed_storage[offset: offset + view_size]
                res = deserialized_objects[view_key]
            else:
                res = typed_storage
            return res
        raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)

    if f_should_read_directly and f.tell() == 0:
        # legacy_load requires that f has fileno()
        # only if offset is zero we can attempt the legacy tar file loader
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if _is_zipfile(f):
                # .zip is used for torch.jit.save and will throw an un-pickling error here
                raise RuntimeError(
                    f"{f.name} is a zip archive (did you mean to use torch.jit.load()?)") from None
            # if not a tarfile, reset file offset and proceed
            f.seek(0)

    if not hasattr(f, 'readinto') and (3, 8, 0) <= sys.version_info < (3, 8, 2):
        raise RuntimeError(
            "torch.load does not work with file-like objects that do not implement readinto on Python 3.8.0 and 3.8.1. "
            f"Received object of type \"{type(f)}\". Please update to Python 3.8.2 or newer to restore this "
            "functionality.")

    magic_number = pickle_module.load(f, **pickle_load_args)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f, **pickle_load_args)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")

    _sys_info = pickle_module.load(f, **pickle_load_args)
    unpickler = UnpicklerWrapper(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()
    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)

    offset = f.tell() if f_should_read_directly else None
    for key in deserialized_storage_keys:
        assert key in deserialized_objects
        typed_storage = deserialized_objects[key].storage
        f.read(8) # trick for read
        array = np.frombuffer(f.read(typed_storage.nbytes), typed_storage.dtype)
        deserialized_objects[key].storage = array
        if offset is not None:
            offset = f.tell()

    new_result = {}
    for k, v in result.items():
        num_elemets = reduce(operator.mul, v.size)
        array = v.storage.storage[v.storage_offset: v.storage_offset + num_elemets]
        stride = v.stride
        size = v.size
        if stride is not None and len(stride) > 1 and stride[0] == 1 and stride[1] > 1:
            stride = tuple((s * 4 for s in stride))
            array = np.lib.stride_tricks.as_strided(array, size, stride)
        else:
            order = "C"
            array = array.reshape(size, order=order)
        if array.dtype == bfloat16 and not SUPPORT_BF16:
            logger.warning_once("MindSpore do not support bfloat16 dtype, we will automaticlly convert to float16")
            array = array.astype(np.float16)
        new_result[k] = mindspore.Tensor(array)

    return new_result

def _load(zip_file, pickle_module, overall_storage=None, pickle_file='data.pkl', **pickle_load_args):
    """
    Loads data from a zip file using pickle serialization.
    
    Args:
        zip_file (zipfile.ZipFile): The zip file containing the data.
        pickle_module (module): The pickle module to use for deserialization.
        overall_storage (numpy.memmap, optional): The overall storage for loading the data.
        pickle_file (str, optional): The name of the pickle file within the zip file. Default is 'data.pkl'.
        **pickle_load_args: Additional keyword arguments to pass to the pickle module's load function.
    
    Returns:
        None
    
    Raises:
        ValueError: If an unknown endianness type is encountered.
        ValueError: If an invalid load endianness type is encountered.
        UserWarning: If the default load endianness is changed on big endian machines.
    
    """
    loaded_storages = {}
    # check if byteswapping is needed
    byteordername = 'byteorder'
    byteorderdata = None
    if zip_file.has_record(byteordername):
        byteorderdata = zip_file.read_record(byteordername)
        if byteorderdata not in [b'little', b'big']:
            raise ValueError('Unknown endianness type: ' + byteorderdata.decode())
    elif get_default_load_endianness() == LoadEndianness.LITTLE or \
            get_default_load_endianness() is None:
        byteorderdata = b'little'
    elif get_default_load_endianness() == LoadEndianness.BIG:
        byteorderdata = b'big'
    elif get_default_load_endianness() == LoadEndianness.NATIVE:
        pass
    else:
        raise ValueError('Invalid load endianness type')

    if not zip_file.has_record(byteordername) and \
            get_default_load_endianness() is None and \
            sys.byteorder == 'big':
        # Default behaviour was changed
        # See https://github.com/pytorch/pytorch/issues/101688
        warnings.warn("The default load endianness for checkpoints without a byteorder mark "
                      "on big endian machines was changed from 'native' to 'little' endian, "
                      "to avoid this behavior please use "
                      "torch.serialization.set_default_load_endianness to set "
                      "the desired default load endianness",
                      UserWarning)

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert typename == 'storage', \
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data

        name = f'data/{key}'
        if name in loaded_storages:
            return loaded_storages[name]

        if overall_storage is not None:
            array = np.memmap(overall_storage, dtype=dtype_map[storage_type], offset=zip_file.open_record(name)._fileobj.tell(), shape=(numel,))
        else:
            array = np.frombuffer(zip_file.read_record(name), dtype_map[storage_type])
        loaded_storages[name] = array
        return array

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        'torch.tensor': 'torch._tensor'
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if mod_name == 'torch._utils':
                return eval(name)
            if mod_name == 'torch':
                return str(name)

            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = zip_file.open_record(pickle_file)

    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    return result

def convert_torch_to_mindspore(pth_file):
    """convert torch checkpoint to mindspore"""
    try:
        import torch # pylint: disable=import-error
    except Exception as exc:
        raise ImportError("'import torch' failed, please install torch by "
                        "`pip install torch` or instructions from 'https://pytorch.org'") \
                        from exc
    if pth_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(pth_file)
        ms_ckpt_path = pth_file.replace('model-', 'mindspore-')
        ms_ckpt_path = ms_ckpt_path.replace('.safetensors', '.ckpt')

    else:
        ms_ckpt_path = pth_file.replace('pytorch_model', 'mindspore')
        ms_ckpt_path = ms_ckpt_path.replace('.bin', '.ckpt')

        state_dict = torch.load(pth_file, map_location='cpu')

    if os.path.exists(ms_ckpt_path):
        return ms_ckpt_path

    ms_ckpt = []
    logger.info('Starting checkpoint conversion.')

    has_bf16 = False
    for key, value in state_dict.items():
        if value.dtype == torch.bfloat16:
            data = Tensor(value.to(torch.float).numpy(), dtype=mindspore.float16)
            if not has_bf16:
                has_bf16 = True
        else:
            data = Tensor(value.numpy())
        ms_ckpt.append({'name': key, 'data': data})

    if has_bf16:
        logger.warning("MindSpore do not support bfloat16 dtype, we will automaticlly convert to float16")

    try:
        mindspore.save_checkpoint(ms_ckpt, ms_ckpt_path)
    except Exception as exc:
        raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, '
                            f'please checkout the path.') from exc

    return ms_ckpt_path

def _check_save_filelike(f):
    if not isinstance(f, (str, os.PathLike)) and not hasattr(f, 'write'):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with "
            "a 'write' attribute")

def save(obj, f, pickle_module = pickle, pickle_protocol = 2):
    _check_save_filelike(f)
    with _open_zipfile_writer(f) as opened_zipfile:
        _save(obj, opened_zipfile, pickle_module, pickle_protocol)

def _save(obj, zip_file, pickle_module, pickle_protocol):
    serialized_storages = {}

    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    zip_file.write_record('archive/data.pkl', data_value, len(data_value))

    for key in sorted(serialized_storages.keys()):
        name = f'archive/data/{key}'
        storage = serialized_storages[key]
        storage_data = storage.inner_data
        zip_file.write_record(name, storage_data)


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
    with safetensors.safe_open(filename, 'np') as f:
        for key in f.keys():
            dtype = f.get_tensor(key).dtype
            break

    state_dict = safetensors.numpy.load_file(filename)
    if (not SUPPORT_BF16 and dtype != bfloat16) or SUPPORT_BF16:
        out_states = {k: mindspore.Tensor(v) for k, v in state_dict.items()}
        return out_states

    out_states = {k: mindspore.Tensor(v.astype(np.float16)) for k, v in state_dict.items()}
    return out_states


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

def get_data_list(param_dict):
    """Get state dict of the Peft model for saving."""
    data_list = OrderedDict()  # {key: [dims, tensor_type, data]}

    for key, value in param_dict.items():
        data_list[key] = []
        dims = []
        if value.shape == ():
            dims.append(0)
        else:
            for dim in value.shape:
                dims.append(dim)
        data_list[key].append(dims)
        tensor_type = str(value.dtype)
        data_list[key].append(tensor_type)
        data = value.asnumpy().reshape(-1)
        data_list[key].append(data)

    return data_list

def save_checkpoint(save_obj, ckpt_file_name):
    r"""
    Save checkpoint to a specified file.
    """
    if isinstance(save_obj, Module):
        data_list = get_data_list(save_obj.state_dict())
    elif isinstance(save_obj, dict):
        data_list = get_data_list(save_obj)
    else:
        raise ValueError(f'not support save object {type(save_obj)}')
    _exec_save(ckpt_file_name, data_list)

def load_checkpoint(ckpt_file_name):
    """
    Load checkpoint info from a specified file.
    """
    try:
        checkpoint_list = _parse_ckpt_proto(ckpt_file_name, None, None) # pylint: disable=no-value-for-parameter
    except:
        checkpoint_list = _parse_ckpt_proto(ckpt_file_name, None, None, None)

    parameter_dict = {}
    try:
        param_data_list = []

        for element_id, element in enumerate(checkpoint_list.value):
            if element.tag == "random_op":
                parameter_dict["random_op"] = element.tensor.tensor_content
                continue

            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type.get(data_type)
            ms_type = tensor_to_ms_type[data_type]
            if data_type == 'str':
                str_length = int(len(data) / 4)
                np_type = np_type + str(str_length)
            if data_type == "BFloat16":
                dims = element.tensor.dims
                param_data = np.frombuffer(data, np_type)
                param_data = param_data.reshape(list(dims))
                parameter = Tensor(param_data, ms_type)
                parameter_dict[element.tag] = parameter
                continue
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                new_data = b"".join(param_data_list)
                param_data = np.frombuffer(new_data, np_type)
                param_data_list.clear()
                dims = element.tensor.dims
                if dims == [0] and data_type == 'str':
                    parameter_dict[element.tag] = str(element_data[0])
                else:
                    if dims == [0] and 'Float' in data_type:
                        param_data = float(param_data[0])
                    if dims == [0] and 'Int' in data_type:
                        param_data = int(param_data[0])
                    if dims not in ([0], [1]):
                        param_data = param_data.reshape(list(dims))
                    parameter = Tensor(param_data, ms_type)
                    parameter_dict[element.tag] = parameter

    except BaseException as e:
        raise ValueError(e.__str__() + "\nFor 'load_checkpoint', "
                                       "failed to load the checkpoint file {}.".format(ckpt_file_name)) from e

    if not parameter_dict:
        raise ValueError("The loaded parameter dict is empty after filter or specify, please check whether "
                         "'filter_prefix' or 'specify_prefix' are set correctly.")

    return parameter_dict
