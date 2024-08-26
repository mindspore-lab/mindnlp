# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import os
import sys
import warnings
from types import ModuleType
from collections import OrderedDict
from functools import wraps, lru_cache
from typing import Tuple, Union
import importlib.util
from packaging import version

from . import logging

if sys.version_info >= (3, 8):
    # For Python 3.8 and later
    from importlib import metadata as importlib_metadata
else:
    # For Python versions earlier than 3.8
    import importlib_metadata


logger = logging.get_logger(__name__)

def _is_package_available(
        pkg_name: str, return_version: bool = False
) -> Union[Tuple[bool, str], bool]:
    """
    Checks if a specified package is available and optionally returns its version.
    
    Args:
        pkg_name (str): The name of the package to check for availability.
        return_version (bool, optional): Indicates whether to return the package version along with availability status. Defaults to False.
    
    Returns:
        Union[Tuple[bool, str], bool]: If return_version is True, returns a tuple containing a boolean indicating package availability and a string representing the package version. 
        If return_version is False, returns a boolean indicating package availability.
    
    Raises:
        No specific exceptions are raised within this function.
    """
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib_metadata.version(pkg_name)
            package_exists = True
        except importlib_metadata.PackageNotFoundError:
            package_exists = False
        logger.debug(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    return package_exists


_ftfy_available = _is_package_available("ftfy")
_einops_available = _is_package_available('einops')
_tiktoken_available = _is_package_available('tiktoken')
_bs4_available = importlib.util.find_spec("bs4") is not None
_pytest_available = _is_package_available("pytest")
_datasets_available = _is_package_available("datasets")
_sentencepiece_available = _is_package_available("sentencepiece")
_soundfile_available = _is_package_available("soundfile")
_tokenizers_available = _is_package_available("tokenizers")
_pyctcdecode_available = _is_package_available("pyctcdecode")
_safetensors_available = _is_package_available("safetensors")
_modelscope_available = _is_package_available("modelscope")
_jieba_available = _is_package_available("jieba")
_pytesseract_available = _is_package_available("pytesseract")
_g2p_en_available = _is_package_available("g2p_en")
_phonemizer_available = _is_package_available("phonemizer")
_mindspore_version, _mindspore_available = _is_package_available(
    "mindspore", return_version=True
)
_sudachipy_available, _sudachipy_version = _is_package_available("sudachipy", return_version=True)

_librosa_available = _is_package_available("librosa")
_scipy_available = _is_package_available("scipy")
_sacremoses_available = _is_package_available("sacremoses")

_pretty_midi_available = importlib.util.find_spec("pretty_midi") is not None
try:
    _pretty_midi_version = importlib_metadata.version("pretty_midi")
    logger.debug(f"Successfully imported pretty_midi version {_pretty_midi_version}")
except importlib_metadata.PackageNotFoundError:
    _pretty_midi_available = False

_essentia_available = importlib.util.find_spec("essentia") is not None
try:
    _essentia_version = importlib_metadata.version("essentia")
    logger.debug(f"Successfully imported essentia version {_essentia_version}")
except importlib_metadata.PackageNotFoundError:
    _essentia_version = False

_levenshtein_available = _is_package_available("Levenshtein")
_nltk_available = _is_package_available("nltk")


_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib.metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib.metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib.metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib.metadata.PackageNotFoundError:
        _faiss_available = False

def is_faiss_available():
    return _faiss_available

def is_levenshtein_available():
    return _levenshtein_available


def is_nltk_available():
    return _nltk_available


def is_einops_available():
    return _einops_available


def is_sudachi_available():
    """
    Checks if SudachiPy is available for use.
    
    Returns:
        None: Indicates whether SudachiPy is available or not.
    
    """
    return _sudachipy_available


def get_sudachi_version():
    '''
    Returns the version of SudachiPy.
    
    Returns:
        None: This function does not take any parameters.
    
    Raises:
        None
    '''
    return _sudachipy_version


def is_bs4_available():
    return _bs4_available

def is_sudachi_projection_available():
    """
    Checks if Sudachi projection is available.
    
    This function checks if Sudachi is available and if the Sudachi version is equal to or greater than 0.6.8.
    
    Returns:
        None
    
    Raises:
        None
    """
    if not is_sudachi_available():
        return False

    # NOTE: We require sudachipy>=0.6.8 to use projection option in sudachi_kwargs for the forwardor of BertJapaneseTokenizer.
    # - `projection` option is not supported in sudachipy<0.6.8, see https://github.com/WorksApplications/sudachi.rs/issues/230
    return version.parse(_sudachipy_version) >= version.parse("0.6.8")

def is_sacremoses_available():
    """
    Checks if the sacremoses library is available in the current environment.
    
    Returns:
        None: Indicates whether the sacremoses library is available or not.
    
    Raises:
        None.
    """
    return _sacremoses_available


def is_mindspore_available():
    '''
    Checks if MindSpore is available.
    
    Args:
        None
    
    Returns:
        None: Indicates that the function does not return any value.
    
    Raises:
        None: No exceptions are raised by this function.
    '''
    return _mindspore_available


def get_mindspore_version():
    """
    Returns the current version of MindSpore.
    
    Args:
    
    Returns:
        None: This function does not take any parameters.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    return _mindspore_version



def is_ftfy_available():
    return _ftfy_available


def is_datasets_available():
    """
    Checks if datasets are available.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    return _datasets_available


def is_sentencepiece_available():
    """
    Checks if SentencePiece library is available.
    
    Returns:
        None: Indicates whether the SentencePiece library is available or not.
    
    Raises:
        None.
    """
    return _sentencepiece_available


def is_tokenizers_available():
    """Check if tokenizers are available.
    
    This function checks if tokenizers are available for use. It does not take any parameters.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    return _tokenizers_available


def is_safetensors_available():
    """
    Checks if SafeTensors is available in the current environment.
    
    Returns:
        None: Indicates whether SafeTensors is available or not.
    
    """
    return _safetensors_available


def is_modelscope_available():
    '''
    Checks if the model scope is available.
    
    Returns:
        None: Indicates whether the model scope is available or not.
    '''
    return _modelscope_available


def is_cython_available():
    """
    Checks if Cython is available in the current environment.
    
    Returns:
        None: Indicates whether Cython is available or not.
    
    Raises:
        None
    """
    return importlib.util.find_spec("pyximport") is not None


def is_protobuf_available():
    """
    Checks if the Google Protocol Buffers (protobuf) library is available.
    
    Returns:
        bool: True if the protobuf library is available, False otherwise.
    
    Raises:
        No specific exceptions are raised by this function.
    """
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_pytest_available():
    """
    Check if the pytest library is available.
    
    Returns:
        None: This function does not return any value.
    
    """
    return _pytest_available


def is_pretty_midi_available():
    """
    Checks if the 'pretty_midi' library is available.
    
    Returns:
        None
    
    Raises:
        None
    """
    return _pretty_midi_available


def is_librosa_available():
    """
    Checks if the 'librosa' library is available.
    
    Returns:
        None
    
    Raises:
        None
    """
    return _librosa_available


def is_essentia_available():
    """
    Checks if the 'essentia' library is available.
    
    Returns:
        None.
    
    Raises:
        None.
    """
    return _essentia_available


def is_pyctcdecode_available():
    """
    Check if the PyCTCDecode library is available.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None
    """
    return _pyctcdecode_available


def is_scipy_available():
    """
    Checks if the SciPy library is available.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    return _scipy_available


def is_jieba_available():
    ''' 
    Checks if the Jieba library is available.
    
    Returns:
        None: The function does not return any value.
    
    '''
    return _jieba_available


def is_pytesseract_available():
    """
    Check if pytesseract library is available.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    return _pytesseract_available


def is_g2p_en_available():
    return _g2p_en_available


def is_tiktoken_available():
    return _tiktoken_available


def is_phonemizer_available():
    return _phonemizer_available


@lru_cache()
def is_vision_available():
    """
    Checks if the Pillow library is available for image processing.
    
    Returns:
        bool: True if Pillow library is available, False otherwise.
    
    Raises:
        PackageNotFoundError: If Pillow or Pillow-SIMD package is not found.
    """
    _pil_available = importlib.util.find_spec("PIL") is not None
    if _pil_available:
        try:
            package_version = importlib_metadata.version("Pillow")
        except importlib_metadata.PackageNotFoundError:
            try:
                package_version = importlib_metadata.version("Pillow-SIMD")
            except importlib_metadata.PackageNotFoundError:
                return False
        logger.debug(f"Detected PIL version {package_version}")
    return _pil_available


def is_in_notebook():
    """
    This function checks if the code is running in a Jupyter notebook environment by examining the current execution environment and relevant environment variables.
    
    Returns:
        bool: Returns True if the code is running in a Jupyter notebook environment, otherwise False.
    
    Raises:
        AttributeError: If an attribute error occurs during the execution of the function.
        ImportError: If the code is running in the console, VS Code, or Databricks environment, respective ImportError with the environment name is raised.
        KeyError: If a key error occurs during the execution of the function.
    """
    try:
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")
        if (
                "DATABRICKS_RUNTIME_VERSION" in os.environ
                and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0"
        ):
            # Databricks Runtime 11.0 and above uses IPython kernel by default so it should be compatible with Jupyter notebook
            # https://docs.microsoft.com/en-us/azure/databricks/notebooks/ipython-kernel
            raise ImportError("databricks")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


# docstyle-ignore
CYTHON_IMPORT_ERROR = """
{0} requires the Cython library but it was not found in your environment. You can install it with pip: `pip install
Cython`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
DATASETS_IMPORT_ERROR = """
{0} requires the 🤗 Datasets library but it was not found in your environment. You can install it with:
```
pip install datasets
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install datasets
```
then restarting your kernel.

Note that if you have a local folder named `datasets` or a local python file named `datasets.py` in your current
working directory, python may try to import this instead of the 🤗 Datasets library. You should rename this folder or
that python file if that's the case. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TOKENIZERS_IMPORT_ERROR = """
{0} requires the 🤗 Tokenizers library but it was not found in your environment. You can install it with:
```
pip install tokenizers
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tokenizers
```
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
MINDSPORE_IMPORT_ERROR = """
{0} requires the MindSpore library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.mindspore.cn/install/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

LIBROSA_IMPORT_ERROR = """
{0} requires thes librosa library. But that was not found in your environment. You can install them with pip:
`pip install librosa`
Please note that you may need to restart your runtime after installation.
"""

ESSENTIA_IMPORT_ERROR = """
{0} requires essentia library. But that was not found in your environment. You can install them with pip:
`pip install essentia==2.1b6.dev1034`
Please note that you may need to restart your runtime after installation.
"""

SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`. Please note that you may need to restart your runtime after installation.
"""

PRETTY_MIDI_IMPORT_ERROR = """
{0} requires thes pretty_midi library. But that was not found in your environment. You can install them with pip:
`pip install pretty_midi`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`. Please note that you may need to restart your runtime after installation.
"""

JIEBA_IMPORT_ERROR = """
{0} requires the jieba library but it was not found in your environment. You can install it with pip: `pip install
jieba`. Please note that you may need to restart your runtime after installation.
"""

VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
G2P_EN_IMPORT_ERROR = """
{0} requires the g2p-en library but it was not found in your environment. You can install it with pip:
`pip install g2p-en`. Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("mindspore", (is_mindspore_available, MINDSPORE_IMPORT_ERROR)),
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("essentia", (is_essentia_available, ESSENTIA_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("pretty_midi", (is_pretty_midi_available, PRETTY_MIDI_IMPORT_ERROR)),
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        ("jieba", (is_jieba_available, JIEBA_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("g2p_en", (is_g2p_en_available, G2P_EN_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    """
    Function to check if the specified backends are available for the given object.
    
    Args:
        obj (object): The object for which backends availability needs to be checked.
        backends (list or tuple or str): The backend(s) to be checked for availability. Can be a single backend as a string or a list/tuple of backends.
    
    Returns:
        None. This function does not return any value.
    
    Raises:
        ImportError: If any of the specified backends are not available for the object.
    """
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """
    def __getattribute__(cls, key):
        """
        This method is called automatically when an attribute is accessed on the 'DummyObject' class or any of its subclasses.
        
        Args:
            cls (type): The class object that the method was called on.
            key (str): The name of the attribute being accessed.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        if key.startswith("_") and key != "_from_config":
            return super().__getattribute__(key)
        requires_backends(cls, cls._backends)


def mindspore_required(func):
    """
    This function decorates another function to require the presence of MindSpore framework. 
    
    Args:
        func (function): The function to be decorated. 
    
    Returns:
        None. The function returns None.
    
    Raises:
        FutureWarning: If the method `torch_required` is deprecated. 
        ImportError: If the decorated function requires MindSpore but MindSpore is not available.
    """
    warnings.warn(
        "The method `torch_required` is deprecated. Use `requires_backends` instead.",
        FutureWarning,
    )

    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_mindspore_available():
            return func(*args, **kwargs)
        raise ImportError(f"Method `{func.__name__}` requires MindSpore.")

    return wrapper


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""
def direct_transformers_import(path: str, file="__init__.py") -> ModuleType:
    """Imports transformers directly

    Args:
        path (`str`): The path to the source file
        file (`str`, optional): The file to join with the path. Defaults to "__init__.py".

    Returns:
        `ModuleType`: The resulting imported module
    """
    name = "mindnlp.transformers"
    location = os.path.join(path, file)
    spec = importlib.util.spec_from_file_location(
        name, location, submodule_search_locations=[path]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module = sys.modules[name]
    return module


def is_soundfile_availble():
    return _soundfile_available
