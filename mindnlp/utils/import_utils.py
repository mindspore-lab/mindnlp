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
# pylint: disable=missing-function-docstring
# pylint: disable=logging-fstring-interpolation
# pylint: disable=inconsistent-return-statements
# pylint: disable=wrong-import-position
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import os
import sys
import warnings
import importlib
import importlib.util
from types import ModuleType
from collections import OrderedDict
from functools import wraps
from typing import Tuple, Union, Any
from itertools import chain

if sys.version_info >= (3, 8):
    # For Python 3.8 and later
    from importlib import metadata as importlib_metadata
else:
    # For Python versions earlier than 3.8
    import importlib_metadata

from . import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _is_package_available(
    pkg_name: str, return_version: bool = False
) -> Union[Tuple[bool, str], bool]:
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


_pytest_available = _is_package_available("pytest")
_datasets_available = _is_package_available("datasets")
_sentencepiece_available = _is_package_available("sentencepiece")
_tokenizers_available = _is_package_available("tokenizers")
_safetensors_available = _is_package_available("safetensors")
_modelscope_available = _is_package_available("modelscope")
_mindspore_version, _mindspore_available = _is_package_available(
    "mindspore", return_version=True
)


def is_mindspore_available():
    return _mindspore_available


def get_mindspore_version():
    return _mindspore_version


def is_datasets_available():
    return _datasets_available


def is_sentencepiece_available():
    return _sentencepiece_available


def is_tokenizers_available():
    return _tokenizers_available

def is_safetensors_available():
    return _safetensors_available

def is_modelscope_available():
    return _modelscope_available

def is_cython_available():
    return importlib.util.find_spec("pyximport") is not None

def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_pytest_available():
    return _pytest_available


def is_in_notebook():
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

BACKENDS_MAPPING = OrderedDict(
    [
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
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
        if key.startswith("_") and key != "_from_config":
            return super().__getattribute__(key)
        requires_backends(cls, cls._backends)


def mindspore_required(func):
    warnings.warn(
        "The method `torch_required` is deprecated and will be removed in v4.36. Use `requires_backends` instead.",
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
    name = "transformers"
    location = os.path.join(path, file)
    spec = importlib.util.spec_from_file_location(
        name, location, submodule_search_locations=[path]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module = sys.modules[name]
    return module


class _LazyModule(ModuleType):
    """
    _LazyModule 类是一个懒加载模块，用于按需导入模块和对象。
    """

    # 构造函数，初始化懒加载模块
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # 用于在 IDE 中自动完成
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # 用于在 IDE 中自动完成
    def __dir__(self):
        result = super().__dir__()
        # 只添加不在 dir 中的 self.__all__ 元素，因为有些可能已经被访问
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    # 获取属性时的动态导入
    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    # 根据模块名执行实际导入
    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as exception:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{exception}"
            ) from exception

    # 用于序列化
    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
