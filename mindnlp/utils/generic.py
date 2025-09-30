# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Generic utils.
"""
import inspect
from enum import Enum
from collections import OrderedDict, UserDict
from dataclasses import fields
from typing import Any, Tuple, ContextManager, List, TypedDict, Optional
from contextlib import ExitStack
from functools import wraps
import numpy as np
import mindspore


def is_tensor(x):
    """
    Tests if `x` is a `mindspore.Tensor` or `np.ndarray`.
    """
    if isinstance(x, mindspore.Tensor):
        return True

    return isinstance(x, np.ndarray)


def _is_mindspore(x):
    """
    Checks if the input x is a MindSpore tensor.

    Args:
        x (object): The input object to be checked.

    Returns:
        None: This function does not return any value.

    Raises:
        None: This function does not raise any exceptions.
    """
    return isinstance(x, mindspore.Tensor)


def is_mindspore_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    return _is_mindspore(x)


def set_attribute_for_modules(module, key: str, value: Any):
    """
    Set a value to a module and all submodules.
    """
    setattr(module, key, value)
    for submodule in module.children():
        set_attribute_for_modules(submodule, key, value)


def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    use_return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        is_requested_to_return_tuple = kwargs.pop("return_dict", True) is False
        is_configured_to_return_tuple = (
            self.config.use_return_dict is False if hasattr(self, "config") else False
        )

        # The following allows to convert output to tuple ONLY on top level forward call,
        # while internal modules of the model will return Output objects
        # to be able to use name-based attribute access in modeling code.

        # We will check if we are on top level module, if so, turn off to tuple conversion for all
        # underling calls.
        is_top_level_module = getattr(self, "_is_top_level_module", True)
        if is_configured_to_return_tuple and is_top_level_module:
            set_attribute_for_modules(self, "_is_top_level_module", False)

        try:
            output = func(self, *args, **kwargs)
            if is_requested_to_return_tuple or (
                is_configured_to_return_tuple and is_top_level_module
            ):
                output = output.to_tuple()
        finally:
            # Remove the flag after the model forward call is finished.
            # if is_configured_to_return_tuple and is_top_level_module:
            #     del_attribute_from_modules(self, "_is_top_level_module")
            pass

        return output

    return wrapper


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        """
        This method `_missing_` in the class `ExplicitEnum` is a class method used to handle missing values in the ExplicitEnum class.

        Args:
            cls (class): The class itself, used for referring to the class instance inside the method.
            value (any): The value that was not found in the ExplicitEnum class.

        Returns:
            None: This method does not return any value as it raises an exception when called.

        Raises:
            ValueError: If the value provided is not a valid member of the Enum class, a ValueError is raised with a message listing the valid options to choose from.
        """
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    MINDSPORE = "ms"
    NUMPY = "np"


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class LossKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the loss function

    Attributes:
        num_items_in_batch (`int`, *optional*):
            Number of items in the batch. It is recommended to pass it when
            you are doing gradient accumulation.
    """

    num_items_in_batch: Optional[int]


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        """Perform post-initialization actions for the ModelOutput class.

        This method is automatically called after the initialization of a ModelOutput object.

        Args:
            self: An instance of the ModelOutput class.

        Returns:
            None

        Raises:
            ValueError: If the ModelOutput object has no fields or more than one required field.
            ValueError: If a key/value pair in the first field is not a tuple or if it does not follow the format (key, value).
            ValueError: If the key/value pair cannot be set for a given element in the first field.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if len(class_fields) == 0:
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        """
        __delitem__

        Deletes an item from the ModelOutput instance.

        Args:
            self (ModelOutput): The ModelOutput instance from which the item will be deleted.

        Returns:
            None. This method does not return a value.

        Raises:
            RuntimeError: If the '__delitem__' method is attempted to be used on a ModelOutput instance, a RuntimeError is raised with a message indicating that this method cannot be used on the instance.
        """
        raise RuntimeError(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        """
                Sets a default value in the ModelOutput instance.

                Args:
                    self: The ModelOutput instance itself.

                Returns:
                    None. This method does not return any value.

                Raises:
                    RuntimeError: This exception is raised if the method 'setdefault' is called on a ModelOutput instance. The message in the exception states that the 'setdefault' method cannot be used on a
        ModelOutput instance.

                Note:
                    The 'setdefault' method is not supported for ModelOutput instances as it can only be used on dictionary objects.
        """
        raise RuntimeError(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        """
        Method that raises a RuntimeError to prevent the use of 'pop' on a ModelOutput instance.

        Args:
            self (object): The ModelOutput instance on which 'pop' is being called.
                           This parameter is required and represents the current instance of the class.

        Returns:
            None. This method does not return any value.

        Raises:
            RuntimeError: Raised when attempting to use 'pop' method on a ModelOutput instance. The exception message
                          specifies that 'pop' cannot be used on a ModelOutput instance to prevent unintended behavior.
        """
        raise RuntimeError(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        """
        Updates the current instance of the ModelOutput class.

        Args:
            self (ModelOutput): The instance of the ModelOutput class.

        Returns:
            None: This method does not return any value.

        Raises:
            RuntimeError: If the method is called on an instance of the ModelOutput class. This is to prevent using the 'update' method on a ModelOutput instance, as it is not allowed.

        Note:
            The 'update' method is not allowed to be used on a ModelOutput instance. If called, it will raise a RuntimeError.
        """
        raise RuntimeError(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        """
                This method allows accessing the elements of the ModelOutput object using the square bracket notation.

                Args:
                    self (ModelOutput): The instance of the ModelOutput class.
                    k (str or int): The key or index for accessing the element. If k is a string, it is used as a key to retrieve the corresponding value. If k is an integer, it is used as an index to retrieve the
        element.

                Returns:
                    None: This method does not return any value directly. The retrieved value is returned based on the input key or index.

                Raises:
                    TypeError: If the input parameter k is not a string or an integer.
                    KeyError: If the input key k is not found in the internal dictionary when k is a string.
                    IndexError: If the input index k is out of range when k is an integer.
        """
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        return self.to_tuple()[k]

    def __setattr__(self, name, value):
        """
                Method __setattr__ in the class ModelOutput sets the value for the specified attribute name.

                Args:
                    self (object): The instance of the ModelOutput class.
                    name (str): The name of the attribute to be set.
                    value (any): The value to be assigned to the attribute. It can be of any type.

                Returns:
                    None. This method does not return any value.

                Raises:
                    No specific exceptions are raised by this method. However, if the attribute name is not in the keys of the object, it will be added as a new attribute. If the value is None, the attribute will be
        set to None.
        """
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        """
        This method '__setitem__' in the class 'ModelOutput' allows setting key-value pairs in the model output object.

        Args:
            self (ModelOutput): The instance of the ModelOutput class.
            key (Any): The key to be set in the model output object.
            value (Any): The value corresponding to the key to be set in the model output object.

        Returns:
            None. This method does not return any value explicitly.

        Raises:
            This method may raise the following exceptions:
            - TypeError: If the key is not of a valid type.
            - ValueError: If the value is not acceptable for the given key.
            - Other exceptions related to the internal implementation of the ModelOutput class.
        """
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(v for _, v in self.items())


# vendored from distutils.util
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        """
        Method '__get__' in the class 'cached_property'.

        Args:
            self (object): The current instance of the class.
            obj (object): The object on which the method is being called.
            objtype (object): The type of the object, if available. Defaults to None.

        Returns:
            None: The method returns a value of type None.

        Raises:
            AttributeError: If the attribute is unreadable, this exception is raised.
        """
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


def _is_numpy(x):
    """
    This function checks if the input is a NumPy array.

    Args:
        x (any): The input to be checked for being a NumPy array.

    Returns:
        None: This function does not return a value.

    Raises:
        None
    """
    return isinstance(x, np.ndarray)


def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    return _is_numpy(x)


def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'mindspore."):
        return "ms"
    if representation.startswith("<class 'numpy."):
        return "np"


def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "ms": is_mindspore_tensor,
        "np": is_numpy_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    # We will test this one first, then numpy, then the others.
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend(
        [f for f in framework_to_test if f not in [preferred_framework, "np"]]
    )
    return {f: framework_to_test[f] for f in frameworks}


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    framework_to_py_obj = {
        "ms": lambda obj: obj.asnumpy().tolist(),
        "np": lambda obj: obj.tolist(),
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_py_obj[framework](obj)

    # tolist also works on 0d np arrays
    if isinstance(obj, np.number):
        return obj.tolist()
    return obj


def to_numpy(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array.
    """
    framework_to_numpy = {
        "ms": lambda obj: obj.asnumpy(),
        "np": lambda obj: obj,
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return np.array(obj)

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        """
        __init__

        Args:
            self: The instance of the class.
            context_managers (List[ContextManager]): A list of context managers to be initialized.

        Returns:
            None: This method does not return any value.

        Raises:
            No specific exceptions are raised by this method.
        """
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        """
                Method '__enter__' in the class 'ContextManagers'.

                Args:
                    self (object): The instance of the ContextManagers class on which the method is called. It is used to access the instance attributes and methods.

                Returns:
                    None. This method does not return any value explicitly, it performs context management operations within the class.

                Raises:
                    This method may raise exceptions if the context managers encountered during the iteration in the for loop raise any exceptions. Ensure proper error handling is in place to catch and handle any
        exceptions that may occur during the context management operations.
        """
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        """
        __exit__

        Method in the class ContextManagers.

        Args:
            self: (object) The instance of the class.

        Returns:
            None: This method does not return any value.

        Raises:
            This method does not explicitly raise any exceptions.
        """
        self.stack.__exit__(*args, **kwargs)


def find_labels(model_class):
    """
    Find the labels used by a given model.

    Args:
        model_class (`type`): The class of the model.
    """
    model_name = model_class.__name__
    signature = inspect.signature(model_class.forward)  # TensorFlow models

    if "QuestionAnswering" in model_name:
        return [
            p
            for p in signature.parameters
            if "label" in p or p in ("start_positions", "end_positions")
        ]
    else:
        return [p for p in signature.parameters if "label" in p]


def can_return_loss(model_class):
    """
    Check if a given model can return loss.

    Args:
        model_class (`type`): The class of the model.
    """
    signature = inspect.signature(model_class.forward)  # TensorFlow models

    for p in signature.parameters:
        if p == "return_loss" and signature.parameters[p].default is True:
            return True

    return False
