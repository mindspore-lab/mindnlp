# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Factory function to build auto-model classes."""
import copy
import importlib
from collections import OrderedDict

from mindnlp.utils import logging
from .configuration_auto import AutoConfig, model_type_to_module_name
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


def _get_model_class(config, model_mapping):
    """
    This function retrieves the model class based on the provided configuration and model mapping.
    
    Args:
        config (object): The configuration object used to determine the model class. Must be of a supported type.
        model_mapping (dict): A mapping of configuration types to supported model classes.
    
    Returns:
        object: The model class selected based on the configuration and model mapping.
    
    Raises:
        None.
    
    Note:
        If the supported_models value in the model_mapping is not a list or tuple, it is returned directly.
        Otherwise, the function searches for the model class based on the 'architectures' attribute of the config object.
        The first architecture found in the mapping is returned. If no architectures are found, the first supported model is returned.
    """
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class _BaseAutoModelClass:

    """
    _BaseAutoModelClass is a base class for AutoModels that provides methods for instantiating models from configurations or pretrained models. 
    
    The class includes methods for creating instances from configurations, pretrained models, and for registering new models within the class. 
    
    Methods:
        from_config(cls, config, **kwargs): Instantiate a model from a configuration object.
        from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs): Instantiate a model from a pretrained model or model path.
        register(cls, config_class, model_class, exist_ok=False): Register a new model for the class based on a configuration and model class pair.
    
    For more details on each method's parameters and usage, refer to the method's specific documentation.
    
    Note: This class is intended to be inherited from and customized for specific auto-model implementations.
    """
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of the _BaseAutoModelClass class.
        
        Args:
            self:
                The instance of the class.

                - Type: _BaseAutoModelClass
                - Purpose: Represents the current instance of the class.
        
        Returns:
            None.
        
        Raises:
            EnvironmentError: If the __init__ method is called directly. 
            The error message will specify that instances of _BaseAutoModelClass should be instantiated using the 
            'from_pretrained(pretrained_model_name_or_path)' or 'from_config(config)' methods.
        """
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Converts a configuration object into an instance of the current `_BaseAutoModelClass` class.
        
        Args:
            cls (type): The class object for the `_BaseAutoModelClass` class.
            config: The configuration object that needs to be converted.
                It should be of a type that is recognized by the `_model_mapping` dictionary.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the `config` object is not recognized as a valid configuration class for this `AutoModel` class.
                The error message will indicate the unrecognized configuration class and the expected valid configuration classes,
                which are determined by the `_model_mapping` dictionary.
        
        Note:
            1. The `config` parameter should be of a type that is present in the `_model_mapping` dictionary.
                The `_model_mapping` dictionary maps configuration types to model classes.
            2. This method is a class method, denoted by the `@classmethod` decorator.
                It can be called directly on the class object without needing to create an instance of the class.
            3. The `_from_config` method is called on the appropriate model class, determined by the `_model_mapping` dictionary,
                to perform the conversion from the configuration object to an instance of the model class.
        """
        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Performs a series of operations to load a pretrained model from either a local file or a remote repository.
        
        Args:
            cls (class): The class object that the method is called on.
            pretrained_model_name_or_path (str): The name or path of the pretrained model.
                It can be a local file path or a remote repository URL.
        
        Returns:
            None
        
        Raises:
            ValueError: If the configuration class of the pretrained model is not recognized by the AutoModel class.
        
        Note:
            This method is a class method, meaning it can be called on the class object itself without instantiation.
        
        Example:
            ```python
            >>> _BaseAutoModelClass.from_pretrained("bert-base-uncased")
            ```

        In the provided code snippet, the method `from_pretrained` is a class method of the `_BaseAutoModelClass` class.
        It is used to load a pretrained model by either specifying its name or providing the path to the model file.
        The method performs various operations to correctly configure and initialize the model.

        The `cls` parameter is a reference to the class object itself. It is automatically passed when calling the method on the class.

        The `pretrained_model_name_or_path` parameter is a string that represents the name or path of the pretrained model.
        It can be either a local file path or a remote repository URL.
        This parameter is required to specify the pretrained model to load.

        The method does not return any value, as indicated by the `None` return type.

        During the execution of the method, the following steps are performed:
        >   1. The `config` parameter is obtained from the `kwargs` dictionary. If `config` is not provided, it is set to `None`.
        >   2. The values of `from_pt`, `mirror`, `revision`, and `token` are obtained from the `kwargs` dictionary using the `get` method.
        If any of these parameters is not provided, default values are used.
        >   3. If the `config` parameter is not an instance of `PretrainedConfig`, the `kwargs` dictionary is deep copied to `kwargs_orig`.
        >   4. If the `ms_dtype` parameter is set to `'auto'` in `kwargs`, it is removed from `kwargs`.
        >   5. If the `quantization_config` parameter is not `None` in `kwargs`, it is removed from `kwargs`.
        >   6. The `AutoConfig.from_pretrained` method is called with `pretrained_model_name_or_path` and the remaining `kwargs` as arguments.
        The return values are assigned to `config` and `kwargs`.
        >   7. If the `torch_dtype` parameter is set to `'auto'` in `kwargs_orig`, it is added to `kwargs` with the same value.
        >   8. If the `quantization_config` parameter is not `None` in `kwargs_orig`, it is added to `kwargs` with the same value.
        >   9. The `token`, `mirror`, and `revision` parameters are added to `kwargs` with their respective values obtained earlier.
        >   10. If the type of `config` is one of the keys in `_model_mapping` of the class,
        the corresponding model class is obtained using `_get_model_class` function with `config` and `_model_mapping` as  arguments.
        >   11. The `from_pretrained` method of the obtained `model_class` is called with `pretrained_model_name_or_path`,
        `model_args`, `config`, and the updated `kwargs` as arguments.
        >   12. If the type of `config` is not recognized, a `ValueError` is raised with an informative error message.

        Note that this docstring assumes the presence of additional helper functions and variables that are not included in the provided code snippet.
        It is recommended to refer to the complete implementation for a comprehensive understanding.
        """
        config = kwargs.pop("config", None)
        _ = kwargs.get('from_pt', True)
        mirror = kwargs.get('mirror', 'huggingface')
        revision = kwargs.get('revision', 'main')
        token = kwargs.get('token', None)
        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("ms_dtype", None) == "auto":
                _ = kwargs.pop("ms_dtype")
            # to not overwrite the quantization_config if config has a quantization_config
            if kwargs.get("quantization_config", None) is not None:
                _ = kwargs.pop("quantization_config")
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                **kwargs,
            )
            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"
            if kwargs_orig.get("quantization_config", None) is not None:
                kwargs["quantization_config"] = kwargs_orig["quantization_config"]

        kwargs['token'] = token
        kwargs['mirror'] = mirror
        kwargs['revision'] = revision
        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(model_class, "config_class") and model_class.config_class != config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)


def insert_head_doc(docstring, head_doc=""):
    """
    Inserts a specified 'head_doc' into the provided 'docstring' to modify the description of a model class in the library.
    
    Args:
        docstring (str): The original docstring that describes a model class in the library.
        head_doc (str): The head type to insert into the docstring. It is used to modify the description of the model class.
    
    Returns:
        None.
    
    Raises:
        None.
    """
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    return docstring.replace(
        "one of the model classes of the library ", "one of the base model classes of the library "
    )


def get_values(model_mapping):
    """
    This function takes a dictionary called 'model_mapping' as a parameter and returns a list of values from the dictionary. 
    
    Args:
        model_mapping (dict): A dictionary that maps keys to values. The values can be either a single object or a list/tuple of objects.
    
    Returns:
        list: A list containing all the values from the 'model_mapping' dictionary. If a value is a list or tuple, its elements are included in the final list.
    
    Raises:
        None.
    
    """
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result


def getattribute_from_module(module, attr):
    """
    This function retrieves an attribute from a given module.
    
    Args:
        module (module): The module from which to retrieve the attribute.
        attr (str or tuple): The name of the attribute to retrieve. If a tuple is provided, the function will recursively retrieve each attribute specified in the tuple.
    
    Returns:
        None: If the attribute is None or cannot be found in the module.
    
    Raises:
        ValueError: If the attribute is not found in the provided module or any of its dependencies.
    """
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("mindnlp.transformers")

    if module != transformers_module:
        try:
            return getattribute_from_module(transformers_module, attr)
        except ValueError as exc:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {transformers_module}!") from exc
    else:
        raise ValueError(f"Could not find {attr} in {transformers_module}!")


class _LazyAutoMapping(OrderedDict):
    """
    A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        config_mapping: The map model type to config class
        model_mapping: The map model type to model (or tokenizer) class
    """
    def __init__(self, config_mapping, model_mapping):
        """
        Initializes a new instance of the _LazyAutoMapping class.
        
        Args:
            self: The instance of the _LazyAutoMapping class.
            config_mapping (dict): A dictionary that represents the mapping of configuration values.
            model_mapping (dict): A dictionary that represents the mapping of model values.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self):
        """
        Returns the length of the _LazyAutoMapping object.
        
        Args:
            self (_LazyAutoMapping): The instance of the _LazyAutoMapping class.
                The self parameter is automatically passed when calling this method.
        
        Returns:
            int: The total number of common keys between the _config_mapping and _model_mapping dictionaries,
                plus the number of elements in the _extra_content list.
        
        Raises:
            None.
        
        Note:
            This method does not modify the _LazyAutoMapping object.
        """
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key):
        """
        Args:
            self (object): The instance of the '_LazyAutoMapping' class.
            key (object): The key used to retrieve the value from the mapping. It should be a valid key present in the mapping.
        
        Returns:
            None: This method does not explicitly return any value. The retrieved value is returned based on the key provided.
        
        Raises:
            KeyError: If the provided key is not found in the mapping, a KeyError is raised.
        """
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        """
        Load attribute from module based on model type.
        
        Args:
            self (_LazyAutoMapping): The instance of the _LazyAutoMapping class.
            model_type (str): The type of the model for which the attribute needs to be loaded.
            attr (str): The attribute to be loaded from the module.
        
        Returns:
            None.
        
        Raises:
            ModuleNotFoundError: If the specified module for the model type is not found.
            AttributeError: If the specified attribute is not found in the module for the model type.
        """
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "mindnlp.transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        """
        This method retrieves the keys from the _LazyAutoMapping instance.
        
        Args:
            self (_LazyAutoMapping): The instance of the _LazyAutoMapping class.
            
        Returns:
            list: A list of keys from the _LazyAutoMapping instance.
        
        Raises:
            None
        """
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        """
        This method retrieves the value associated with the specified key from the _LazyAutoMapping instance, and returns a default value if the key is not present.
        
        Args:
            self (_LazyAutoMapping): The instance of the _LazyAutoMapping class.
            key (any): The key whose associated value is to be retrieved.
            default (any): The default value to be returned if the key is not present in the mapping.
        
        Returns:
            If the key is present in the mapping, the method returns the value associated with the key.
            If the key is not present, the method returns the specified default value.
        
        Raises:
            KeyError: If the specified key is not present in the mapping, the method raises a KeyError.
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        """
        This method '__bool__' in the class '_LazyAutoMapping' returns a boolean value indicating whether the mapping object has any keys.
        
        Args:
            self (object):
                The instance of the '_LazyAutoMapping' class.
                It represents the mapping object for which the method is being called.
        
        Returns:
            bool:
                A boolean value indicating whether the mapping object has any keys.
                Returns True if the mapping object has keys, otherwise False.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        return bool(self.keys())

    def values(self):
        """
        Method 'values' in the class '_LazyAutoMapping' retrieves values from the mapping and extra content.
        
        Args:
            self: The reference to the current instance of the class '_LazyAutoMapping'.
            
        Returns:
            A list of values retrieved from the mapping and extra content. The values are of type 'None'.
        
        Raises:
            No exceptions are raised by this method.
        """
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        """
        items(self)
            This method returns a list of tuples representing the mapping items between the config and model mappings along with any extra content.
        
        Args:
            self (object): The instance of the _LazyAutoMapping class.
        
        Returns:
            list: A list of tuples representing the mapping items between the config and model mappings along with any extra content.
        
        Raises:
            None
        """
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        """
        Method '__iter__' in the class '_LazyAutoMapping'.
        
        Args:
            self (object): The instance of the '_LazyAutoMapping' class.
                Represents the current instance of the class for which the iterator is being generated.
        
        Returns:
            None: This method returns an iterator over the keys of the instance.
        
        Raises:
            None.
        """
        return iter(self.keys())

    def __contains__(self, item):
        ''' 
            This method checks if an item is present in the '_extra_content' attribute or if the item's name is present
            in the '_reverse_config_mapping' attribute of the '_LazyAutoMapping' class instance.
        
            Args:
                self (_LazyAutoMapping): The instance of the '_LazyAutoMapping' class.
                item (Any):
                    The item to be checked for presence in the '_extra_content' attribute
                    or the item's name in the '_reverse_config_mapping' attribute.
        
            Returns:
                bool:
                    Returns True if the item is found in the '_extra_content' attribute
                    or if the item's name is found in the '_reverse_config_mapping' attribute. Returns False otherwise.
        
            Raises:
                None.
        '''
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value, exist_ok=False):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        self._extra_content[key] = value
