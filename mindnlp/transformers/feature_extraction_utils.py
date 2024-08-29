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
"""
 Feature extraction saving/loading class for common feature extractors.
"""

import copy
import json
import os
import warnings
from collections import UserDict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from mindnlp.configs import FEATURE_EXTRACTOR_NAME
from mindnlp.utils import (
    TensorType,
    cached_file,
    download_url,
    is_numpy_array,
    is_offline_mode,
    is_remote_url,
    is_mindspore_available,
    logging,
    requires_backends,
)


if is_mindspore_available():
    import mindspore
    from mindspore import ops


logger = logging.get_logger(__name__)

PreTrainedFeatureExtractor = Union["SequenceFeatureExtractor"]  # noqa: F821


class BatchFeature(UserDict):
    r"""
    Holds the output of the [`~SequenceFeatureExtractor.pad`] and feature extractor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        """
        Initializes a new instance of the BatchFeature class.
        
        Args:
            self: The instance of the BatchFeature class.
            data (Optional[Dict[str, Any]]): A dictionary containing the data for the batch feature. Defaults to None.
                If provided, it should be a dictionary with string keys and values of any type.
            tensor_type (Union[None, str, TensorType]): The type of tensor to convert the data to. Defaults to None.
                It can be None, a string, or an instance of the TensorType class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def __getitem__(self, item: str) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):
            return self.data[item]
        raise KeyError("Indexing with integers is not available when using Python based feature extractors")

    def __getattr__(self, item: str):
        """
        This method allows access to attributes of the BatchFeature class that are not directly defined.
        
        Args:
            self: Instance of the BatchFeature class.
                - Type: BatchFeature
                - Purpose: Represents the current instance of the class.
                - Restrictions: Must be an instance of the BatchFeature class.
        
            item: The attribute name being accessed.
                - Type: str
                - Purpose: Indicates the name of the attribute to retrieve.
                - Restrictions: Must be a string representing a valid attribute name.
        
        Returns:
            None:
                If the attribute 'item' does not exist in the data dictionary of the BatchFeature instance.
        
        Raises:
            AttributeError: Raised when the specified 'item' attribute does not exist in the data dictionary of the
                BatchFeature instance.
        """
        try:
            return self.data[item]
        except KeyError as exc:
            raise AttributeError from exc

    def __getstate__(self):
        """
        
        Description:
            This method is used to retrieve the state of an instance of the 'BatchFeature' class. It returns a
            dictionary containing the 'data' attribute of the instance.

        Args:
            self: Represents the instance of the 'BatchFeature' class. It is automatically passed as the first
                argument when the method is called.

        Returns:
            None: This method does not explicitly return a value.
                However, it implicitly returns a dictionary containing the 'data' attribute of the instance.

        Raises:
            None.

        """
        return {"data": self.data}

    def __setstate__(self, state):
        """
        Sets the state of the BatchFeature object.

        Args:
            self (BatchFeature): The BatchFeature object to set the state for.
            state (dict): A dictionary containing the state information to be set. It should include the 'data' key.

        Returns:
            None.

        Raises:
            None.

        This method is automatically called by the pickle module when unpickling a BatchFeature object.
        It sets the state of the object based on the provided 'state' dictionary.
        The 'data' key in the 'state' dictionary is used to set the 'data' attribute of the BatchFeature object.
        """
        if "data" in state:
            self.data = state["data"]

    # Copied from transformers.tokenization_utils_base.BatchEncoding.keys
    def keys(self):
        '''
        Method: keys

        Description:
            This method returns a list of all the keys in the BatchFeature's data dictionary.

        Args:
            self: (object) The instance of the BatchFeature class.

        Returns:
            None: This method does not return any value, as it directly operates on the instance's data.

        Raises:
            None.
        '''
        return self.data.keys()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.values
    def values(self):
        """
        Retrieves and returns the values stored in the 'data' attribute of the BatchFeature class instance.

        Args:
            self (BatchFeature): The current instance of the BatchFeature class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.data.values()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.items
    def items(self):
        """
        Method items in class BatchFeature.

        Args:
            self: BatchFeature object
                The self parameter refers to the instance of the BatchFeature class.

        Returns:
            dict_items
                Returns a view object that displays a list of a given dictionary's (key, value) tuple pairs.

        Raises:
            None
        """
        return self.data.items()

    def _get_is_as_tensor_fns(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        This method '_get_is_as_tensor_fns' in the class 'BatchFeature' is responsible for returning functions
        for checking if a given input is a tensor and converting values to tensors based on the specified
        tensor type.

        Args:
            self: An instance of the class BatchFeature.
            tensor_type: An optional parameter indicating the type of tensor to be used for conversion.
                It can be a string or an instance of the enum TensorType. If not provided, the default value is None.

        Returns:
            tuple:
                A tuple containing two functions:

                - is_tensor: A function that checks whether a given value is a tensor.
                - as_tensor: A function that converts a given value to a tensor based on the specified tensor_type.

        Raises:
            ImportError: If the tensor_type is set to MindSpore but MindSpore is not installed.
        """
        if tensor_type is None:
            return None, None

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.MINDSPORE:
            if not is_mindspore_available():
                raise ImportError("Unable to convert output to MindSpore tensors format, MindSpore is not installed.")

            def as_tensor(value):
                if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    value = np.array(value)
                return mindspore.tensor(value)

            is_tensor = ops.is_tensor
        else:
            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                elif isinstance(value, mindspore.Tensor):
                    return value.asnumpy()
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array
        return is_tensor, as_tensor

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    self[key] = tensor
            except Exception as exc:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ") from exc
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                ) from exc
        return self

    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["mindspore"])

        new_data = {}
        # Check if the args are a device or a dtype
        if len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if isinstance(arg, mindspore._c_expression.typing.Type):
                # The first argument is a dtype
                pass
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            # check if v is a floating point
            if ops.is_floating_point(v):
                # cast and send to device
                new_data[k] = v.to(*args, **kwargs)
            else:
                new_data[k] = v
        self.data = new_data
        return self


class FeatureExtractionMixin():
    """
    This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """
    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Pop "processor_class" as it should be saved as private attribute
        self._processor_class = kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        r"""
        Instantiate a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a feature extractor, *e.g.* a
        derived class of [`SequenceFeatureExtractor`].

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  hf-mirror.com. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on hf-mirror.com, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final feature extractor object. If `True`, then this
                functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            A feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`].

        Example:
            ```python
            >>> # We can't instantiate directly the base class *FeatureExtractionMixin* nor *SequenceFeatureExtractor* so let's show the examples on a
            >>> # derived class: *Wav2Vec2FeatureExtractor*
            >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            >>>     "facebook/wav2vec2-base-960h"
            >>> )  # Download feature_extraction_config from hf-mirror.com and cache.
            >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            >>>     "./test/saved_model/"
            >>> )  # E.g. feature_extractor (or model) was saved using *save_pretrained('./test/saved_model/')*
            >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./test/saved_model/preprocessor_config.json")
            >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            >>>     "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False
            >>> )
            >>> assert feature_extractor.return_attention_mask is False
            >>> feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained(
            >>>     "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False, return_unused_kwargs=True
            >>> )
            >>> assert feature_extractor.return_attention_mask is False
            >>> assert unused_kwargs == {"foo": False}
            ```
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a feature_extractor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        # If we save using the predefined names, we can load using `from_pretrained`
        output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)

        self.to_json_file(output_feature_extractor_file)
        logger.info(f"Feature extractor saved in {output_feature_extractor_file}")

        return [output_feature_extractor_file]

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        revision = kwargs.pop('revision', 'main')
        mirror = kwargs.pop('mirror', 'huggingface')

        user_agent = {"file_type": "feature extractor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_feature_extractor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            feature_extractor_file = pretrained_model_name_or_path
            resolved_feature_extractor_file = download_url(pretrained_model_name_or_path)
        else:
            feature_extractor_file = FEATURE_EXTRACTOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_feature_extractor_file = cached_file(
                    pretrained_model_name_or_path,
                    feature_extractor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    user_agent=user_agent,
                    revision=revision,
                    mirror=mirror,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception as exc:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load feature extractor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://hf-mirror.com/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {FEATURE_EXTRACTOR_NAME} file"
                ) from exc

        try:
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)

        except json.JSONDecodeError as exc:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_feature_extractor_file}' is not a valid JSON file."
            ) from exc

        if is_local:
            logger.info(f"loading configuration file {resolved_feature_extractor_file}")
        else:
            logger.info(
                f"loading configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
            )

        return feature_extractor_dict, kwargs

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor:
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]:
                The feature extractor object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "window" in output:
            del output["window"]
        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedFeatureExtractor:
        """
        Instantiates a feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] from the path to
        a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`]:
                The feature_extractor object instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
        return cls(**feature_extractor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        """
        This method '__repr__' in the 'FeatureExtractionMixin' class generates a string representation of the object.
        
        Args:
            self (object): The instance of the class.
                This parameter refers to the current instance of the class.
        
        Returns:
            None:
                This method does not return any value explicitly;
                it generates a formatted string representation of the object.
        
        Raises:
            None.
        """
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoFeatureExtractor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoFeatureExtractor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoFeatureExtractor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import mindnlp.transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class
