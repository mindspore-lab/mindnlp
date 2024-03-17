# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
# pylint: disable=invalid-name
# pylint: disable=no-else-return
# pylint: disable=no-else-raise
# pylint: disable=unspecified-encoding
# pylint: disable=redefined-builtin
# pylint: disable=missing-class-docstring
"""pipeline base"""

import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from os.path import abspath, exists
from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore
from mindspore import ops
from mindspore.dataset import Dataset

from mindnlp.utils import (
    ModelOutput,
    is_mindspore_available,
    logging,
)
from ..feature_extraction_utils import PreTrainedFeatureExtractor
# from ..image_processing_utils import BaseImageProcessor
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer

from ..models.auto.modeling_auto import AutoModel
from ..modeling_utils import PreTrainedModel

GenericTensor = Union[List["GenericTensor"], "mindspore.Tensor"]

logger = logging.get_logger(__name__)


def no_collate_fn(items):
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


def _pad(items, key, padding_value, padding_side):
    batch_size = len(items)
    if isinstance(items[0][key], mindspore.Tensor):
        # Others include `attention_mask` etc...
        shape = items[0][key].shape
        dim = len(shape)
        if key in ["pixel_values", "image"]:
            # This is probable image so padding shouldn't be necessary
            # B, C, H, W
            return ops.cat([item[key] for item in items], axis=0)
        elif dim == 4 and key == "input_features":
            # this is probably a mel spectrogram batched
            return ops.cat([item[key] for item in items], axis=0)
        max_length = max(item[key].shape[1] for item in items)
        min_length = min(item[key].shape[1] for item in items)
        dtype = items[0][key].dtype

        if dim == 2:
            if max_length == min_length:
                # Bypass for `ImageGPT` which doesn't provide a padding value, yet
                # we can consistently pad since the size should be matching
                return ops.cat([item[key] for item in items], axis=0)
            tensor = ops.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            tensor = ops.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value
        elif dim == 4:
            tensor = ops.zeros((batch_size, max_length, shape[-2], shape[-1]), dtype=dtype) + padding_value

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].copy()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].copy()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].copy()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].copy()
            elif dim == 4:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :, :] = item[key][0].copy()
                else:
                    tensor[i, : len(item[key][0]), :, :] = item[key][0].copy()

        return tensor
    else:
        return [item[key] for item in items]


def pad_collate_fn(tokenizer, feature_extractor):
    # Tokenizer
    t_padding_side = None
    # Feature extractor
    f_padding_side = None
    if tokenizer is None and feature_extractor is None:
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor can be images, where no padding is expected
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)

    if t_padding_side is not None and f_padding_side is not None and t_padding_side != f_padding_side:
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    padding_side = "right"
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side

    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                    f" {keys})"
                )
        # input_values, input_pixels, input_ids, ...
        padded = {}
        for key in keys:
            if key in {"input_ids"}:
                # ImageGPT uses a feature extractor
                if tokenizer is None and feature_extractor is not None:
                    _padding_value = f_padding_value
                else:
                    _padding_value = t_padding_value
            elif key in {"input_values", "pixel_values", "input_features"}:
                _padding_value = f_padding_value
            elif key in {"p_mask", "special_tokens_mask"}:
                _padding_value = 1
            elif key in {"attention_mask", "token_type_ids"}:
                _padding_value = 0
            else:
                # This is likely another random key maybe even user provided
                _padding_value = 0
            padded[key] = _pad(items, key, _padding_value, padding_side)
        return padded

    return inner


def load_model(
    model,
    config: AutoConfig,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    **model_kwargs,
):
    """
    Select framework (TensorFlow or PyTorch) to use from the `model` passed. Returns a tuple (framework, model).

    If `model` is instantiated, this function will just infer the framework from the model class. Otherwise `model` is
    actually a checkpoint name and this method will try to instantiate it using `model_classes`. Since we don't want to
    instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for `model`, PyTorch is selected.

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to infer the framework from. If `str`, a checkpoint name. The model to infer the framewrok from.
        config ([`AutoConfig`]):
            The config associated with the model to help using the correct class
        model_classes (dictionary `str` to `type`, *optional*):
            A mapping framework to class.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        `Tuple`: A tuple framework, model.
    """
    if not is_mindspore_available():
        raise RuntimeError(
            "MindSpore should be installed. "
            "To install MindSpore, read the instructions at https://www.mindspore.cn/."
        )
    if isinstance(model, str):
        # model_kwargs["_from_pipeline"] = task
        class_tuple = ()
        if model_classes:
            class_tuple = class_tuple + model_classes.get("ms", (AutoModel,))
        if config.architectures:
            classes = []
            for architecture in config.architectures:
                transformers_module = importlib.import_module("mindnlp.transformers")
                _class = getattr(transformers_module, architecture, None)
                if _class is not None:
                    classes.append(_class)
            class_tuple = class_tuple + tuple(classes)

        if len(class_tuple) == 0:
            raise ValueError(f"Pipeline cannot infer suitable model classes from {model}")

        all_traceback = {}
        for model_class in class_tuple:
            kwargs = model_kwargs.copy()
            try:
                model = model_class.from_pretrained(model, **kwargs)
                model = model.set_train(False)
                # Stop loading on the first successful load.
                break
            except (OSError, ValueError):
                all_traceback[model_class.__name__] = traceback.format_exc()
                continue

        if isinstance(model, str):
            error = ""
            for class_name, trace in all_traceback.items():
                error += f"while loading with {class_name}, an error is thrown:\n{trace}\n"
            raise ValueError(
                f"Could not load model {model} with any of the following classes: {class_tuple}. See the original errors:\n\n{error}\n"
            )

    return model


def from_model(
    model,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    **model_kwargs,
):
    """
    Select framework (TensorFlow or PyTorch) to use from the `model` passed. Returns a tuple (framework, model).

    If `model` is instantiated, this function will just infer the framework from the model class. Otherwise `model` is
    actually a checkpoint name and this method will try to instantiate it using `model_classes`. Since we don't want to
    instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for `model`, PyTorch is selected.

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to infer the framework from. If `str`, a checkpoint name. The model to infer the framewrok from.
        model_classes (dictionary `str` to `type`, *optional*):
            A mapping framework to class.
        task (`str`):
            The task defining which pipeline will be returned.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        `Tuple`: A tuple framework, model.
    """
    if isinstance(model, str):
        config = AutoConfig.from_pretrained(model, _from_pipeline=task, **model_kwargs)
    else:
        config = model.config
    return load_model(
        model, config, model_classes=model_classes, _from_pipeline=task, task=task, **model_kwargs
    )


def get_default_model_and_revision(
    targeted_task: Dict, task_options: Optional[Any]
) -> Union[str, Tuple[str, str]]:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        `str` The model string representing the default model for this pipeline
    """
    framework = "ms"

    defaults = targeted_task["default"]
    if task_options:
        if task_options not in defaults:
            raise ValueError(f"The task does not provide any default models for options {task_options}")
        default_models = defaults[task_options]["model"]
    elif "model" in defaults:
        default_models = targeted_task["default"]["model"]
    else:
        # parametrized
        raise ValueError('The task defaults can\'t be correctly selected. You probably meant "translation_XX_to_YY"')

    return default_models[framework]


class PipelineException(Exception):
    """
    Raised by a [`Pipeline`] when handling __call__.

    Args:
        task (`str`): The task of the pipeline.
        model (`str`): The model used by the pipeline.
        reason (`str`): The error message to display.
    """

    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each [`~pipelines.Pipeline`].
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing. Supported data formats
    currently includes:

    - JSON
    - CSV
    - stdin/stdout (pipe)

    `PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets columns to
    pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite: bool = False,
    ):
        self.output_path = output_path
        self.input_path = input_path
        self.column = column.split(",") if column is not None else [""]
        self.is_multi_columns = len(self.column) > 1

        if self.is_multi_columns:
            self.column = [tuple(c.split("=")) if "=" in c else (c, c) for c in self.column]

        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError(f"{self.output_path} already exists on disk")

        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError(f"{self.input_path} doesnt exist on disk")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`dict` or list of `dict`): The data to store.
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (`dict` or list of `dict`): The data to store.

        Returns:
            `str`: Path where the data has been saved.
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        return binary_path

    @staticmethod
    def from_str(
        format: str,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ) -> "PipelineDataFormat":
        """
        Creates an instance of the right subclass of [`~pipelines.PipelineDataFormat`] depending on `format`.

        Args:
            format (`str`):
                The format of the desired pipeline. Acceptable values are `"json"`, `"csv"` or `"pipe"`.
            output_path (`str`, *optional*):
                Where to save the outgoing data.
            input_path (`str`, *optional*):
                Where to look for the input data.
            column (`str`, *optional*):
                The column to read.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether or not to overwrite the `output_path`.

        Returns:
            [`~pipelines.PipelineDataFormat`]: The proper data format.
        """
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError(f"Unknown reader {format} (Available reader are json/csv/pipe)")


class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    def save(self, data: List[dict]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`List[dict]`): The data to store.
        """
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using JSON file format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

        with open(input_path, "r") as f:
            self._entries = json.load(f)

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    def save(self, data: dict):
        """
        Save the provided data object in a json file.

        Args:
            data (`dict`): The data to store.
        """
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process. For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __iter__(self):
        for line in sys.stdin:
            # Split for multi-columns
            if "\t" in line:
                line = line.split("\t")
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                yield line

    def save(self, data: dict):
        """
        Print the data.

        Args:
            data (`dict`): The data to store.
        """
        print(data)

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )

        return super().save_binary(data)


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()


class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance [`FeatureExtractionPipeline`] (`'feature-extraction'`) output large tensor object
    as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output`
    constructor argument. If set to `True`, the output will be stored in the pickle format.
    """

    default_input_names = None

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        image_processor: Optional['BaseImageProcessor'] = None,
        modelcard: Optional['ModelCard'] = None,
        task: str = "",
        ms_dtype: Optional[Union[str, "mindspore.common.dtype.Dtype"]] = None,
        binary_output: bool = False,
        **kwargs,
    ):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self.modelcard = modelcard
        self.ms_dtype = ms_dtype
        self.binary_output = binary_output

        # Update config and generation_config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))
            if self.model.can_generate():
                self.model.generation_config.update(**task_specific_params.get(task))

        self.call_count = 0
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = kwargs.pop("num_workers", None)
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)

        # if self.image_processor is None and self.feature_extractor is not None:
        #     if isinstance(self.feature_extractor, BaseImageProcessor):
        #         # Backward compatible change, if users called
        #         # ImageSegmentationPipeline(.., feature_extractor=MyFeatureExtractor())
        #         # then we should keep working
        #         self.image_processor = self.feature_extractor

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            safe_serialization (`str`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch or Tensorflow.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory, safe_serialization=safe_serialization)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)

        if self.image_processor is not None:
            self.image_processor.save_pretrained(save_directory)

        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (`List[str]` or `dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models_names = []
            for _, model_name in supported_models.items():
                # Mapping can now contain tuples of models for the same configuration.
                if isinstance(model_name, tuple):
                    supported_models_names.extend(list(model_name))
                else:
                    supported_models_names.append(model_name)
            if hasattr(supported_models, "_model_mapping"):
                for _, model in supported_models._model_mapping._extra_content.items():
                    if isinstance(model_name, tuple): # pylint: disable=undefined-loop-variable
                        supported_models_names.extend([m.__name__ for m in model])
                    else:
                        supported_models_names.append(model.__name__)
            supported_models = supported_models_names
        if self.model.__class__.__name__ not in supported_models:
            logger.error(
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are"
                f" {supported_models}."
            )

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        """
        _forward will receive the prepared dictionary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        raise NotImplementedError("postprocess not implemented")

    def forward(self, model_inputs, **forward_params):
        model_outputs = self._forward(model_inputs, **forward_params)
        return model_outputs

    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        if args:
            logger.warning(f"Ignoring args : {args}")

        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        self.call_count += 1
        if self.call_count > 10 and self.framework == "pt" and self.device.type == "cuda":
            warnings.warn(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a"
                " dataset",
                UserWarning,
            )

        is_dataset = isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)

        is_iterable = is_dataset or is_generator or is_list

        if is_list:
            return self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        elif is_iterable:
            return self.iterate(inputs, preprocess_params, forward_params, postprocess_params)
        else:
            return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        # This function should become `get_iterator` again, this is a temporary
        # easy solution.
        if isinstance(inputs, Dataset):
            for input_ in inputs.create_dict_iterator(output_numpy=True):
                yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)
        else:
            for input_ in inputs:
                yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)


class ChunkPipeline(Pipeline):
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        all_outputs = []
        for model_inputs in self.preprocess(inputs, **preprocess_params):
            model_outputs = self.forward(model_inputs, **forward_params)
            all_outputs.append(model_outputs)
        outputs = self.postprocess(all_outputs, **postprocess_params)
        return outputs


class PipelineRegistry:
    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        self.supported_tasks = supported_tasks
        self.task_aliases = task_aliases

    def get_supported_tasks(self) -> List[str]:
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()
        return supported_task

    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        if task in self.task_aliases:
            task = self.task_aliases[task]
        if task in self.supported_tasks:
            targeted_task = self.supported_tasks[task]
            return task, targeted_task, None

        if task.startswith("translation"):
            tokens = task.split("_")
            if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
                targeted_task = self.supported_tasks["translation"]
                task = "translation"
                return task, targeted_task, (tokens[1], tokens[3])
            raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")

        raise KeyError(
            f"Unknown task {task}, available tasks are {self.get_supported_tasks() + ['translation_XX_to_YY']}"
        )

    def register_pipeline(
        self,
        task: str,
        pipeline_class: type,
        model: Optional[Union[type, Tuple[type]]] = None,
        default: Optional[Dict] = None,
        type: Optional[str] = None,
    ) -> None:
        if task in self.supported_tasks:
            logger.warning(f"{task} is already registered. Overwriting pipeline for task {task}...")

        if model is None:
            model = ()
        elif not isinstance(model, tuple):
            model = (model,)

        task_impl = {"impl": pipeline_class, "ms": model}

        if default is not None:
            if "model" not in default and "ms" in default:
                default = {"model": default}
            task_impl["default"] = default

        if type is not None:
            task_impl["type"] = type

        self.supported_tasks[task] = task_impl
        pipeline_class._registered_impl = {task: task_impl}

    def to_dict(self):
        return self.supported_tasks
