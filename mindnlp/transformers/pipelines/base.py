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

from ...utils import (
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
    """
    This function checks if the input list 'items' has a length of 1, raising a ValueError if not. 
    It is intended to be used with a batch size of 1.

    Args:
        items (list): A list of items to be checked for length.

    Returns:
        None.

    Raises:
        ValueError: If the length of 'items' is not equal to 1.
    """
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


def _pad(items, key, padding_value, padding_side):
    """
    Args:
        items (list): A list of dictionaries representing items, where each dictionary contains
            the key-value pair for the specified key.
        key (str): The key within each dictionary representing the data to be padded.
        padding_value (int): The value used for padding the data.
        padding_side (str): The side on which padding should be applied, either 'left' or 'right'.
    
    Returns:
        None
    
    Raises:
        None
    """
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
    """
    This function takes in two parameters, tokenizer and feature_extractor, and returns None.
    
    Args:
        tokenizer (object): An optional tokenizer object. If provided, it should have a pad_token_id attribute.
            If not provided, the function expects a feature_extractor object.
        feature_extractor (object): An optional feature_extractor object.
            If provided, it can have padding_value and padding_side attributes.
    
    Returns:
        None
    
    Raises:
        ValueError: If both tokenizer and feature_extractor are not provided.
        ValueError: If tokenizer is provided but does not have a pad_token_id attribute.
        ValueError: If feature_extractor is provided but does not have padding_value and padding_side attributes.
        ValueError: If tokenizer and feature_extractor have different padding sides.
        ValueError: If the elements of the batch contain different keys.
    """
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
           "ms" or None, representing a specific framework if it was specified, or None if we don't know yet.

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
        """
        Initializes a new instance of PipelineException.

        Args:
            self (object): The instance of the class.
            task (str): The task for which the exception occurred.
            model (str): The model related to the exception.
            reason (str): The reason for the exception.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each [`~pipelines.Pipeline`].
    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        This method is an abstract method that should be implemented in a subclass.
        It is intended to serve as a generic callable interface for argument handling.

        Args:
            self (ArgumentHandler): The instance of the ArgumentHandler class.

        Returns:
            None:
                This method does not return any value.

        Raises:
            NotImplementedError:
                This exception is raised when the method is not implemented in a subclass.
        """
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
        """Initializes an instance of the PipelineDataFormat class.

        Args:
            output_path (Optional[str]): The path to the output file. Defaults to None.
            input_path (Optional[str]): The path to the input file. Defaults to None.
            column (Optional[str]): The column(s) to use for data processing. Defaults to None.
                If multiple columns are provided, they should be comma-separated.
                Each column can be specified as 'name' or 'name=value' to map input and output columns.
            overwrite (bool, optional): Determines whether to overwrite the output file if it already exists.
                Defaults to False.

        Returns:
            None:
                This method does not return a value.

        Raises:
            OSError: If the output_path is provided and the overwrite parameter is False,
                and the output_path already exists on disk.
            OSError: If the input_path is provided and the input_path does not exist on disk.
        """
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
        """
        This method '__iter__' in the class 'PipelineDataFormat' is used to define an iterator for instances of the class.

        Args:
            self: An instance of the 'PipelineDataFormat' class.

        Returns:
            None:
                This method does not return any value explicitly
                but is meant to be implemented by subclasses to return an iterator.

        Raises:
            NotImplementedError:
                This exception is raised if the method is not implemented by a subclass.
                It serves as a reminder for the subclass to implement its own iteration logic.
        """
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
        """
        Initializes an instance of the CsvPipelineDataFormat class.

        Args:
            output_path (Optional[str]): The path to the output file. If specified, the processed data will be written to this file.
            input_path (Optional[str]): The path to the input file. If specified, the data will be read from this file.
            column (Optional[str]): The name of the column to process. If specified, only the data in this column will be processed.
            overwrite (bool, optional): Indicates whether the output file should be overwritten if it already exists. Defaults to False.

        Returns:
            None.

        Raises:
            None:
                However, this method may raise exceptions if the input or output file paths are invalid or
                if there are any issues during the data processing.

        Note:
            - The 'output_path', 'input_path', and 'column' parameters are optional. They can be left empty or set to
            None if not required.
            - The 'overwrite' parameter is optional and defaults to False.
        """
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        """
        Iterates over the rows of a CSV file and yields the specified columns as a dictionary.

        Args:
            self: An instance of the CsvPipelineDataFormat class.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the specified input file path does not exist.
            csv.Error: If there are issues with reading the CSV file.
            IndexError: If the column index is out of range.
            KeyError: If the column key is not found in the row dictionary.
            TypeError: If the column parameter is not a valid type.
            ValueError: If the column parameter is not properly formatted.

        Note:
            - The CSV file is read using the 'r' mode.
            - The CSV file is expected to have a header row.
            - If self.is_multi_columns is True, the method yields a dictionary with keys from the specified column list
            and values from the corresponding columns in the CSV file.
            - If self.is_multi_columns is False, the method yields the value from the specified column index in each row.

        Example:
            ```python
            >>> data_format = CsvPipelineDataFormat()
            >>> data_format.input_path = 'data.csv'
            >>> data_format.is_multi_columns = True
            >>> for row in data_format:
            >>>     print(row)
            ...
            >>> Output:
            >>> {'col1': 'value1', 'col2': 'value2'}
            >>> {'col1': 'value3', 'col2': 'value4'}
            ...
            ```
        """
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
        """
        Initializes a JsonPipelineDataFormat object.

        Args:
            self: The instance of the class.
            output_path (Optional[str]): The path to the output file where the processed data will be saved.
            input_path (Optional[str]): The path to the input file containing the data to be processed.
            column (Optional[str]): The column in the input data to be processed.
            overwrite (bool): Indicates whether to overwrite the existing output file if it already exists.
                Default is False.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the input file specified by 'input_path' does not exist.
            json.JSONDecodeError: If the input file does not contain valid JSON data.
            IOError: If there is an issue with reading the input file.
        """
        super().__init__(output_path, input_path, column, overwrite=overwrite)

        with open(input_path, "r") as f:
            self._entries = json.load(f)

    def __iter__(self):
        """
        Iterates over the entries of the JsonPipelineDataFormat object.

        Args:
            self (JsonPipelineDataFormat): The JsonPipelineDataFormat object itself.

        Returns:
            None

        Raises:
            None

        This method iterates over the entries stored in the JsonPipelineDataFormat object and yields each entry
        as a dictionary. If the JsonPipelineDataFormat object is configured with multiple columns, each yielded entry
        is a dictionary where the keys correspond to the column names and the values are the values of the respective
        columns for that entry. If the JsonPipelineDataFormat object is not configured with multiple columns,
        each yielded entry is a single value corresponding to the first column specified in the 'column' attribute of
        the JsonPipelineDataFormat object.
        """
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
        '''
        Iterates over input lines from the standard input and yields formatted data.

        Args:
            self (PipedPipelineDataFormat): An instance of the PipedPipelineDataFormat class.

        Returns:
            None

        Raises:
            None

        Description:
            This method is used to iterate over input lines read from the standard input.
            Each line is checked for the presence of a tab character ('\t').
            If a tab character is found, the line is split using the tab character as the delimiter.
            If the PipedPipelineDataFormat instance has a defined column attribute,
            a dictionary is yielded containing key-value pairs where the keys are the column names
            and the values are extracted from the corresponding line elements.
            If the column attribute is not defined, a tuple containing the line elements is yielded.
            If a line does not contain a tab character, the entire line is yielded as is.
        '''
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
        """
        Save binary data to an output file path.

        Args:
            self (PipedPipelineDataFormat): An instance of the PipedPipelineDataFormat class.
            data (Union[dict, List[dict]]): The binary data to be saved. It can be either a single dictionary or
                a list of dictionaries.

        Returns:
            str: The output file path where the binary data was saved.

        Raises:
            KeyError: If the `output_path` attribute of `self` is None, indicating that an output file path is required
                when using piped input on pipeline outputting large objects. The error message will
                prompt the user to provide the output path through the `--output` argument.

        """
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
        """
        This method named 'transform' in the class '_ScikitCompat' is an abstract method that must be implemented by subclasses.

        Args:
            self: Represents the instance of the class. It is used to access and modify class attributes.
            X: Input data to be transformed. It can be in the form of a list, array, or dataframe.
                No specific restrictions.

        Returns:
            None: This method does not return any value. It is meant to transform the input data in place.

        Raises:
            NotImplementedError:
                This exception is raised when the method is called directly without being implemented in a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        """
        Predicts the target values for the input data.

        Args:
            self (_ScikitCompat): The instance of the _ScikitCompat class.
            X (array-like): The input data to make predictions on.
                It should be a 2D array-like object where each row represents a sample and each column represents a feature.

        Returns:
            None: This method does not return any value explicitly but should update the internal state or
                perform computations for making predictions.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass that inherits from _ScikitCompat.
        """
        raise NotImplementedError()


class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

    `Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output`

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance [`FeatureExtractionPipeline`] (`'feature-extraction'`) output large tensor object
    as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output`
    forwardor argument. If set to `True`, the output will be stored in the pickle format.
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
        """
        Initializes a new instance of the Pipeline class.

        Args:
            model (PreTrainedModel): The pre-trained model to be used in the pipeline.
            tokenizer (Optional[PreTrainedTokenizer]): An optional pre-trained tokenizer for processing input data.
            feature_extractor (Optional[PreTrainedFeatureExtractor]):
                An optional feature extractor for extracting features from the input data.
            image_processor (Optional[BaseImageProcessor]): An optional image processor for handling image data.
            modelcard (Optional[ModelCard]): An optional model card containing information about the model.
            task (str): The task that the pipeline is designed to perform.
            ms_dtype (Optional[Union[str, mindspore.common.dtype.Dtype]]): An optional data type for MindSpore computations.
            binary_output (bool): A flag indicating whether the output should be binary.
            **kwargs: Additional keyword arguments for configuring the pipeline.

        Returns:
            None.

        Raises:
            None.
        """
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
        """
        This method performs the forward pass of the pipeline model.

        Args:
            self (Pipeline): The instance of the Pipeline class.
            model_inputs: The inputs to the model for the forward pass.
                Type can vary depending on the model architecture and input requirements.

        Returns:
            None: This method returns None as it directly returns the model outputs.

        Raises:
            None:
                However, the _forward method it calls may raise exceptions based on the model's implementation.
        """
        model_outputs = self._forward(model_inputs, **forward_params)
        return model_outputs

    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        """
        Performs the main processing logic for the Pipeline class.

        Args:
            self (Pipeline): The instance of the Pipeline class.
            inputs: The input data for processing. It can be a Dataset, GeneratorType, or list.

        Returns:
            None: This method does not return any value.

        Raises:
            UserWarning:
                If the method is called more than 10 times,
                a warning is raised to prompt the user to use a dataset for efficiency.
        """
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
        if self.call_count > 10:
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
        """
        Method that runs a series of input items through the pipeline.

        Args:
            self (Pipeline): The instance of the Pipeline class.
            inputs (list): A list of input items to be processed by the pipeline.
            preprocess_params (dict): Parameters for preprocessing the input items.
            forward_params (dict): Parameters for the forward pass through the pipeline.
            postprocess_params (dict): Parameters for postprocessing the output items.

        Returns:
            None: This method does not return any value but processes the input items through the pipeline.

        Raises:
            None.
        """
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        """
        This method 'run_single' is a member of the 'Pipeline' class and is responsible for executing a single run of the pipeline.

        Args:
            self (object): The instance of the Pipeline class.
            inputs (object): The input data to be processed by the pipeline.
            preprocess_params (dict): Parameters for the preprocessing step, used to configure the preprocessing behavior.
            forward_params (dict): Parameters for the forward step, used to configure the forward pass behavior.
            postprocess_params (dict): Parameters for the postprocessing step, used to configure the postprocessing behavior.

        Returns:
            None.

        Raises:
            Any exceptions that is:
                raised by the 'preprocess', 'forward', or 'postprocess' methods called within this method
                will be propagated to the caller.
        """
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        """
        Iterates through the input data and yields the result of running each input through the pipeline.

        Args:
            self (Pipeline): The instance of the Pipeline class.
            inputs (Union[Dataset, List[Any]]): The input data to iterate over.

                - If inputs is a Dataset object, it will be iterated over by creating a dictionary iterator.
                - If inputs is a list of inputs, each input will be iterated over individually.
            preprocess_params (Any): The parameters used for preprocessing the input data.
            forward_params (Any): The parameters used for the forward pass of the pipeline.
            postprocess_params (Any): The parameters used for postprocessing the output data.

        Returns:
            None.

        Raises:
            None.
        """
        # This function should become `get_iterator` again, this is a temporary
        # easy solution.
        if isinstance(inputs, Dataset):
            for input_ in inputs.create_dict_iterator(output_numpy=True):
                yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)
        else:
            for input_ in inputs:
                yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)


class ChunkPipeline(Pipeline):

    """
    ChunkPipeline is a class that represents a pipeline for chunk processing. It inherits from the Pipeline class.

    The ChunkPipeline class provides a method called run_single, which takes inputs, preprocess_params, forward_params,
    and postprocess_params as arguments. It performs chunk processing on the inputs using the specified parameters.

    The run_single method internally calls the preprocess, forward, and postprocess methods to process the inputs.
    It preprocesses the inputs using the preprocess_params, performs forward processing using the forward_params,
    and finally postprocesses the outputs using the postprocess_params.

    The preprocess method takes the inputs and preprocess_params as arguments and returns a generator that yields model_inputs.
    It splits the inputs into chunks and applies preprocessing operations based on the preprocess_params.

    The forward method takes model_inputs and forward_params as arguments and returns the model_outputs.
    It applies forward processing operations on the model_inputs based on the forward_params.

    The postprocess method takes all_outputs and postprocess_params as arguments and returns the outputs.
    It applies postprocessing operations on the all_outputs based on the postprocess_params.

    The run_single method collects all the model_outputs generated during the chunk processing and returns the final outputs.

    Example:
        ```python
        >>> chunk_pipeline = ChunkPipeline()
        >>> inputs = [input_1, input_2, input_3]
        >>> preprocess_params = {'param1': value1, 'param2': value2}
        >>> forward_params = {'param3': value3, 'param4': value4}
        >>> postprocess_params = {'param5': value5, 'param6': value6}
        >>> outputs = chunk_pipeline.run_single(inputs, preprocess_params, forward_params, postprocess_params)
        ```

    Note:
        - The ChunkPipeline class should be instantiated and the run_single method should be called to perform chunk
        processing.
        - The preprocess_params, forward_params, and postprocess_params should be provided as dictionaries with the
        required parameters for each step of chunk processing.
    """
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        """
        This method 'run_single' is part of the 'ChunkPipeline' class and is used to process inputs through a pipeline
        of preprocessing, forward pass, and postprocessing.

        Args:
            self (object): The instance of the ChunkPipeline class.
            inputs (object): The input data to be processed through the pipeline.
            preprocess_params (dict): A dictionary containing parameters for the preprocessing step.
            forward_params (dict): A dictionary containing parameters for the forward pass step.
            postprocess_params (dict): A dictionary containing parameters for the postprocessing step.

        Returns:
            None: This method does not return a value explicitly, but it processes the inputs through the pipeline
                and updates the internal state of the ChunkPipeline instance.

        Raises:
            None: However, exceptions may be raised within
                the preprocess, forward, or postprocess steps, which are not explicitly documented here.
        """
        all_outputs = []
        for model_inputs in self.preprocess(inputs, **preprocess_params):
            model_outputs = self.forward(model_inputs, **forward_params)
            all_outputs.append(model_outputs)
        outputs = self.postprocess(all_outputs, **postprocess_params)
        return outputs


class PipelineRegistry:

    """
    The PipelineRegistry class represents a registry for managing supported tasks and their corresponding pipelines.
    It provides methods for registering pipelines, checking tasks, and converting the registry to a dictionary.
    The class maintains a dictionary of supported tasks and their respective aliases, and also allows for the
    registration of pipelines for specific tasks.

    Attributes:
        supported_tasks (Dict[str, Any]): A dictionary containing the supported tasks and their associated pipeline
            implementations and models.
        task_aliases (Dict[str, str]): A dictionary containing task aliases for supported tasks.

    Methods:
        __init__:
            Initializes the PipelineRegistry instance with the provided supported tasks and task aliases.

        get_supported_tasks:
            Retrieves a sorted list of supported tasks, including aliases.

        check_task:
            Checks if a given task is supported and returns the task name, its targeted task, and additional
            parameters if applicable.

        register_pipeline:
            Registers a pipeline for a specific task, and optionally specifies the pipeline class, model, d
            efault parameters, and type.

        to_dict:
            Converts the PipelineRegistry instance to a dictionary representation.

    Note:
        - Task aliases are also considered for supported tasks.
        - If a task is already registered, registering a new pipeline for the same task will overwrite the
        existing pipeline.
        - The 'translation_XX_to_YY' format is expected for certain translation tasks.
    """
    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        """
        Initializes an instance of the PipelineRegistry class.

        Args:
            self: The instance of the class.
            supported_tasks (Dict[str, Any]): A dictionary containing the supported tasks as keys and their
                corresponding values.
            task_aliases (Dict[str, str]): A dictionary mapping task aliases to their actual task names.

        Returns:
            None.

        Raises:
            None.
        """
        self.supported_tasks = supported_tasks
        self.task_aliases = task_aliases

    def get_supported_tasks(self) -> List[str]:
        """
        Retrieve a list of supported tasks from the PipelineRegistry.

        Args:
            self (PipelineRegistry): The instance of PipelineRegistry class.
                This parameter is required to access the supported tasks and task aliases.

        Returns:
            List[str]: A sorted list of supported tasks available in the PipelineRegistry.

        Raises:
            None:
                This method does not raise any specific exceptions.
        """
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()
        return supported_task

    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        """
        Checks if a given task is supported by the PipelineRegistry.

        Args:
            self (PipelineRegistry): The instance of the PipelineRegistry class.
            task (str): The task to be checked.

        Returns:
            Tuple[str, Dict, Any]:
                A tuple containing the following:

                - task (str): The modified task name after resolving aliases (if any).
                - targeted_task (Dict): The dictionary representing the targeted task.
                - translation_params (Any): Additional parameters for translation tasks (if applicable).

        Raises:
            KeyError: If the provided task is not found in the supported tasks or aliases.
            KeyError: If the provided translation task is not in the correct format 'translation_XX_to_YY'.

        Note:
            - If the task is found in the task_aliases dictionary, it will be replaced with the corresponding value.
            - If the task is found in the supported_tasks dictionary, targeted_task will be set to the corresponding value.
            - If the task starts with 'translation' and follows the format 'translation_XX_to_YY', targeted_task will be set
              to the value corresponding to 'translation' and translation_params will be a tuple containing source and target
              languages (XX and YY respectively).
            - The available tasks include the supported tasks and the translation tasks in the format 'translation_XX_to_YY'.
        """
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
        """
        Register a pipeline in the PipelineRegistry.

        Args:
            self: The instance of the PipelineRegistry class.
            task (str): The task for which the pipeline is being registered.
            pipeline_class (type): The class implementing the pipeline.
            model (Optional[Union[type, Tuple[type]]]): The model or models associated with the pipeline.
                Defaults to None. If a single model is passed, it should be of type 'type'.
                If multiple models are passed, they should be provided as a tuple of 'type'.
            default (Optional[Dict]): The default configuration for the pipeline.
                Defaults to None. If provided, it should be a dictionary with 'model' or 'ms' as keys.
                If 'model' is not present but 'ms' is, it will be wrapped in a dictionary with 'model' as key.
            type (Optional[str]): The type of the pipeline. Defaults to None.

        Returns:
            None.

        Raises:
            None.

        Note:
            - If the 'task' is already registered, the existing pipeline for that task will be overwritten.
            - If no 'model' is provided, it will default to an empty tuple.
            - The 'model' parameter can be a single model or a tuple of multiple models.
            - The 'default' parameter, if provided, should be a dictionary with 'model' or 'ms' as keys.
              If 'model' is not present but 'ms' is, it will be wrapped in a dictionary with 'model' as key.
            - The 'type' parameter is optional and can be used to specify the type of the pipeline.
            - The 'supported_tasks' attribute of the PipelineRegistry instance will be updated with the registered
            pipeline.
            - The '_registered_impl' attribute of the pipeline_class will be updated with the registered task and
            its implementation.
        """
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
        """
        Converts the PipelineRegistry object into a dictionary representation.
        
        Args:
            self (PipelineRegistry): The current instance of the PipelineRegistry class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.supported_tasks
