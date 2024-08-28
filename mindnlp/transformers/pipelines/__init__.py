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
"""pipelines"""
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import requests
from addict import Dict as ADDict

from mindnlp.configs import HF_ENDPOINT
from mindnlp.utils import (
    is_offline_mode,
    logging,
)
from mindnlp.utils.peft_utils import find_adapter_config_file
from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from ..tokenization_utils import PreTrainedTokenizer

from .base import (
    # ArgumentHandler,
    CsvPipelineDataFormat,
    JsonPipelineDataFormat,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    # PipelineException,
    PipelineRegistry,
    get_default_model_and_revision,
    load_model,
)
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .text2text_generation import Text2TextGenerationPipeline
from .question_answering import QuestionAnsweringPipeline
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from .zero_shot_classification import ZeroShotClassificationArgumentHandler, ZeroShotClassificationPipeline
from .document_question_answering import DocumentQuestionAnsweringPipeline
from .fill_mask import FillMaskPipeline
from .table_question_answering import TableQuestionAnsweringPipeline

from ..models.auto.modeling_auto import (
    # AutoModel,
    # AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForMaskedLM,
    # AutoModelForMaskGeneration,
    # AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTableQuestionAnswering,
    # AutoModelForTableQuestionAnswering,
    # AutoModelForTextToSpectrogram,
    # AutoModelForTextToWaveform,
    # AutoModelForTokenClassification,
    # AutoModelForVideoClassification,
    # AutoModelForVision2Seq,
    # AutoModelForVisualQuestionAnswering,
    # AutoModelForZeroShotImageClassification,
    # AutoModelForZeroShotObjectDetection,
)


from ..modeling_utils import PreTrainedModel
from ..tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)


# Register all the supported tasks here
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
    "vqa": "visual-question-answering",
    "text-to-speech": "text-to-audio",
}

SUPPORTED_TASKS = {
    "automatic-speech-recognition": {
        "impl": AutomaticSpeechRecognitionPipeline,
        "ms": (AutoModelForCTC, AutoModelForSpeechSeq2Seq),
        "default": {"model": {"ms": ("facebook/wav2vec2-base-960h", "55bb623")}},
        "type": "multimodal",
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "ms": (AutoModelForSequenceClassification,),
        "default": {
            "model": {
                "ms": ("distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
            },
        },
        "type": "text",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "ms": (AutoModelForCausalLM,),
        "default": {
            "model": {
                "ms": ("gpt-2", "6c0e608"),
            },
        },
        "type": "text",
    },
    "text2text-generation": {
        "impl": Text2TextGenerationPipeline,
        "ms": (AutoModelForSeq2SeqLM,),
        "defult": {
            "model": {
                "ms": ("t5-base", "686f1db"),
            },
        },
        "type": "text",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "ms": (AutoModelForQuestionAnswering,),
        "default": {
            "model": {
                "ms": ("distilbert/distilbert-base-cased-distilled-squad", "626af31"),
            },
        },
        "type": "text",
    },
    "table-question-answering": {
        "impl": TableQuestionAnsweringPipeline,
        "ms": (AutoModelForTableQuestionAnswering,),
        "default": {
            "model": {
                "ms": ("google/tapas-base-finetuned-wtq", "69ceee2"),
            },
        },
        "type": "text",
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "ms": (AutoModelForSequenceClassification,),
        "default": {
            "model": {
                "ms": ("facebook/bart-large-mnli", "c626438"),
            },
            "config": {
                "ms": ("facebook/bart-large-mnli", "c626438"),
            },
        },
        "type": "text",
    },
    "document-question-answering": {
        "impl": DocumentQuestionAnsweringPipeline,
        "ms": (AutoModelForDocumentQuestionAnswering,),
        "default": {
            "model": {"ms": ("layoutlm-document-qa", "52e01b3")},
        },
        "type": "multimodal",
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "ms": (AutoModelForMaskedLM,),
        "default": {
            "model": {
                "ms": ("distilbert/distilroberta-base", "ec58a5b"),
            }
        },
        "type": "text",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_IMAGE_PROCESSOR_TASKS = set()
NO_TOKENIZER_TASKS = set()

# Those model configs are special, they are generic over their task, meaning
# any tokenizer/feature_extractor might be use for a given model so we cannot
# use the statically defined TOKENIZER_MAPPING and FEATURE_EXTRACTOR_MAPPING to
# see if the model defines such objects or not.
MULTI_MODEL_CONFIGS = {"SpeechEncoderDecoderConfig", "VisionEncoderDecoderConfig", "VisionTextDualEncoderConfig"}
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] in {"image", "video"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] in {"audio"}:
        NO_TOKENIZER_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return PIPELINE_REGISTRY.get_supported_tasks()

def model_info(
    repo_id: str,
    *,
    timeout: Optional[float] = None,
    securityStatus: Optional[bool] = None,
    files_metadata: bool = False,
):
    """
    This function retrieves information about a model from the specified repository.
    
    Args:
        repo_id (str): The identifier of the repository containing the model.
    
        timeout (Optional[float], optional): The maximum time (in seconds) to wait for the server to respond. Defaults to None.
        
        securityStatus (Optional[bool], optional): If True, includes security status information in the response. Defaults to None.
        
        files_metadata (bool, optional): If True, includes metadata about the model's files in the response. Defaults to False.
    
    Returns:
        None: This function does not return a value directly, but rather instantiates an ADDict object with the retrieved data.
    
    Raises:
        (requests.exceptions.RequestException): If a request error occurs.
        (json.JSONDecodeError): If the response is not valid JSON.
    """
    path = f"{HF_ENDPOINT}/api/models/{repo_id}"

    params = {}
    if securityStatus:
        params["securityStatus"] = True
    if files_metadata:
        params["blobs"] = True
    r = requests.get(path, timeout=timeout, params=params)
    data = r.json()
    return ADDict(**data)

def get_task(model: str) -> str:
    """
    This function retrieves the task associated with the input model.
    
    Args:
        model (str): The model for which the task needs to be retrieved.
    
    Returns:
        str: The task associated with the input model.
    
    Raises:
        RuntimeError: When attempting to infer task in offline mode.
        RuntimeError: When instantiating a pipeline without a task set raises an error.
        RuntimeError: When the model does not have a correct `pipeline_tag` set to infer the task automatically.
        RuntimeError: When the model is not meant to be used with transformers library.
    """
    if is_offline_mode():
        raise RuntimeError("You cannot infer task automatically within `pipeline` when using offline mode")
    try:
        info = model_info(model)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}") from e
    if not info.pipeline_tag:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    if getattr(info, "library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {info.library_name} not with transformers")
    task = info.pipeline_tag
    return task


def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"depth-estimation"`
            - `"document-question-answering"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"image-segmentation"`
            - `"image-to-text"`
            - `"image-to-image"`
            - `"object-detection"`
            - `"question-answering"`
            - `"summarization"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"` (alias `"sentiment-analysis"` available)
            - `"text-generation"`
            - `"text-to-audio"` (alias `"text-to-speech"` available)
            - `"token-classification"` (alias `"ner"` available)
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"video-classification"`
            - `"visual-question-answering"`
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`
            - `"zero-shot-object-detection"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) The normalized task name
        (removed alias and options). The actual dictionary required to initialize the pipeline and some extra task
        options for parametrized tasks like "translation_XX_to_YY"


    """
    return PIPELINE_REGISTRY.check_task(task)


def clean_custom_task(task_info):
    """
    This function cleans a custom task by performing the following steps:
    - Checks if the 'impl' key is present in the 'task_info' dictionary. If not, it raises a RuntimeError indicating that the model introduces a custom pipeline without specifying its implementation.
    - Retrieves the 'ms' key from the 'task_info' dictionary and converts it to a tuple if it is a string.
    - Retrieves the class names specified in the 'ms' key and fetches the corresponding classes from the 'transformers' module.
    - Updates the 'task_info' dictionary with the tuple of classes obtained from the 'ms' key.
    - Returns the updated 'task_info' dictionary and None.
    
    Args:
        task_info (dict): A dictionary containing information about the custom task.
            - The 'impl' key specifies the implementation of the custom pipeline.
            - The 'ms' key specifies the class names to be fetched from the 'transformers' module.
    
    Returns:
        tuple: A tuple containing the updated 'task_info' dictionary and None.
    
    Raises:
        RuntimeError: If the 'impl' key is not present in the 'task_info' dictionary.
    
    """
    from mindnlp import transformers

    if "impl" not in task_info:
        raise RuntimeError("This model introduces a custom pipeline without specifying its implementation.")
    ms_class_names = task_info.get("ms", ())
    if isinstance(ms_class_names, str):
        ms_class_names = [ms_class_names]
    task_info["ms"] = tuple(getattr(transformers, c) for c in ms_class_names)
    return task_info, None


def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[str] = None,
    use_fast: bool = True,
    ms_dtype=None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs,
) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    Pipelines are made of:

    - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
    - A [model](model) to make predictions from the inputs.
    - Some (optional) post processing for enhancing model's output.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
            - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
            - `"conversational"`: will return a [`ConversationalPipeline`].
            - `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
            - `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
            - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
            - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
            - `"image-classification"`: will return a [`ImageClassificationPipeline`].
            - `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
            - `"image-to-image"`: will return a [`ImageToImagePipeline`].
            - `"image-to-text"`: will return a [`ImageToTextPipeline`].
            - `"mask-generation"`: will return a [`MaskGenerationPipeline`].
            - `"object-detection"`: will return a [`ObjectDetectionPipeline`].
            - `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
            - `"summarization"`: will return a [`SummarizationPipeline`].
            - `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
            - `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a
              [`TextClassificationPipeline`].
            - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
            - `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
            - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
            - `"translation"`: will return a [`TranslationPipeline`].
            - `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
            - `"video-classification"`: will return a [`VideoClassificationPipeline`].
            - `"visual-question-answering"`: will return a [`VisualQuestionAnsweringPipeline`].
            - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
            - `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
            - `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
            - `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].

        model (`str` or [`PreTrainedModel`] or [`TFPreTrainedModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from [`PreTrainedModel`] (for PyTorch) or
            [`TFPreTrainedModel`] (for TensorFlow).

            If not provided, the default for the `task` will be loaded.
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].

            If not provided, the default configuration file for the requested model will be used. That means that if
            `model` is given, its default configuration will be used. However, if `model` is not supplied, this
            `task`'s default model's config is used instead.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from [`PreTrainedFeatureExtractor`].

            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.

            If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
            is a string). However, if `config` is also not given or not a string, then the default feature extractor
            for the given `task` will be loaded.

        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).

        ms_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).

        model_kwargs (`Dict[str, Any]`, *optional*):
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.

    Example:
        ```python
        >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        >>> # Sentiment analysis pipeline
        >>> analyzer = pipeline("sentiment-analysis")

        >>> # Question answering pipeline, specifying the checkpoint identifier
        >>> oracle = pipeline(
        ...     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
        ... )

        >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
        >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
        ```
    """
    if model_kwargs is None:
        model_kwargs = {}
    # Make sure we only pass use_auth_token once as a kwarg (it used to be possible to pass it in model_kwargs,
    # this is to keep BC).

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
            " path/identifier to a pretrained model when providing tokenizer."
        )
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided"
            " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class"
            " or a path/identifier to a pretrained model when providing feature_extractor."
        )
    if isinstance(model, Path):
        model = str(model)

    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(
            config, _from_pipeline=task, **model_kwargs
        )
    elif config is None and isinstance(model, str):
        # Check for an adapter file in the model path if PEFT is available
        # `find_adapter_config_file` doesn't accept `trust_remote_code`
        maybe_adapter_path = find_adapter_config_file(model)

        if maybe_adapter_path is not None:
            with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
                model = adapter_config["base_model_name_or_path"]

        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, **model_kwargs
        )
        # hub_kwargs["_commit_hash"] = config._commit_hash

    custom_tasks = {}
    if config is not None and len(getattr(config, "custom_pipelines", {})) > 0:
        custom_tasks = config.custom_pipelines
        if task is None:
            if len(custom_tasks) == 1:
                task = list(custom_tasks.keys())[0]
            else:
                raise RuntimeError(
                    "We can't infer the task automatically for this model as there are multiple tasks available. Pick "
                    f"one in {', '.join(custom_tasks.keys())}"
                )

    if task is None and model is not None:
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`. "
                f"{model} is not a valid model_id."
            )
        task = get_task(model)

    normalized_task, targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        # At that point framework might still be undetermined
        model, _ = get_default_model_and_revision(targeted_task, task_options)

        if config is None and isinstance(model, str):
            config = AutoConfig.from_pretrained(model, _from_pipeline=task, **model_kwargs)

    if ms_dtype is not None:
        if "ms_dtype" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... ms_dtype=..., model_kwargs={"ms_dtype":...})` as those'
                " arguments might conflict, use only one.)"
            )
        model_kwargs["ms_dtype"] = ms_dtype

    model_name = model if isinstance(model, str) else None

    # Load the correct model if possible
    if isinstance(model, str):
        model_classes = {"ms": targeted_task["ms"]}
        model = load_model(
            model,
            model_classes=model_classes,
            config=config,
            **model_kwargs,
        )

    model_config = model.config

    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None

    # If `model` (instance of `PretrainedModel` instead of `str`) is passed (and/or same for config), while
    # `image_processor` or `feature_extractor` is `None`, the loading will fail. This happens particularly for some
    # vision tasks when calling `pipeline()` with `model` and only one of the `image_processor` and `feature_extractor`.
    # TODO: we need to make `NO_IMAGE_PROCESSOR_TASKS` and `NO_FEATURE_EXTRACTOR_TASKS` more robust to avoid such issue.
    # This block is only temporarily to make CI green.
    if (
        tokenizer is None
        and not load_tokenizer
        and normalized_task not in NO_TOKENIZER_TASKS
        # Using class name to avoid importing the real class.
        and model_config.__class__.__name__ in MULTI_MODEL_CONFIGS
    ):
        # This is a special category of models, that are fusions of multiple models
        # so the model_config might not define a tokenizer, but it seems to be
        # necessary for the task, so we're force-trying to load it.
        load_tokenizer = True

    if task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False

    if load_tokenizer:
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer is None:
            if isinstance(model_name, str):
                tokenizer = model_name
            elif isinstance(config, str):
                tokenizer = config
            else:
                # Impossible to guess what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # Instantiate tokenizer if needed
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs.copy()
                tokenizer_kwargs.pop("ms_dtype", None)

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **tokenizer_kwargs
            )

    if task == "translation" and model.config.task_specific_params:
        for key in model.config.task_specific_params:
            if key.startswith("translation"):
                task = key
                warnings.warn(
                    f'"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{task}"',
                    UserWarning,
                )
                break

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    if ms_dtype is not None:
        kwargs["ms_dtype"] = ms_dtype

    if image_processor is not None:
        kwargs["image_processor"] = image_processor

    return pipeline_class(model=model, task=task, **kwargs)

__all__ = [
    'CsvPipelineDataFormat',
    'FillMaskPipeline',
    'JsonPipelineDataFormat',
    'PipedPipelineDataFormat',
    'Pipeline',
    'PipelineDataFormat',
    'TextClassificationPipeline',
    'Text2TextGenerationPipeline',
    'TextGenerationPipeline',
    'QuestionAnsweringPipeline',
    'ZeroShotClassificationPipeline',
    'DocumentQuestionAnsweringPipeline',
    'TableQuestionAnsweringPipeline',
    'pipeline',
]
